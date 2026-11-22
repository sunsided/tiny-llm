//! `tiny-train`: generate questions, train, evaluate and export TINY
//! networks, like the original `gen2.py` and `train2.py`.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::tensor::Device;
use clap::{Args, Parser, Subcommand};
use tiny::model::{Config, Network, Weights};
use tiny_train::checkpoint::Checkpoint;
use tiny_train::eval::{Item, Score, evaluate, load};
use tiny_train::export::to_bin;
use tiny_train::model::Model;
use tiny_train::pyrandom::PyRandom;
use tiny_train::questions::{KINDS, Kit};
use tiny_train::train::{Example, Options, Schedule, train};

/// The network shape `train2.py` trains from scratch: 96 wide, 4 layers, 4 heads, 256 hidden.
const DEFAULT_CONFIG: Config = Config {
    width: 96,
    layers: 4,
    heads: 4,
    hidden: 256,
};

/// Command-line interface.
#[derive(Debug, Parser)]
#[command(
    version,
    about = "Generate questions, train, evaluate and export TINY networks"
)]
struct Cli {
    /// What to do.
    #[command(subcommand)]
    command: Command,
}

/// The subcommands.
#[derive(Debug, Subcommand)]
enum Command {
    /// Show the generator's mix of question kinds and a sample of each.
    Gen {
        /// Seed of the generator.
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Train for a number of minutes, saving a `.safetensors` checkpoint.
    Train(TrainArgs),
    /// Score a network on the frozen 780-question test set (8-bit, as TINY runs it).
    Eval {
        /// A TINY.BIN file, a train2.py `.pt` checkpoint or a `.safetensors` checkpoint.
        model: PathBuf,
        /// Print this many wrong answers.
        #[arg(long, default_value_t = 20)]
        show: usize,
    },
    /// Write a checkpoint as a TINY.BIN file.
    Export {
        /// A `.pt` or `.safetensors` checkpoint.
        checkpoint: PathBuf,
        /// The TINY.BIN file to write.
        out: PathBuf,
    },
}

/// Options of `tiny-train train` (the flags of `train2.py train`).
#[derive(Debug, Args)]
struct TrainArgs {
    /// Length of the run in minutes.
    minutes: f64,
    /// The `.safetensors` checkpoint to write (also after every evaluation).
    out: PathBuf,
    /// Continue from this checkpoint (`.pt` or `.safetensors`) instead of starting fresh.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Peak learning rate.
    #[arg(long, default_value_t = 2e-3)]
    lr: f64,
    /// Constant rate, then linear decay over this last fraction of the run (default: cosine).
    #[arg(long)]
    decay: Option<f64>,
    /// Evaluate (and save) every this many minutes.
    #[arg(long)]
    eval_every: Option<f64>,
    /// Examples per batch.
    #[arg(long, default_value_t = 64)]
    batch: usize,
    /// Seed of the generator and the batch order (default: the current time).
    #[arg(long)]
    seed: Option<u64>,
    /// Where to train: `cpu` (pure-Rust Flex backend), `cuda` (build with
    /// `--features cuda`) or `wgpu` (build with `--features wgpu`).
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Gen { seed } => show_samples(&kit()?, seed),
        Command::Train(args) => run_training(&args)?,
        Command::Eval { model, show } => {
            let score = evaluate_file(&model)?;
            print_misses(&score, show);
            println!("eval {:.1}%  {}", 100.0 * score.total(), score.line());
        }
        Command::Export { checkpoint, out } => {
            let bin = to_bin(&Checkpoint::load(&checkpoint)?)?;
            std::fs::write(&out, &bin).with_context(|| format!("writing {}", out.display()))?;
            println!("wrote {} bytes", bin.len());
        }
    }

    Ok(())
}

/// Trains as `args` say, saving and evaluating along the way and at the end.
fn run_training(args: &TrainArgs) -> anyhow::Result<()> {
    let kit = kit()?;
    let tests = test_set()?;
    let device = device(&args.device)?.autodiff();
    let model = match &args.resume {
        Some(path) => Model::from_checkpoint(&Checkpoint::load(path)?, &device)?,
        None => Model::new(DEFAULT_CONFIG, &device),
    };
    let seed = args.seed.unwrap_or_else(time_seed);
    let options = Options {
        schedule: Schedule {
            peak: args.lr,
            total: args.minutes * 60.0,
            decay: args.decay,
        },
        batch_size: args.batch,
        log_every: 200,
        eval_every: args.eval_every.map(|minutes| minutes * 60.0),
        seed,
    };
    let config = model.config();
    println!(
        "model {config:?}, {} parameters, seed {seed}",
        config.parameter_count()
    );

    let mut rng = PyRandom::new(seed);
    let source = || {
        let sample = kit.sample(&mut rng, false, None);
        Example::new(&sample.question, &sample.answer)
    };
    let on_eval = |model: &Model, elapsed: f64| -> anyhow::Result<()> {
        let checkpoint = model.to_checkpoint();
        checkpoint.save(&args.out)?;
        let score = evaluate(
            &mut Network::new(Weights::from_bytes(&to_bin(&checkpoint)?)?),
            &tests,
        );
        println!(
            "  EVAL at {:.0} min: {:.1}%  {}",
            elapsed / 60.0,
            100.0 * score.total(),
            score.line()
        );
        Ok(())
    };

    let (_, steps) = train(model, &options, source, on_eval)?;
    println!("saved {}, {steps} steps", args.out.display());
    Ok(())
}

/// Prints the mix of kinds over 20,000 samples and one training and one test sample per kind.
fn show_samples(kit: &Kit, seed: u64) {
    let mut rng = PyRandom::new(seed);
    let mut counts = vec![0usize; KINDS.len()];
    let mut total_length = 0;
    for _ in 0..20_000 {
        let sample = kit.sample(&mut rng, false, None);
        total_length += sample.question.len() + sample.answer.len() + 2;
        if let Some(i) = KINDS.iter().position(|(name, _)| *name == sample.kind) {
            counts[i] += 1;
        }
    }

    let mix: Vec<String> = KINDS
        .iter()
        .zip(&counts)
        .map(|((name, _), n)| format!("{name} {n}"))
        .collect();
    println!("mix: {}", mix.join(", "));
    println!("mean length {:.1}", count(total_length) / 20_000.0);
    for (name, _) in KINDS {
        for test in [false, true] {
            let sample = kit.sample(&mut rng, test, Some(name));
            let label = if test { "TEST" } else { "" };
            println!(
                "  {name:<10} {label:<5} {:<52} -> {}",
                sample.question, sample.answer
            );
        }
    }
}

/// The compute device named on the command line.
fn device(name: &str) -> anyhow::Result<Device> {
    match name {
        "cpu" => Ok(Device::flex()),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::cuda(0)),
        #[cfg(feature = "wgpu")]
        "wgpu" => Ok(Device::wgpu(Default::default())),
        "cuda" | "wgpu" => anyhow::bail!("build with `--features {name}` to train on {name}"),
        other => anyhow::bail!("unknown device {other}; use cpu, cuda or wgpu"),
    }
}

/// The fact files bundled with this crate.
fn kit() -> anyhow::Result<Kit> {
    Kit::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("data/facts"))
}

/// Scores a TINY.BIN file or a checkpoint (exported to 8 bits first).
fn evaluate_file(path: &Path) -> anyhow::Result<Score> {
    let is_bin = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"));
    let bytes = if is_bin {
        std::fs::read(path).with_context(|| format!("reading {}", path.display()))?
    } else {
        to_bin(&Checkpoint::load(path)?)?
    };

    Ok(evaluate(
        &mut Network::new(Weights::from_bytes(&bytes)?),
        &test_set()?,
    ))
}

/// The frozen test set of the original kit.
fn test_set() -> anyhow::Result<Vec<Item>> {
    load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval.jsonl"))
}

/// Prints up to `count` wrong answers like the original eval output.
fn print_misses(score: &Score, count: usize) {
    for (kind, question, answer, wanted) in score.misses.iter().take(count) {
        println!("   x [{kind}] {question} -> {answer}   (want: {wanted})");
    }
}

/// Seconds since the Unix epoch, as `train2.py` seeds its generator.
fn time_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// A count as `f64` for averages.
fn count(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}
