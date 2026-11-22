//! `tiny-train`: evaluate and export TINY networks, and (once the generator
//! is in) train them, like the original `train2.py`.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};
use tiny::model::{Network, Weights};
use tiny_train::checkpoint::Checkpoint;
use tiny_train::eval::{Item, Score, evaluate, load};
use tiny_train::export::to_bin;

/// Command-line interface.
#[derive(Debug, Parser)]
#[command(version, about = "Train, evaluate and export TINY networks")]
struct Cli {
    /// What to do.
    #[command(subcommand)]
    command: Command,
}

/// The subcommands.
#[derive(Debug, Subcommand)]
enum Command {
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

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
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
