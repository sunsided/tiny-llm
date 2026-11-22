//! `tiny`: ask the TINY network questions, in a full-screen neon interface or
//! as plain text (automatic when input or output is redirected).

#![forbid(unsafe_code)]

mod plain;
mod sound;
mod tui;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use tiny::model::{LoadError, Network, Weights};
use tiny::think::Creativity;

/// Command-line options.
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent command-line switches"
)]
#[derive(Debug, Parser)]
#[command(version, about = "TINY 2.1 - a neural network that answers questions")]
struct Cli {
    /// 0 always gives the surest answer, 5 words answers differently each time, 9 is wildest.
    #[arg(default_value_t = 5, value_parser = clap::value_parser!(u8).range(0..=9))]
    creativity: Creativity,

    /// Weight file to use instead of the built-in TINY.BIN.
    #[arg(long, value_name = "FILE")]
    model: Option<PathBuf>,

    /// Plain text, no full screen.
    #[arg(short, long)]
    plain: bool,

    /// TINY asks itself sample questions; press a key to take over.
    #[arg(long)]
    demo: bool,

    /// Grey colours, for monochrome screens.
    #[arg(long, conflicts_with = "classic")]
    mono: bool,

    /// Standard terminal colours instead of the neon palette.
    #[arg(long)]
    classic: bool,

    /// No sound at all.
    #[arg(short, long)]
    quiet: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let net = match load(cli.model.as_deref()) {
        Ok(net) => net,
        Err(err) => {
            eprintln!("TINY: {err}");
            return ExitCode::FAILURE;
        }
    };

    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let result = if cli.plain || !interactive {
        plain::run(net, cli.creativity)
    } else {
        tui::run(net, tui::Options::from(&cli))
    };

    if let Err(err) = result {
        eprintln!("TINY: {err}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

/// Loads the network from `path`, or the built-in one.
fn load(path: Option<&std::path::Path>) -> Result<Network, LoadError> {
    let weights = match path {
        Some(path) => Weights::load(path)?,
        None => Weights::from_bytes(tiny::BUILTIN_MODEL)?,
    };

    Ok(Network::new(weights))
}
