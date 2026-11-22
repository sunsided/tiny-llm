//! Plain text mode: a prompt on stdin, answers on stdout, same wording as `TINY.C`.

use std::io::{self, BufRead, Write};
use std::time::Instant;

use tiny::model::Network;
use tiny::text::{clean, is_word};
use tiny::think::{Creativity, Event, Thinking};

/// Runs the question loop until end of input or QUIT.
///
/// # Errors
///
/// Fails when stdin or stdout fails.
pub fn run(mut net: Network, creativity: Creativity) -> io::Result<()> {
    let config = net.config();
    let mut out = io::stdout().lock();
    let mut rng = rand::rng();

    writeln!(out, "TINY 2 - a neural network that answers questions.")?;
    writeln!(
        out,
        "Ready: {} layers, {} wide. Creativity {creativity} (tiny 0 to tiny 9).",
        config.layers, config.width
    )?;
    writeln!(
        out,
        "Ask about countries, animals, space, the body, nature, science, food,"
    )?;
    writeln!(
        out,
        "sums, numbers, spelling, words, shapes, history, music or computers."
    )?;
    writeln!(out, "Type QUIT to stop.")?;

    for line in io::stdin().lock().lines() {
        let q = clean(&line?);
        if q.is_empty() {
            continue;
        }

        if is_word(&q, "quit") || is_word(&q, "exit") {
            break;
        }

        write!(out, "\n> {q}\n  ")?;
        let started = Instant::now();
        let mut thinking = Thinking::new(&q);
        let mut letters = 0;
        loop {
            match thinking.advance(&mut net, creativity, &mut rng) {
                Event::Read { .. } => {}
                Event::Letter { letter, .. } => {
                    letters += 1;
                    write!(out, "{letter}")?;
                    out.flush()?;
                }
                Event::Done => break,
            }
        }

        let seconds = started.elapsed().as_secs_f32();
        writeln!(out, "\n   [{letters} letters in {seconds:.3} seconds]")?;
    }

    writeln!(out, "Goodbye!")
}
