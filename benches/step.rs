//! Benchmarks of the network's hot path: one letter through all layers.

use std::hint::black_box;

use criterion::Criterion;
use tiny::model::{Network, Weights};
use tiny::text::{clean, token};

/// A typical question, long enough to fill the attention cache a bit.
const QUESTION: &str = "what is the capital of peru";

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_step(&mut criterion);
    bench_read_question(&mut criterion);
    criterion.final_summary();
}

/// The released network.
fn network() -> Network {
    Network::new(Weights::from_bytes(tiny::BUILTIN_MODEL).expect("built-in model"))
}

/// One step at position 40, with 40 earlier letters in the cache.
fn bench_step(criterion: &mut Criterion) {
    let mut net = network();
    for pos in 0..40 {
        net.step(token(b'a'), pos);
    }

    criterion.bench_function("step at position 40", |b| {
        b.iter(|| net.step(black_box(token(b'e')), black_box(40)).scores[0]);
    });
}

/// Reading a whole question from position 0.
fn bench_read_question(criterion: &mut Criterion) {
    let mut net = network();
    let tokens: Vec<usize> = clean(QUESTION).bytes().map(token).collect();

    criterion.bench_function("read a 27-letter question", |b| {
        b.iter(|| {
            for (pos, &tok) in tokens.iter().enumerate() {
                net.step(black_box(tok), pos);
            }
        });
    });
}
