//! The shipped network scored on the frozen test set of the original kit.

use std::path::Path;

use tiny::model::{Network, Weights};
use tiny_train::eval::{evaluate, load};

/// The released `TINY.BIN`, run through the 8-bit inference of `tiny`, scores
/// close to the 61% the original README reports for unseen questions.
#[test]
fn released_network_scores_like_the_original() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval.jsonl");
    let items = load(&path).expect("fixture");
    let mut net = Network::new(Weights::from_bytes(tiny::BUILTIN_MODEL).expect("model"));
    let score = evaluate(&mut net, &items);

    println!("eval {:.1}%  {}", 100.0 * score.total(), score.line());
    assert_eq!(items.len(), 780);
    assert!(score.total() > 0.55, "{:.3}", score.total());
}
