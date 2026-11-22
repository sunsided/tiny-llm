//! Exporting the original full-precision checkpoint must reproduce the
//! released `TINY.BIN` byte for byte, as the original README promises.

use std::path::Path;

use tiny_train::checkpoint::Checkpoint;
use tiny_train::export::to_bin;

/// `tiny2.pt` exported is exactly `crates/tiny/assets/TINY.BIN`.
#[test]
fn exporting_tiny2_gives_the_released_bin() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let checkpoint = Checkpoint::load_pt(&root.join("data/tiny2.pt")).expect("tiny2.pt loads");
    let bin = to_bin(&checkpoint).expect("export");

    assert_eq!(bin.len(), tiny::BUILTIN_MODEL.len());
    let first_difference = bin
        .iter()
        .zip(tiny::BUILTIN_MODEL)
        .position(|(a, b)| a != b);
    assert_eq!(first_difference, None);
}
