//! The burn model with the released weights computes what `tiny`'s 8-bit
//! inference computes, up to the 8-bit key/value cache of the latter.

use std::path::Path;

use burn::tensor::{Device, Int, Tensor, TensorData};
use tiny::model::{Network, Weights};
use tiny::text::{NEWLINE, token};
use tiny_train::checkpoint::{Array, Checkpoint};
use tiny_train::export::{quantise_row, to_bin};
use tiny_train::model::Model;

/// A question and its answer as the model sees them.
const SEQUENCE: &str = "what is the capital of peru\nThe capital of Peru is Lima.\n";

/// Largest logit difference allowed between the two implementations.
const TOLERANCE: f32 = 1.0;

/// `tiny2.pt` with every matrix rounded to 8 bits like the export, so both sides share weights.
fn quantised_checkpoint() -> Checkpoint {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/tiny2.pt");
    let mut checkpoint = Checkpoint::load_pt(&path).expect("tiny2.pt");
    for array in checkpoint.tensors.values_mut() {
        if let [_, cols] = array.shape[..] {
            let data = array
                .data
                .chunks_exact(cols)
                .flat_map(|row| {
                    let (scale, q) = quantise_row(row);
                    q.into_iter().map(move |w| f32::from(w) * scale)
                })
                .collect();
            *array = Array {
                shape: array.shape.clone(),
                data,
            };
        }
    }

    checkpoint
}

/// Letter tokens of [`SEQUENCE`].
fn tokens() -> Vec<usize> {
    SEQUENCE
        .bytes()
        .map(|c| if c == b'\n' { NEWLINE } else { token(c) })
        .collect()
}

/// Index of the highest score and its lead over the runner-up.
fn best_and_lead(scores: &[f32]) -> (usize, f32) {
    let best = scores
        .iter()
        .enumerate()
        .fold(0, |b, (i, s)| if *s > scores[b] { i } else { b });
    let second = scores
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != best)
        .fold(f32::NEG_INFINITY, |m, (_, &s)| m.max(s));

    (best, scores[best] - second)
}

/// Logits agree position by position within the error of `tiny`'s 8-bit
/// key/value cache, and the best letter is the same wherever `tiny`'s choice
/// is clear (a lead of more than twice that error).
#[test]
fn burn_logits_match_tiny_inference() {
    let device = Device::flex();
    let checkpoint = quantised_checkpoint();
    let model = Model::from_checkpoint(&checkpoint, &device).expect("model");
    let mut net =
        Network::new(Weights::from_bytes(&to_bin(&checkpoint).expect("bin")).expect("weights"));
    let tokens = tokens();

    let ids: Vec<i64> = tokens
        .iter()
        .map(|&t| i64::try_from(t).expect("token"))
        .collect();
    let input = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, tokens.len()]), &device);
    let logits = model
        .forward(input)
        .into_data()
        .try_to_vec::<f32>()
        .expect("logits");

    for (pos, &tok) in tokens.iter().enumerate() {
        let scores = &net.step(tok, pos).scores;
        let row = &logits[pos * scores.len()..][..scores.len()];
        let diff = row
            .iter()
            .zip(scores)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < TOLERANCE, "logits differ by {diff} at {pos}");

        let (best, lead) = best_and_lead(scores);
        if lead > 2.0 * TOLERANCE {
            assert_eq!(best_and_lead(row).0, best, "best letter at {pos}");
        }
    }
}

/// Every tensor of the checkpoint is used, and round-trips unchanged.
#[test]
fn checkpoint_round_trips() {
    let checkpoint = quantised_checkpoint();
    let model = Model::from_checkpoint(&checkpoint, &Device::flex()).expect("model");

    assert_eq!(
        to_bin(&model.to_checkpoint()).expect("bin"),
        to_bin(&checkpoint).expect("bin")
    );
}
