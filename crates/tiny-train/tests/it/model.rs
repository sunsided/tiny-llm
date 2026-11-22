//! The candle model with the released weights computes what `tiny`'s 8-bit
//! inference computes, up to the 8-bit key/value cache of the latter.

use std::path::Path;

use candle_core::{D, Device, Tensor};
use tiny::model::{Network, Weights};
use tiny::text::{NEWLINE, token};
use tiny_train::checkpoint::Checkpoint;
use tiny_train::export::{quantise_row, to_bin};
use tiny_train::model::Model;

/// A question and its answer as the model sees them.
const SEQUENCE: &str = "what is the capital of peru\nThe capital of Peru is Lima.\n";

/// `tiny2.pt` with every matrix rounded to 8 bits like the export, so both sides share weights.
fn quantised_checkpoint() -> Checkpoint {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/tiny2.pt");
    let mut checkpoint = Checkpoint::load_pt(&path).expect("tiny2.pt");
    for tensor in checkpoint.tensors.values_mut() {
        if tensor.rank() == 2 {
            let rows: Vec<Vec<f32>> = tensor
                .to_vec2::<f32>()
                .expect("matrix")
                .iter()
                .map(|row| {
                    let (scale, q) = quantise_row(row);
                    q.iter().map(|&w| f32::from(w) * scale).collect()
                })
                .collect();
            *tensor = Tensor::new(rows, &Device::Cpu).expect("tensor");
        }
    }

    checkpoint
}

/// Letter tokens of [`SEQUENCE`].
fn tokens() -> Vec<u32> {
    SEQUENCE
        .bytes()
        .map(|c| if c == b'\n' { NEWLINE } else { token(c) })
        .map(|t| u32::try_from(t).expect("token"))
        .collect()
}

/// Logits agree position by position within the error of `tiny`'s 8-bit
/// key/value cache, and the best letter is the same wherever `tiny`'s choice
/// is clear (a lead of more than twice that error).
#[test]
fn candle_logits_match_tiny_inference() {
    const TOLERANCE: f32 = 1.0;
    let checkpoint = quantised_checkpoint();
    let model = Model::from_checkpoint(&checkpoint, &Device::Cpu).expect("model");
    let mut net =
        Network::new(Weights::from_bytes(&to_bin(&checkpoint).expect("bin")).expect("weights"));
    let tokens = tokens();

    let input = Tensor::new(tokens.as_slice(), &Device::Cpu)
        .expect("input")
        .unsqueeze(0)
        .expect("batch");
    let logits = model
        .forward(&input)
        .expect("forward")
        .squeeze(0)
        .expect("unbatch");
    let candle_best = logits
        .argmax(D::Minus1)
        .expect("argmax")
        .to_vec1::<u32>()
        .expect("best");
    let candle_rows = logits.to_vec2::<f32>().expect("rows");

    for (pos, &tok) in tokens.iter().enumerate() {
        let scores = &net.step(usize::try_from(tok).expect("token"), pos).scores;
        let diff = candle_rows[pos]
            .iter()
            .zip(scores)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < TOLERANCE, "logits differ by {diff} at {pos}");

        let (best, lead) = best_and_lead(scores);
        if lead > 2.0 * TOLERANCE {
            assert_eq!(
                usize::try_from(candle_best[pos]).expect("index"),
                best,
                "best letter at {pos}"
            );
        }
    }
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

/// Every tensor of the checkpoint is used, and round-trips unchanged.
#[test]
fn checkpoint_round_trips() {
    let checkpoint = quantised_checkpoint();
    let model = Model::from_checkpoint(&checkpoint, &Device::Cpu).expect("model");
    let back = model.to_checkpoint().expect("checkpoint");

    assert_eq!(
        to_bin(&back).expect("bin"),
        to_bin(&checkpoint).expect("bin")
    );
}
