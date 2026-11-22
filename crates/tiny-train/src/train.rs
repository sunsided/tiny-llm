//! The pieces of `train2.py`'s training loop: encoding question/answer pairs,
//! batching, the loss on answer letters only, the learning-rate schedule and
//! gradient clipping.

mod run;

pub use run::{Options, train};

use burn::module::ParamId;
use burn::optim::GradientsParams;
use burn::tensor::activation::log_softmax;
use burn::tensor::{Device, Int, Tensor, TensorData};
use tiny::model::CTX;
use tiny::text::{NEWLINE, token};

use crate::model::Model;

/// One training example: letter tokens and which of them the loss counts.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    /// `question \n answer \n` as tokens, at most [`CTX`] long.
    pub tokens: Vec<u32>,
    /// 0 for the question and its newline, 1 for the answer and its newline.
    pub mask: Vec<f32>,
}

impl Example {
    /// Encodes a question/answer pair like `train2.batch`.
    ///
    /// # Panics
    ///
    /// Panics if the text contains a byte outside printable ASCII; the generator never makes one.
    #[must_use]
    pub fn new(question: &str, answer: &str) -> Self {
        let text = format!("{question}\n{answer}\n");
        let tokens: Vec<u32> = text
            .bytes()
            .map(|c| if c == b'\n' { NEWLINE } else { token(c) })
            .map(|t| u32::try_from(t).expect("token fits"))
            .take(CTX)
            .collect();
        let prompt = (question.len() + 1).min(tokens.len());
        let mask = (0..tokens.len())
            .map(|i| if i < prompt { 0.0 } else { 1.0 })
            .collect();

        Self { tokens, mask }
    }
}

/// Batches are padded to a multiple of this many letters. GPU backends compile
/// kernels per shape, so few distinct lengths keep that cost small; the extra
/// padding is masked out of the loss and, sitting after every real letter,
/// never changes what the causal model computes for them.
pub const LENGTH_STEP: usize = 16;

/// A padded batch: tokens `B × T` (padding 0) and loss weights `B × T`.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Letter tokens.
    pub tokens: Tensor<2, Int>,
    /// 1 where the loss counts, 0 elsewhere (question letters and padding).
    pub mask: Tensor<2>,
}

impl Batch {
    /// Pads `examples` to the longest one, rounded up to [`LENGTH_STEP`] (at most [`CTX`]).
    #[must_use]
    pub fn new(examples: &[Example], device: &Device) -> Self {
        let longest = examples.iter().map(|e| e.tokens.len()).max().unwrap_or(0);
        let len = longest.next_multiple_of(LENGTH_STEP).clamp(2, CTX);
        let mut tokens = vec![0i64; examples.len() * len];
        let mut mask = vec![0f32; examples.len() * len];
        for (i, example) in examples.iter().enumerate() {
            for (slot, &t) in tokens[i * len..].iter_mut().zip(&example.tokens) {
                *slot = i64::from(t);
            }

            mask[i * len..][..example.mask.len()].copy_from_slice(&example.mask);
        }

        let shape = [examples.len(), len];
        Self {
            tokens: Tensor::from_data(TensorData::new(tokens, shape), device),
            mask: Tensor::from_data(TensorData::new(mask, shape), device),
        }
    }
}

/// Mean cross-entropy of predicting each next letter, over answer letters only (`train2.loss_fn`).
#[must_use]
pub fn loss(model: &Model, batch: &Batch) -> Tensor<1> {
    let [rows, len] = batch.tokens.dims();
    let input = batch.tokens.clone().slice([0..rows, 0..len - 1]);
    let target = batch.tokens.clone().slice([0..rows, 1..len]);
    let weight = batch.mask.clone().slice([0..rows, 1..len]);

    let log_probs = log_softmax(model.forward(input), 2);
    let picked = (log_probs * crate::model::one_hot(target))
        .sum_dim(2)
        .squeeze_dim::<2>(2);

    (picked.neg() * weight.clone()).sum() / weight.sum()
}

/// The learning rate over a time-boxed run (`train2.train`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schedule {
    /// Highest learning rate.
    pub peak: f64,
    /// Length of the run in seconds.
    pub total: f64,
    /// `None`: cosine decay to 3% of the peak. `Some(f)`: constant, then linear
    /// decay to zero over the last fraction `f` of the run.
    pub decay: Option<f64>,
}

/// Steps of linear warm-up at the start of every run.
const WARMUP_STEPS: f64 = 200.0;

impl Schedule {
    /// Learning rate at optimizer step `step` (0-based) and `elapsed` seconds.
    #[must_use]
    pub fn rate(&self, step: u32, elapsed: f64) -> f64 {
        let warm = ((f64::from(step) + 1.0) / WARMUP_STEPS).min(1.0);
        let shape = match self.decay {
            None => 0.03 + 0.97 * 0.5 * (1.0 + (std::f64::consts::PI * elapsed / self.total).cos()),
            Some(fraction) => ((self.total - elapsed) / (fraction * self.total)).clamp(0.0, 1.0),
        };

        self.peak * warm * shape
    }
}

/// Scales all gradients down so their global norm is at most `max_norm`
/// (`torch.nn.utils.clip_grad_norm_`); returns the norm before clipping.
///
/// `matrices` and `gains` are the ids of the rank-2 and rank-1 parameters.
pub fn clip_gradients(
    grads: &mut GradientsParams,
    matrices: &[ParamId],
    gains: &[ParamId],
    max_norm: f64,
) -> f64 {
    let mut squares: Option<Tensor<1>> = None;
    let mut add =
        |part: Tensor<1>| squares = Some(squares.take().map_or(part.clone(), |s| s + part));
    for &id in matrices {
        if let Some(g) = grads.get::<2>(id) {
            add(g.powi_scalar(2).sum());
        }
    }
    for &id in gains {
        if let Some(g) = grads.get::<1>(id) {
            add(g.powi_scalar(2).sum());
        }
    }

    let norm = squares.map_or(0.0, |s| f64::from(s.into_scalar::<f32>()).sqrt());
    let factor = max_norm / (norm + 1e-6);
    if factor < 1.0 {
        for &id in matrices {
            if let Some(g) = grads.remove::<2>(id) {
                grads.register(id, g * factor);
            }
        }
        for &id in gains {
            if let Some(g) = grads.remove::<1>(id) {
                grads.register(id, g * factor);
            }
        }
    }

    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The question and its newline are masked out, the answer and its newline count.
    #[test]
    fn mask_covers_the_answer() {
        let example = Example::new("hi", "Yo.");

        assert_eq!(example.tokens.len(), 7);
        assert_eq!(example.mask, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    }

    /// Warm-up, then cosine down to 3% at the end; linear decay hits zero.
    #[test]
    fn schedules() {
        let cosine = Schedule {
            peak: 1.0,
            total: 100.0,
            decay: None,
        };
        let linear = Schedule {
            peak: 1.0,
            total: 100.0,
            decay: Some(0.3),
        };

        assert!((cosine.rate(0, 0.0) - 1.0 / 200.0).abs() < 1e-12);
        assert!((cosine.rate(1000, 0.0) - 1.0).abs() < 1e-12);
        assert!((cosine.rate(1000, 100.0) - 0.03).abs() < 1e-12);
        assert!((linear.rate(1000, 50.0) - 1.0).abs() < 1e-12);
        assert!(linear.rate(1000, 100.0).abs() < 1e-12);
    }
}
