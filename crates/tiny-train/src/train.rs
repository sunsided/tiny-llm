//! The pieces of `train2.py`'s training loop: encoding question/answer pairs,
//! batching, the loss on answer letters only, the learning-rate schedule and
//! gradient clipping.

mod run;

pub use run::{Options, train};

use candle_core::backprop::GradStore;
use candle_core::{D, DType, Device, Tensor, Var};
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

/// A padded batch: tokens `B × T` (`u32`, padding 0) and loss weights `B × T`.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Letter tokens.
    pub tokens: Tensor,
    /// 1 where the loss counts, 0 elsewhere (question letters and padding).
    pub mask: Tensor,
}

impl Batch {
    /// Pads `examples` to the longest one.
    ///
    /// # Errors
    ///
    /// Fails when candle cannot build the tensors.
    pub fn new(examples: &[Example], device: &Device) -> anyhow::Result<Self> {
        let len = examples.iter().map(|e| e.tokens.len()).max().unwrap_or(0);
        let mut tokens = vec![0u32; examples.len() * len];
        let mut mask = vec![0f32; examples.len() * len];
        for (i, example) in examples.iter().enumerate() {
            tokens[i * len..][..example.tokens.len()].copy_from_slice(&example.tokens);
            mask[i * len..][..example.mask.len()].copy_from_slice(&example.mask);
        }

        let shape = (examples.len(), len);
        Ok(Self {
            tokens: Tensor::from_vec(tokens, shape, device)?,
            mask: Tensor::from_vec(mask, shape, device)?,
        })
    }
}

/// Mean cross-entropy of predicting each next letter, over answer letters only (`train2.loss_fn`).
///
/// # Errors
///
/// Fails on a shape mismatch.
pub fn loss(model: &Model, batch: &Batch) -> anyhow::Result<Tensor> {
    let len = batch.tokens.dim(1)?;
    let input = batch.tokens.narrow(1, 0, len - 1)?.contiguous()?;
    let target = batch.tokens.narrow(1, 1, len - 1)?.contiguous()?;
    let weight = batch.mask.narrow(1, 1, len - 1)?;

    let log_probs = candle_nn::ops::log_softmax(&model.forward(&input)?, D::Minus1)?;
    let picked = (log_probs * crate::model::one_hot(&target)?)?.sum(D::Minus1)?;
    let total = (picked.neg()? * &weight)?.sum_all()?;

    Ok((total
        / weight
            .sum_all()?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()
            .map(f64::from)?)?)
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
/// # Errors
///
/// Fails when candle cannot compute on the gradients.
pub fn clip_gradients(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> anyhow::Result<f64> {
    let mut squares = 0.0f64;
    for var in vars {
        if let Some(g) = grads.get(var) {
            squares += f64::from(g.sqr()?.sum_all()?.to_scalar::<f32>()?);
        }
    }

    let norm = squares.sqrt();
    let factor = max_norm / (norm + 1e-6);
    if factor < 1.0 {
        for var in vars {
            if let Some(g) = grads.remove(var) {
                grads.insert(var, (g * factor)?);
            }
        }
    }

    Ok(norm)
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
