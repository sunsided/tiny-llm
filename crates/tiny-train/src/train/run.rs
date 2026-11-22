//! The time-boxed training loop of `train2.train`: AdamW in two groups,
//! warm-up plus schedule, clipped gradients, a smoothed loss log and
//! optional evaluations along the way.
//!
//! ```mermaid
//! flowchart LR
//!     pool[batch pool] --> loss --> grads[backward + clip] --> opt[AdamW: matrices / gains]
//!     opt --> pool
//!     opt -- every N minutes --> eval[save + evaluate]
//! ```

use std::time::Instant;

use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use super::{Batch, Example, Schedule, clip_gradients, loss};
use crate::model::Model;

/// Examples generated per batch slot when the pool refills; sorting that many
/// by length keeps padding low.
const POOL_FACTOR: usize = 32;

/// Weight decay of the matrices; gains get none.
const WEIGHT_DECAY: f64 = 0.01;

/// Settings of one training run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// Learning-rate schedule, including the run's length in seconds.
    pub schedule: Schedule,
    /// Examples per batch.
    pub batch_size: usize,
    /// Print the smoothed loss every this many steps.
    pub log_every: u32,
    /// Evaluate every this many seconds, if set.
    pub eval_every: Option<f64>,
    /// Seed of the batch shuffling.
    pub seed: u64,
}

/// Hands out batches of similar length, refilling from a sample source like `train2.batch`.
struct Pool<S> {
    /// Makes one training example.
    source: S,
    /// Batches left, popped from the end.
    batches: Vec<Vec<Example>>,
    /// Examples per batch.
    size: usize,
    /// Shuffles the batch order.
    rng: StdRng,
}

impl<S: FnMut() -> Example> Pool<S> {
    /// The next batch, generating and sorting a fresh pool when empty.
    fn next(&mut self) -> Vec<Example> {
        if self.batches.is_empty() {
            let mut examples: Vec<Example> = (0..self.size * POOL_FACTOR)
                .map(|_| (self.source)())
                .collect();
            examples.sort_by_key(|e| e.tokens.len());
            self.batches = examples
                .chunks(self.size)
                .map(<[Example]>::to_vec)
                .collect();
            self.batches.shuffle(&mut self.rng);
        }

        self.batches.pop().unwrap_or_default()
    }
}

/// Trains `model` in place until the schedule's time is up; returns the number of steps.
///
/// `source` makes training examples. `on_eval` runs every `options.eval_every`
/// seconds and once at the end, with the elapsed seconds.
///
/// # Errors
///
/// Fails when candle fails, or when `on_eval` does.
pub fn train(
    model: &Model,
    options: &Options,
    source: impl FnMut() -> Example,
    mut on_eval: impl FnMut(&Model, f64) -> anyhow::Result<()>,
) -> anyhow::Result<u32> {
    let mut pool = Pool {
        source,
        batches: Vec::new(),
        size: options.batch_size,
        rng: StdRng::seed_from_u64(options.seed),
    };
    let mut optimizer = Optimizers::new(model)?;
    let started = Instant::now();
    let mut log = LossLog::default();
    let mut next_eval = options.eval_every;
    let mut step = 0u32;

    loop {
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed > options.schedule.total {
            break;
        }

        if next_eval.is_some_and(|at| elapsed > at) {
            on_eval(model, elapsed)?;
            next_eval = next_eval
                .zip(options.eval_every)
                .map(|(at, every)| at + every);
        }

        let batch = Batch::new(&pool.next(), model.device())?;
        let rate = options.schedule.rate(step, elapsed);
        log.add(optimizer.step(model, &batch, rate)?);
        step += 1;
        if step.is_multiple_of(options.log_every) {
            println!(
                "step {step:5}  {elapsed:4.0}s  lr {rate:.5}  loss {:.3}",
                log.smoothed
            );
        }
    }

    on_eval(model, started.elapsed().as_secs_f64())?;
    Ok(step)
}

/// One `AdamW` per parameter group, as `train2.py` builds them.
struct Optimizers {
    /// The matrices, with weight decay.
    matrices: AdamW,
    /// The gains, without.
    gains: AdamW,
    /// All variables, for gradient clipping.
    vars: Vec<candle_core::Var>,
}

impl Optimizers {
    /// Optimizers over `model`'s two parameter groups (betas 0.9 and 0.98).
    fn new(model: &Model) -> anyhow::Result<Self> {
        let (matrices, gains) = model.parameter_groups();
        let params = |weight_decay| ParamsAdamW {
            lr: 0.0,
            beta1: 0.9,
            beta2: 0.98,
            eps: 1e-8,
            weight_decay,
        };

        Ok(Self {
            vars: model.vars.all_vars(),
            matrices: AdamW::new(matrices, params(WEIGHT_DECAY))?,
            gains: AdamW::new(gains, params(0.0))?,
        })
    }

    /// One step at learning rate `rate`; returns the batch loss before the step.
    fn step(&mut self, model: &Model, batch: &Batch, rate: f64) -> anyhow::Result<f32> {
        let loss = loss(model, batch)?;
        let mut grads = loss.backward()?;
        clip_gradients(&mut grads, &self.vars, 1.0)?;

        for optimizer in [&mut self.matrices, &mut self.gains] {
            optimizer.set_learning_rate(rate);
            optimizer.step(&grads)?;
        }

        Ok(loss.to_scalar::<f32>()?)
    }
}

/// Exponentially smoothed loss for the log (0.98 old, 0.02 new).
#[derive(Debug, Default)]
struct LossLog {
    /// The smoothed value; the first loss seen starts it.
    smoothed: f32,
    /// Whether any loss was seen.
    started: bool,
}

impl LossLog {
    /// Takes in one batch loss.
    fn add(&mut self, loss: f32) {
        self.smoothed = if self.started {
            0.98 * self.smoothed + 0.02 * loss
        } else {
            loss
        };
        self.started = true;
    }
}
