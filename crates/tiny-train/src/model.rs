//! The trainable TINY network in candle: the same pre-norm transformer as
//! `train2.py`'s `Model`, with variables named like its PyTorch state dict.
//!
//! ```mermaid
//! flowchart LR
//!     idx[letters B×T] --> emb["emb + pos"]
//!     emb --> blk["block × L: x + attn(norm x), then x + f2(relu(f1(norm x)))"]
//!     blk --> nf["norm"] --> head["× embᵀ"] --> logits[B×T×VOCAB]
//! ```

use std::collections::BTreeMap;

use anyhow::{Context, ensure};
use candle_core::{D, DType, Device, Tensor, Var};
use candle_nn::{Init, VarMap};
use tiny::model::{CTX, Config, VOCAB};

use crate::checkpoint::Checkpoint;

/// Added to the mean square before the root in every norm.
const NORM_EPS: f64 = 1e-5;

/// The network's variables, shared with the optimizer through a [`VarMap`].
pub struct Model {
    /// Shape of the network.
    pub config: Config,
    /// Every trainable variable, by state-dict name.
    pub vars: VarMap,
    /// Letter embedding `VOCAB × D`, also the output head.
    emb: Tensor,
    /// Position embedding `CTX × D`.
    pos: Tensor,
    /// The transformer blocks.
    blocks: Vec<Block>,
    /// Final norm gain.
    nf: Tensor,
}

/// Weights of one transformer block.
#[derive(Debug)]
struct Block {
    /// Norm gain before attention.
    n1: Tensor,
    /// Query projection `D × D` (PyTorch layout: out × in).
    q: Tensor,
    /// Key projection.
    k: Tensor,
    /// Value projection.
    v: Tensor,
    /// Attention output projection.
    o: Tensor,
    /// Norm gain before the feed-forward part.
    n2: Tensor,
    /// Feed-forward expansion `HID × D`.
    f1: Tensor,
    /// Feed-forward contraction `D × HID`.
    f2: Tensor,
}

impl Model {
    /// A freshly initialised network, like `train2.Model.__init__`: embeddings
    /// N(0, 0.05), projections N(0, 0.5/√in), gains 1.
    ///
    /// # Errors
    ///
    /// Fails when candle cannot allocate a tensor.
    pub fn new(config: Config, device: &Device) -> anyhow::Result<Self> {
        let vars = VarMap::new();
        let builder = Builder {
            vars: &vars,
            device,
        };
        let (d, hid) = (config.width, config.hidden);

        let emb = builder.normal("emb.weight", (VOCAB, d), 0.05)?;
        let pos = builder.normal("pos.weight", (CTX, d), 0.05)?;
        let blocks = (0..config.layers)
            .map(|l| Block::new(&builder, &format!("blocks.{l}."), d, hid))
            .collect::<anyhow::Result<_>>()?;
        let nf = builder.ones("nf.g", d)?;

        Ok(Self {
            config,
            vars,
            emb,
            pos,
            blocks,
            nf,
        })
    }

    /// A network holding the weights of `checkpoint`.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint's tensor names or shapes differ from the model's.
    ///
    /// # Panics
    ///
    /// Panics if the variable map's lock is poisoned, which only a panic elsewhere can cause.
    pub fn from_checkpoint(checkpoint: &Checkpoint, device: &Device) -> anyhow::Result<Self> {
        let mut model = Self::new(checkpoint.config, device)?;
        // `set_one` copies into the variables in place, so the tensors held by the model follow.
        let names: Vec<String> = model
            .vars
            .data()
            .lock()
            .expect("var map")
            .keys()
            .cloned()
            .collect();
        ensure!(
            names.len() == checkpoint.tensors.len(),
            "checkpoint has {} tensors, the model {}",
            checkpoint.tensors.len(),
            names.len()
        );

        for name in &names {
            let tensor = checkpoint.tensor(name)?.to_device(device)?;
            model
                .vars
                .set_one(name, tensor)
                .with_context(|| format!("setting {name}"))?;
        }

        Ok(model)
    }

    /// The current weights as a checkpoint (for export or saving).
    ///
    /// # Errors
    ///
    /// Fails when a variable cannot be copied.
    ///
    /// # Panics
    ///
    /// Panics if the variable map's lock is poisoned, which only a panic elsewhere can cause.
    pub fn to_checkpoint(&self) -> anyhow::Result<Checkpoint> {
        let data = self.vars.data().lock().expect("var map");
        let tensors: BTreeMap<String, Tensor> = data
            .iter()
            .map(|(name, var)| Ok((name.clone(), var.as_tensor().copy()?)))
            .collect::<candle_core::Result<_>>()?;

        Ok(Checkpoint {
            config: self.config,
            tensors,
        })
    }

    /// Matrices (weight decay 0.01) and gains (no decay), like the two optimizer groups of `train2.py`.
    #[must_use]
    pub fn parameter_groups(&self) -> (Vec<Var>, Vec<Var>) {
        self.vars
            .all_vars()
            .into_iter()
            .partition(|var| var.rank() == 2)
    }

    /// Logits `B × T × VOCAB` for letter tokens `idx` (`B × T`, `u32`, `T ≤ CTX`).
    ///
    /// # Errors
    ///
    /// Fails on a shape mismatch.
    pub fn forward(&self, idx: &Tensor) -> anyhow::Result<Tensor> {
        let (_, t) = idx.dims2()?;
        let device = idx.device();
        let positions = Tensor::arange(0u32, u32::try_from(t)?, device)?;
        let mut x = self
            .emb
            .index_select(&idx.flatten_all()?, 0)?
            .reshape((idx.dim(0)?, t, self.config.width))?
            .broadcast_add(&self.pos.index_select(&positions, 0)?)?;
        let mask = causal_mask(t, device)?;

        for block in &self.blocks {
            x = block.forward(&x, &mask, self.config.heads)?;
        }

        Ok(norm(&x, &self.nf)?.broadcast_matmul(&self.emb.t()?)?)
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Block {
    /// Creates the variables of block `prefix` with their initial values.
    fn new(builder: &Builder<'_>, prefix: &str, d: usize, hid: usize) -> anyhow::Result<Self> {
        let linear = |name: &str, out: usize, inp: usize| {
            builder.normal(
                &format!("{prefix}{name}.weight"),
                (out, inp),
                0.5 / f64_of(inp).sqrt(),
            )
        };

        Ok(Self {
            n1: builder.ones(&format!("{prefix}n1.g"), d)?,
            q: linear("q", d, d)?,
            k: linear("k", d, d)?,
            v: linear("v", d, d)?,
            o: linear("o", d, d)?,
            n2: builder.ones(&format!("{prefix}n2.g"), d)?,
            f1: linear("f1", hid, d)?,
            f2: linear("f2", d, hid)?,
        })
    }

    /// `x + attention(norm(x))`, then `x + f2(relu(f1(norm(x))))`.
    fn forward(&self, x: &Tensor, mask: &Tensor, heads: usize) -> anyhow::Result<Tensor> {
        let h = norm(x, &self.n1)?;
        let x = (x + linear(&self.attention(&h, mask, heads)?, &self.o)?)?;
        let h = linear(&norm(&x, &self.n2)?, &self.f1)?.relu()?;

        Ok((&x + linear(&h, &self.f2)?)?)
    }

    /// Causal multi-head self-attention of `h` (`B × T × D`), before the output projection.
    fn attention(&self, h: &Tensor, mask: &Tensor, heads: usize) -> anyhow::Result<Tensor> {
        let (batch, len, width) = h.dims3()?;
        let head_width = width / heads;
        let split = |w: &Tensor| -> anyhow::Result<Tensor> {
            Ok(linear(h, w)?
                .reshape((batch, len, heads, head_width))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let (q, k, v) = (split(&self.q)?, split(&self.k)?, split(&self.v)?);

        let scores = (q.matmul(&k.t()?)? / f64_of(head_width).sqrt())?.broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;

        Ok(weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, len, width))?)
    }
}

/// Creates variables in a [`VarMap`].
struct Builder<'a> {
    /// Where the variables are registered.
    vars: &'a VarMap,
    /// Where they live.
    device: &'a Device,
}

impl Builder<'_> {
    /// A variable drawn from N(0, `std`).
    fn normal(&self, name: &str, shape: (usize, usize), std: f64) -> anyhow::Result<Tensor> {
        let init = Init::Randn {
            mean: 0.0,
            stdev: std,
        };
        Ok(self.vars.get(shape, name, init, DType::F32, self.device)?)
    }

    /// A gain vector of ones.
    fn ones(&self, name: &str, width: usize) -> anyhow::Result<Tensor> {
        Ok(self
            .vars
            .get(width, name, candle_nn::init::ONE, DType::F32, self.device)?)
    }
}

/// `x · W^T`, a bias-free PyTorch `Linear`.
fn linear(x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
    x.broadcast_matmul(&w.t()?)
}

/// RMS norm over the last dimension, times the gains.
fn norm(x: &Tensor, gain: &Tensor) -> candle_core::Result<Tensor> {
    let scale = (x.sqr()?.mean_keepdim(D::Minus1)? + NORM_EPS)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&scale)?.broadcast_mul(gain)
}

/// `T × T` additive mask: 0 on and below the diagonal, −∞ above it.
fn causal_mask(t: usize, device: &Device) -> candle_core::Result<Tensor> {
    let allowed = Tensor::tril2(t, DType::U8, device)?;
    let zeros = Tensor::zeros((t, t), DType::F32, device)?;
    let blocked = Tensor::full(f32::NEG_INFINITY, (t, t), device)?;
    allowed.where_cond(&zeros, &blocked)
}

/// A small count as `f64`.
fn f64_of(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}
