//! The trainable TINY network in burn: the same pre-norm transformer as
//! `train2.py`'s `Model`, with parameters stored in PyTorch's layout and named
//! like its state dict, so checkpoints load and export unchanged.
//!
//! ```mermaid
//! flowchart LR
//!     idx[letters B×T] --> emb["one-hot × emb + pos"]
//!     emb --> blk["block × L: x + attn(norm x), then x + f2(relu(f1(norm x)))"]
//!     blk --> nf["norm"] --> head["× embᵀ"] --> logits[B×T×VOCAB]
//! ```

use std::collections::BTreeMap;

use anyhow::ensure;
use burn::module::{Module, Param, ParamId};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::{Bool, Device, Distribution, Int, Tensor, TensorData};
use tiny::model::{CTX, Config, VOCAB};

use crate::checkpoint::{Array, Checkpoint};

/// Added to the mean square before the root in every norm.
const NORM_EPS: f64 = 1e-5;

/// The network's parameters.
#[derive(Module, Debug)]
pub struct Model {
    /// Letter embedding `VOCAB × D`, also the output head (`emb.weight`).
    emb: Param<Tensor<2>>,
    /// Position embedding `CTX × D` (`pos.weight`).
    pos: Param<Tensor<2>>,
    /// The transformer blocks (`blocks.N.`).
    blocks: Vec<Block>,
    /// Final norm gain (`nf.g`).
    nf: Param<Tensor<1>>,
    /// Number of attention heads (not recoverable from the shapes).
    heads: usize,
}

/// Parameters of one transformer block, matrices in PyTorch layout (out × in).
#[derive(Module, Debug)]
struct Block {
    /// Norm gain before attention.
    n1: Param<Tensor<1>>,
    /// Query projection `D × D`.
    q: Param<Tensor<2>>,
    /// Key projection.
    k: Param<Tensor<2>>,
    /// Value projection.
    v: Param<Tensor<2>>,
    /// Attention output projection.
    o: Param<Tensor<2>>,
    /// Norm gain before the feed-forward part.
    n2: Param<Tensor<1>>,
    /// Feed-forward expansion `HID × D`.
    f1: Param<Tensor<2>>,
    /// Feed-forward contraction `D × HID`.
    f2: Param<Tensor<2>>,
}

/// One named parameter, for loading and saving in state-dict order.
enum Named<'a> {
    /// A matrix.
    Matrix(String, &'a Param<Tensor<2>>),
    /// A gain vector.
    Gain(String, &'a Param<Tensor<1>>),
}

impl Model {
    /// A freshly initialised network, like `train2.Model.__init__`: embeddings
    /// N(0, 0.05), projections N(0, 0.5/√in), gains 1.
    #[must_use]
    pub fn new(config: Config, device: &Device) -> Self {
        let (d, hid) = (config.width, config.hidden);
        let block = || Block {
            n1: ones(d, device),
            q: normal(d, d, device),
            k: normal(d, d, device),
            v: normal(d, d, device),
            o: normal(d, d, device),
            n2: ones(d, device),
            f1: normal(hid, d, device),
            f2: normal(d, hid, device),
        };

        Self {
            emb: random([VOCAB, d], 0.05, device),
            pos: random([CTX, d], 0.05, device),
            blocks: (0..config.layers).map(|_| block()).collect(),
            nf: ones(d, device),
            heads: config.heads,
        }
    }

    /// A network holding the weights of `checkpoint`.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint's tensor names or shapes differ from the model's.
    pub fn from_checkpoint(checkpoint: &Checkpoint, device: &Device) -> anyhow::Result<Self> {
        let model = Self::new(checkpoint.config, device);
        let names = model.named();
        ensure!(
            names.len() == checkpoint.tensors.len(),
            "checkpoint has {} tensors, the model {}",
            checkpoint.tensors.len(),
            names.len()
        );

        let mut values = BTreeMap::new();
        for named in &names {
            let (name, shape) = match named {
                Named::Matrix(name, p) => (name, p.val().dims().to_vec()),
                Named::Gain(name, p) => (name, p.val().dims().to_vec()),
            };
            let array = checkpoint.tensor(name)?;
            ensure!(
                array.shape == shape,
                "{name}: expected {shape:?}, got {:?}",
                array.shape
            );
            values.insert(name.clone(), array.clone());
        }

        Ok(model.with_values(&values, device))
    }

    /// The current weights as a checkpoint (for export or saving).
    #[must_use]
    pub fn to_checkpoint(&self) -> Checkpoint {
        let tensors = self
            .named()
            .into_iter()
            .map(|named| match named {
                Named::Matrix(name, p) => (name, array(&p.val().into_data())),
                Named::Gain(name, p) => (name, array(&p.val().into_data())),
            })
            .collect();

        Checkpoint {
            config: self.config(),
            tensors,
        }
    }

    /// Shape of the network.
    #[must_use]
    pub fn config(&self) -> Config {
        let [_, width] = self.emb.val().dims();
        let hidden = self.blocks.first().map_or(0, |b| b.f1.val().dims()[0]);

        Config {
            width,
            layers: self.blocks.len(),
            heads: self.heads,
            hidden,
        }
    }

    /// Ids of the gain vectors, the parameter group without weight decay.
    #[must_use]
    pub fn gain_ids(&self) -> Vec<ParamId> {
        let blocks = self.blocks.iter().flat_map(|b| [b.n1.id, b.n2.id]);
        blocks.chain([self.nf.id]).collect()
    }

    /// Ids of the matrices, the parameter group with weight decay.
    #[must_use]
    pub fn matrix_ids(&self) -> Vec<ParamId> {
        let blocks = self
            .blocks
            .iter()
            .flat_map(|b| [b.q.id, b.k.id, b.v.id, b.o.id, b.f1.id, b.f2.id]);
        [self.emb.id, self.pos.id]
            .into_iter()
            .chain(blocks)
            .collect()
    }

    /// Logits `B × T × VOCAB` for letter tokens `idx` (`B × T`, `T ≤ CTX`).
    #[must_use]
    pub fn forward(&self, idx: Tensor<2, Int>) -> Tensor<3> {
        let [batch, len] = idx.dims();
        let width = self.config().width;
        let device = idx.device();
        let letters = one_hot(idx)
            .reshape([batch * len, VOCAB])
            .matmul(self.emb.val());
        let mut x = letters.reshape([batch, len, width])
            + self.pos.val().narrow(0, 0, len).unsqueeze::<3>();
        let mask = Tensor::<2, Bool>::tril_mask([len, len], 0, &device).unsqueeze::<4>();

        for block in &self.blocks {
            x = block.forward(x, &mask, self.heads);
        }

        linear(norm(x, &self.nf), &self.emb)
    }

    /// Every parameter with its state-dict name, in state-dict order.
    fn named(&self) -> Vec<Named<'_>> {
        let mut out = vec![
            Named::Matrix("emb.weight".into(), &self.emb),
            Named::Matrix("pos.weight".into(), &self.pos),
        ];
        for (l, b) in self.blocks.iter().enumerate() {
            let p = format!("blocks.{l}.");
            out.push(Named::Gain(format!("{p}n1.g"), &b.n1));
            for (n, m) in [("q", &b.q), ("k", &b.k), ("v", &b.v), ("o", &b.o)] {
                out.push(Named::Matrix(format!("{p}{n}.weight"), m));
            }
            out.push(Named::Gain(format!("{p}n2.g"), &b.n2));
            out.push(Named::Matrix(format!("{p}f1.weight"), &b.f1));
            out.push(Named::Matrix(format!("{p}f2.weight"), &b.f2));
        }

        out.push(Named::Gain("nf.g".into(), &self.nf));
        out
    }

    /// The same network with every parameter replaced by `values` (shapes already checked).
    fn with_values(self, values: &BTreeMap<String, Array>, device: &Device) -> Self {
        let matrix = |name: &str| Param::from_tensor(tensor::<2>(&values[name], device));
        let gain = |name: &str| Param::from_tensor(tensor::<1>(&values[name], device));
        let blocks = (0..self.blocks.len())
            .map(|l| {
                let p = format!("blocks.{l}.");
                Block {
                    n1: gain(&format!("{p}n1.g")),
                    q: matrix(&format!("{p}q.weight")),
                    k: matrix(&format!("{p}k.weight")),
                    v: matrix(&format!("{p}v.weight")),
                    o: matrix(&format!("{p}o.weight")),
                    n2: gain(&format!("{p}n2.g")),
                    f1: matrix(&format!("{p}f1.weight")),
                    f2: matrix(&format!("{p}f2.weight")),
                }
            })
            .collect();

        Self {
            emb: matrix("emb.weight"),
            pos: matrix("pos.weight"),
            blocks,
            nf: gain("nf.g"),
            heads: self.heads,
        }
    }
}

impl Block {
    /// `x + attention(norm(x))`, then `x + f2(relu(f1(norm(x))))`.
    fn forward(&self, x: Tensor<3>, mask: &Tensor<4, Bool>, heads: usize) -> Tensor<3> {
        let attended = self.attention(&norm(x.clone(), &self.n1), mask, heads);
        let x = x + linear(attended, &self.o);
        let middle = relu(linear(norm(x.clone(), &self.n2), &self.f1));

        x + linear(middle, &self.f2)
    }

    /// Causal multi-head self-attention of `h` (`B × T × D`), before the output projection.
    fn attention(&self, h: &Tensor<3>, mask: &Tensor<4, Bool>, heads: usize) -> Tensor<3> {
        let [batch, len, width] = h.dims();
        let head_width = width / heads;
        let split = |w: &Param<Tensor<2>>| {
            linear(h.clone(), w)
                .reshape([batch, len, heads, head_width])
                .swap_dims(1, 2)
        };
        let (q, k, v) = (split(&self.q), split(&self.k), split(&self.v));

        let scores = q.matmul(k.swap_dims(2, 3)) / f64_of(head_width).sqrt();
        let weights = softmax(scores.mask_fill(mask.clone(), f32::NEG_INFINITY), 3);

        weights
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, len, width])
    }
}

/// `B × T × VOCAB` one-hot rows for letter tokens, as floats.
///
/// Embedding lookups and loss picks go through this and a matrix product,
/// which differentiates cheaply on every backend.
#[must_use]
pub fn one_hot(idx: Tensor<2, Int>) -> Tensor<3> {
    idx.one_hot::<3>(VOCAB).float()
}

/// `x · W^T`, a bias-free PyTorch `Linear`, as one 2-D matrix product.
fn linear(x: Tensor<3>, w: &Param<Tensor<2>>) -> Tensor<3> {
    let [batch, len, inputs] = x.dims();
    let weight = w.val();
    let outputs = weight.dims()[0];

    x.reshape([batch * len, inputs])
        .matmul(weight.transpose())
        .reshape([batch, len, outputs])
}

/// RMS norm over the last dimension, times the gains.
fn norm(x: Tensor<3>, gain: &Param<Tensor<1>>) -> Tensor<3> {
    let scale = (x.clone().powi_scalar(2).mean_dim(2) + NORM_EPS)
        .sqrt()
        .recip();
    x * scale * gain.val().unsqueeze::<3>()
}

/// A `out × in` parameter drawn from N(0, 0.5/√in).
fn normal(out: usize, inputs: usize, device: &Device) -> Param<Tensor<2>> {
    random([out, inputs], 0.5 / f64_of(inputs).sqrt(), device)
}

/// A parameter drawn from N(0, `std`).
fn random(shape: [usize; 2], std: f64, device: &Device) -> Param<Tensor<2>> {
    Param::from_tensor(Tensor::random(
        shape,
        Distribution::Normal(0.0, std),
        device,
    ))
}

/// A gain vector of ones.
fn ones(width: usize, device: &Device) -> Param<Tensor<1>> {
    Param::from_tensor(Tensor::ones([width], device))
}

/// A tensor holding `array`.
fn tensor<const D: usize>(array: &Array, device: &Device) -> Tensor<D> {
    Tensor::from_data(
        TensorData::new(array.data.clone(), array.shape.clone()),
        device,
    )
}

/// Tensor data back as an [`Array`].
fn array(data: &TensorData) -> Array {
    let shape = data.shape().to_vec();
    Array {
        data: data.try_to_vec::<f32>().unwrap_or_default(),
        shape,
    }
}

/// A small count as `f64`.
fn f64_of(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}
