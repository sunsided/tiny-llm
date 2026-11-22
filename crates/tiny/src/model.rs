//! The TINY network: a small pre-norm transformer that reads and writes one
//! letter per step, keeping an 8-bit key/value cache of earlier letters.
//!
//! ```mermaid
//! flowchart LR
//!     tok[letter + position] --> emb[embeddings]
//!     emb --> blk["block × L: norm → attention → + → norm → ReLU MLP → +"]
//!     blk --> nf[final norm] --> head["× embeddingᵀ"] --> scores
//! ```

mod math;
mod weights;

pub use weights::{CTX, Config, Layer, LoadError, Table, VOCAB, Weights};

pub use math::{count, safe_exp};

use math::{dot, norm, quantise};

/// What one [`Network::step`] produced.
#[derive(Debug, Clone)]
pub struct Output {
    /// How well each letter fits as the next one (logits), `VOCAB` long.
    pub scores: Vec<f32>,
    /// Mean `ReLU` output of each block's middle layer, [`Config::layers`] long.
    /// Only used for the activity display.
    pub layer_activity: Vec<f32>,
}

/// The network plus everything it remembers about the current sequence.
#[derive(Debug, Clone)]
pub struct Network {
    /// The trained weights.
    weights: Weights,
    /// One key/value cache per block.
    cache: Vec<LayerCache>,
    /// Work buffers reused across steps.
    scratch: Scratch,
    /// Result of the latest step.
    output: Output,
    /// `sqrt(head width)`, the attention score divisor.
    attention_divisor: f32,
}

/// Keys and values of every position seen so far, one byte per number and
/// one scale per position, like `TINY.C`.
#[derive(Debug, Clone)]
struct LayerCache {
    /// `CTX x width` quantised keys.
    keys: Vec<i8>,
    /// Scale of each cached key row.
    key_scales: Vec<f32>,
    /// `CTX x width` quantised values.
    values: Vec<i8>,
    /// Scale of each cached value row.
    value_scales: Vec<f32>,
}

/// Work buffers for one step.
#[derive(Debug, Clone)]
struct Scratch {
    /// Residual stream, `width` long.
    x: Vec<f32>,
    /// Normalised residual stream, `width` long.
    normed: Vec<f32>,
    /// Query of the current letter, `width` long.
    query: Vec<f32>,
    /// Temporary projection result, `width` long.
    temp: Vec<f32>,
    /// Attention result of all heads, `width` long.
    attended: Vec<f32>,
    /// Feed-forward middle layer, `hidden` long.
    middle: Vec<f32>,
    /// Attention weights over positions, `CTX` long.
    attention: Vec<f32>,
}

impl Network {
    /// Wraps loaded weights and allocates the cache and buffers.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "C rounds double results to float the same way"
    )]
    pub fn new(weights: Weights) -> Self {
        let Config { width, hidden, .. } = weights.config;
        let cache = (0..weights.config.layers)
            .map(|_| LayerCache {
                keys: vec![0; CTX * width],
                key_scales: vec![0.0; CTX],
                values: vec![0; CTX * width],
                value_scales: vec![0.0; CTX],
            })
            .collect();
        let scratch = Scratch {
            x: vec![0.0; width],
            normed: vec![0.0; width],
            query: vec![0.0; width],
            temp: vec![0.0; width],
            attended: vec![0.0; width],
            middle: vec![0.0; hidden],
            attention: vec![0.0; CTX],
        };
        let output = Output {
            scores: vec![0.0; VOCAB],
            layer_activity: vec![0.0; weights.config.layers],
        };
        let head_width = count(weights.config.head_width());

        Self {
            attention_divisor: f64::from(head_width).sqrt() as f32,
            weights,
            cache,
            scratch,
            output,
        }
    }

    /// Shape of the network.
    #[must_use]
    pub fn config(&self) -> Config {
        self.weights.config
    }

    /// Result of the latest step.
    #[must_use]
    pub fn output(&self) -> &Output {
        &self.output
    }

    /// Feeds letter `token` at position `pos` and computes the scores of the next letter.
    ///
    /// Positions must be fed in order 0, 1, 2, …; feeding position 0 starts a new
    /// sequence (later cache entries are simply overwritten).
    ///
    /// # Panics
    ///
    /// Panics if `token >= VOCAB` or `pos >= CTX`.
    pub fn step(&mut self, token: usize, pos: usize) -> &Output {
        assert!(
            token < VOCAB && pos < CTX,
            "letter {token} at {pos} is out of range"
        );

        self.embed(token, pos);
        for l in 0..self.weights.layers.len() {
            self.attend(l, pos);
            self.feed_forward(l);
        }
        self.predict();

        &self.output
    }

    /// Residual stream = letter embedding + position embedding.
    fn embed(&mut self, token: usize, pos: usize) {
        let (letter, letter_scale) = self.weights.embedding.row(token);
        let (place, place_scale) = self.weights.position.row(pos);

        for ((x, &e), &p) in self.scratch.x.iter_mut().zip(letter).zip(place) {
            *x = f32::from(e) * letter_scale + f32::from(p) * place_scale;
        }
    }

    /// Causal self-attention of block `l`, added to the residual stream.
    fn attend(&mut self, l: usize, pos: usize) {
        let layer = &self.weights.layers[l];
        let s = &mut self.scratch;
        let cache = &mut self.cache[l];
        let width = s.x.len();
        let row = pos * width..(pos + 1) * width;

        norm(&s.x, &layer.attention_gain, &mut s.normed);
        layer.key.matvec(&s.normed, &mut s.temp);
        cache.key_scales[pos] = quantise(&mut cache.keys[row.clone()], &s.temp);
        layer.value.matvec(&s.normed, &mut s.temp);
        cache.value_scales[pos] = quantise(&mut cache.values[row], &s.temp);
        layer.query.matvec(&s.normed, &mut s.query);

        let head_width = self.weights.config.head_width();
        for h in 0..self.weights.config.heads {
            let head = HeadView {
                cache,
                offset: h * head_width,
                width,
                head_width,
                divisor: self.attention_divisor,
            };
            let query = &s.query[head.offset..][..head_width];
            head.weigh(query, &mut s.attention[..=pos]);
            head.mix(
                &s.attention[..=pos],
                &mut s.attended[head.offset..][..head_width],
            );
        }

        layer.output.matvec(&s.attended, &mut s.temp);
        add(&mut s.x, &s.temp);
    }

    /// `ReLU` feed-forward part of block `l`, added to the residual stream.
    fn feed_forward(&mut self, l: usize) {
        let layer = &self.weights.layers[l];
        let s = &mut self.scratch;

        norm(&s.x, &layer.feed_gain, &mut s.normed);
        layer.up.matvec(&s.normed, &mut s.middle);
        let mut active = 0.0f32;
        for m in &mut s.middle {
            if *m < 0.0 {
                *m = 0.0;
            } else {
                active += *m;
            }
        }
        self.output.layer_activity[l] = active / count(s.middle.len());

        layer.down.matvec(&s.middle, &mut s.temp);
        add(&mut s.x, &s.temp);
    }

    /// Final norm, then scores against the (tied) letter embedding.
    fn predict(&mut self) {
        let s = &mut self.scratch;
        norm(&s.x, &self.weights.final_gain, &mut s.normed);
        self.weights
            .embedding
            .matvec(&s.normed, &mut self.output.scores);
    }
}

/// One attention head's slice of a block's cache.
struct HeadView<'a> {
    /// The block's cache.
    cache: &'a LayerCache,
    /// First column of this head inside a cache row.
    offset: usize,
    /// Length of a whole cache row.
    width: usize,
    /// Number of columns of this head.
    head_width: usize,
    /// `sqrt(head_width)`.
    divisor: f32,
}

impl HeadView<'_> {
    /// Cached key or value columns of this head at position `j`.
    fn slice<'b>(&self, rows: &'b [i8], j: usize) -> &'b [i8] {
        &rows[j * self.width + self.offset..][..self.head_width]
    }

    /// Softmax attention weights (not yet divided by their sum) of `query`
    /// over every cached position; `weights.len()` is the number of positions.
    fn weigh(&self, query: &[f32], weights: &mut [f32]) {
        let mut big = -1e30f32;
        for (j, w) in weights.iter_mut().enumerate() {
            *w = dot(query, self.slice(&self.cache.keys, j)) * self.cache.key_scales[j]
                / self.divisor;
            big = big.max(*w);
        }

        for w in weights.iter_mut() {
            *w = safe_exp(*w - big);
        }
    }

    /// Weighted mean of the cached values, written to `out`.
    fn mix(&self, weights: &[f32], out: &mut [f32]) {
        let sum: f32 = weights.iter().sum();
        out.fill(0.0);

        for (j, &w) in weights.iter().enumerate() {
            if w == 0.0 {
                continue;
            }

            let k = w * self.cache.value_scales[j] / sum;
            for (o, &v) in out.iter_mut().zip(self.slice(&self.cache.values, j)) {
                *o += k * f32::from(v);
            }
        }
    }
}

/// `x += y`, element by element.
fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}
