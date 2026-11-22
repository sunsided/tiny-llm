//! The `TINY.BIN` version 2 weight file: 8-bit tables with one scale per row.
//!
//! Layout (little endian):
//!
//! ```text
//! "TNY2"  width layers heads hidden            (u16 each)
//! embedding   VOCAB x width   table
//! position    CTX   x width   table
//! per layer:  attention gain, query, key, value, output,
//!             feed-forward gain, up (hidden x width), down (width x hidden)
//! final gain
//! ```
//!
//! A table is stored row by row: an `f32` scale, then one `i8` per column.
//! A gain is `width` plain `f32` values.

use std::path::Path;

/// Number of distinct letters: 0 padding, 1 newline, 2.. the ASCII range `' '..='~'`.
pub const VOCAB: usize = 97;

/// Longest sequence (question, newline and answer) in letters.
pub const CTX: usize = 128;

/// File signature of a version 2 weight file.
const MAGIC: &[u8; 4] = b"TNY2";

/// Shape of the network, read from the file header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Width of the residual stream (`D`). Always a multiple of [`Config::heads`].
    pub width: usize,
    /// Number of transformer blocks (`L`). At least 1.
    pub layers: usize,
    /// Number of attention heads (`H`). At least 1.
    pub heads: usize,
    /// Width of the feed-forward middle layer (`HID`). At least 1.
    pub hidden: usize,
}

impl Config {
    /// Width of one attention head.
    #[must_use]
    pub fn head_width(&self) -> usize {
        self.width / self.heads
    }

    /// Number of weight tables in the file, used for loading progress displays.
    #[must_use]
    pub fn table_count(&self) -> usize {
        2 + 6 * self.layers
    }

    /// Total number of weights and gains, the "367,000" of the title bar.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let (d, h) = (self.width, self.hidden);
        let per_layer = 4 * d * d + 2 * d * h + 2 * d;

        VOCAB * d + CTX * d + self.layers * per_layer + d
    }
}

/// Why a weight file could not be read.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// The file could not be opened or read.
    #[error("cannot read the weight file: {0}")]
    Io(#[from] std::io::Error),
    /// The first four bytes are not `TNY2`.
    #[error("not a version 2 network file")]
    NotVersion2,
    /// The header describes a shape this program cannot run.
    #[error("unsupported network size {0:?}")]
    BadShape(Config),
    /// The file ends before all tables were read.
    #[error("the file is too short or damaged")]
    Truncated,
    /// Bytes are left over after the final gain.
    #[error("the file has {extra} bytes too many")]
    TrailingBytes {
        /// Number of unread bytes at the end.
        extra: usize,
    },
}

/// A weight matrix with one byte per weight and one `f32` scale per row.
#[derive(Debug, Clone)]
pub struct Table {
    /// Scale of each row; the real weight is `weights[r][c] * scales[r]`.
    scales: Vec<f32>,
    /// Row-major quantised weights, `rows * cols` long.
    weights: Vec<i8>,
    /// Length of one row.
    cols: usize,
}

impl Table {
    /// Number of rows (output size of [`Table::matvec`]).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.scales.len()
    }

    /// Quantised weights and scale of row `r`.
    ///
    /// # Panics
    ///
    /// Panics if `r >= self.rows()`.
    #[must_use]
    pub fn row(&self, r: usize) -> (&[i8], f32) {
        (&self.weights[r * self.cols..][..self.cols], self.scales[r])
    }

    /// `y = table * x`, accumulating in `f32` in column order like `TINY.C`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is shorter than a row or `y` shorter than the row count.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        let rows = self.weights.chunks_exact(self.cols).zip(&self.scales);
        for (out, (row, scale)) in y.iter_mut().zip(rows) {
            let sum: f32 = row.iter().zip(x).map(|(&w, &v)| f32::from(w) * v).sum();
            *out = sum * scale;
        }
    }
}

/// Weights of one transformer block.
#[derive(Debug, Clone)]
pub struct Layer {
    /// Norm gain before attention.
    pub attention_gain: Vec<f32>,
    /// Query projection, `width x width`.
    pub query: Table,
    /// Key projection, `width x width`.
    pub key: Table,
    /// Value projection, `width x width`.
    pub value: Table,
    /// Attention output projection, `width x width`.
    pub output: Table,
    /// Norm gain before the feed-forward part.
    pub feed_gain: Vec<f32>,
    /// Feed-forward expansion, `hidden x width`.
    pub up: Table,
    /// Feed-forward contraction, `width x hidden`.
    pub down: Table,
}

/// All weights of the network.
#[derive(Debug, Clone)]
pub struct Weights {
    /// Shape from the header.
    pub config: Config,
    /// Letter embedding, `VOCAB x width`; also the output head.
    pub embedding: Table,
    /// Position embedding, `CTX x width`.
    pub position: Table,
    /// The transformer blocks, `config.layers` long.
    pub layers: Vec<Layer>,
    /// Norm gain before the output head.
    pub final_gain: Vec<f32>,
}

impl Weights {
    /// Reads a weight file from disk.
    ///
    /// # Errors
    ///
    /// [`LoadError::Io`] if the file cannot be read, otherwise as [`Weights::from_bytes`].
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Parses a complete weight file.
    ///
    /// # Errors
    ///
    /// - [`LoadError::NotVersion2`] if the signature is wrong.
    /// - [`LoadError::BadShape`] if a size is zero or the width is not a multiple of the heads.
    /// - [`LoadError::Truncated`] if the data ends early.
    /// - [`LoadError::TrailingBytes`] if data is left after the final gain.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let mut reader = Reader { bytes, pos: 0 };
        let config = reader.header()?;
        let d = config.width;

        let embedding = reader.table(VOCAB, d)?;
        let position = reader.table(CTX, d)?;
        let layers = (0..config.layers)
            .map(|_| reader.layer(&config))
            .collect::<Result<_, _>>()?;
        let final_gain = reader.gains(d)?;

        let extra = bytes.len() - reader.pos;
        if extra > 0 {
            return Err(LoadError::TrailingBytes { extra });
        }

        Ok(Self {
            config,
            embedding,
            position,
            layers,
            final_gain,
        })
    }
}

/// Cursor over the raw file bytes.
struct Reader<'a> {
    /// The whole file.
    bytes: &'a [u8],
    /// Offset of the next unread byte.
    pos: usize,
}

impl Reader<'_> {
    /// Takes the next `n` bytes.
    fn take(&mut self, n: usize) -> Result<&[u8], LoadError> {
        let end = self.pos.checked_add(n).ok_or(LoadError::Truncated)?;
        let slice = self.bytes.get(self.pos..end).ok_or(LoadError::Truncated)?;
        self.pos = end;

        Ok(slice)
    }

    /// Reads a little-endian `u16` as a size.
    fn size(&mut self) -> Result<usize, LoadError> {
        let b = self.take(2)?;
        Ok(usize::from(u16::from_le_bytes([b[0], b[1]])))
    }

    /// Reads a little-endian `f32`.
    fn float(&mut self) -> Result<f32, LoadError> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Reads and validates the signature and the four shape numbers.
    fn header(&mut self) -> Result<Config, LoadError> {
        if self.take(4).map_err(|_| LoadError::NotVersion2)? != MAGIC {
            return Err(LoadError::NotVersion2);
        }

        let config = Config {
            width: self.size()?,
            layers: self.size()?,
            heads: self.size()?,
            hidden: self.size()?,
        };
        let sizes = [config.width, config.layers, config.heads, config.hidden];
        if sizes.contains(&0) || !config.width.is_multiple_of(config.heads) {
            return Err(LoadError::BadShape(config));
        }

        Ok(config)
    }

    /// Reads a `rows x cols` table, row by row.
    fn table(&mut self, rows: usize, cols: usize) -> Result<Table, LoadError> {
        let mut scales = Vec::with_capacity(rows);
        let mut weights = Vec::with_capacity(rows * cols);
        for _ in 0..rows {
            scales.push(self.float()?);
            weights.extend(self.take(cols)?.iter().map(|&b| b.cast_signed()));
        }

        Ok(Table {
            scales,
            weights,
            cols,
        })
    }

    /// Reads `n` plain `f32` gains.
    fn gains(&mut self, n: usize) -> Result<Vec<f32>, LoadError> {
        (0..n).map(|_| self.float()).collect()
    }

    /// Reads one transformer block in file order.
    fn layer(&mut self, config: &Config) -> Result<Layer, LoadError> {
        let (d, hid) = (config.width, config.hidden);

        Ok(Layer {
            attention_gain: self.gains(d)?,
            query: self.table(d, d)?,
            key: self.table(d, d)?,
            value: self.table(d, d)?,
            output: self.table(d, d)?,
            feed_gain: self.gains(d)?,
            up: self.table(hid, d)?,
            down: self.table(d, hid)?,
        })
    }
}
