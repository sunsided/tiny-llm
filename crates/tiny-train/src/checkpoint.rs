//! Full-precision network weights, independent of any tensor library: read
//! from the original `train2.py` checkpoints (`torch.save({"cfg": {...}, "sd":
//! state_dict})`) and saved as safetensors.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, bail, ensure};
use pytorch_reader::{DType as PtType, PytorchReader};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;
use tiny::model::Config;

/// Metadata key under which saved checkpoints keep `D,L,H,HID`.
const CONFIG_KEY: &str = "config";

/// A row-major `f32` array with its shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    /// Size of each dimension, outermost first.
    pub shape: Vec<usize>,
    /// The values, `shape.iter().product()` of them.
    pub data: Vec<f32>,
}

impl Array {
    /// The rows of a matrix (`shape` of rank 2).
    ///
    /// # Errors
    ///
    /// Fails when the array is not a `rows × cols` matrix, or its data does not fill that shape.
    pub fn rows(
        &self,
        rows: usize,
        cols: usize,
    ) -> anyhow::Result<std::slice::ChunksExact<'_, f32>> {
        ensure!(
            self.shape == [rows, cols],
            "expected {rows}x{cols}, got {:?}",
            self.shape
        );
        ensure!(
            self.data.len() == rows * cols,
            "a {rows}x{cols} matrix holds {} values",
            self.data.len()
        );
        Ok(self.data.chunks_exact(cols))
    }
}

/// A network's shape and its named full-precision tensors.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Shape of the network.
    pub config: Config,
    /// State-dict tensors by their PyTorch names (`emb.weight`, `blocks.0.q.weight`, ...).
    pub tensors: BTreeMap<String, Array>,
}

/// The `cfg` dictionary of a `train2.py` checkpoint.
#[derive(Debug, Deserialize)]
#[expect(
    non_snake_case,
    reason = "field names are the dictionary keys of train2.py's Cfg"
)]
struct PtConfig {
    /// Width.
    D: usize,
    /// Layers.
    L: usize,
    /// Heads.
    H: usize,
    /// Middle-layer width.
    HID: usize,
}

impl Checkpoint {
    /// Reads a checkpoint: a PyTorch `.pt` file from `train2.py`, or a
    /// `.safetensors` file written by [`Checkpoint::save`].
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be read as either.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if path.extension().is_some_and(|ext| ext == "pt") {
            Self::load_pt(path)
        } else {
            Self::load_safetensors(path)
        }
    }

    /// Reads a PyTorch checkpoint written by `train2.py` (`tiny2.pt`).
    ///
    /// # Errors
    ///
    /// Fails when the file is not such a checkpoint or holds non-`f32` tensors.
    pub fn load_pt(path: &Path) -> anyhow::Result<Self> {
        let cfg: PtConfig = PytorchReader::load_config(path, Some("cfg"))
            .with_context(|| format!("reading cfg of {}", path.display()))?;
        let reader = PytorchReader::with_top_level_key(path, "sd")?;
        let tensors = reader
            .into_tensors()
            .into_iter()
            .map(|(name, tensor)| {
                ensure!(
                    tensor.dtype() == PtType::F32,
                    "{name} is {:?}, not f32",
                    tensor.dtype()
                );
                let array = Array {
                    shape: tensor.shape().to_vec(),
                    data: floats(&tensor.read()?),
                };
                Ok((name, array))
            })
            .collect::<anyhow::Result<_>>()?;
        let config = Config {
            width: cfg.D,
            layers: cfg.L,
            heads: cfg.H,
            hidden: cfg.HID,
        };

        Ok(Self { config, tensors })
    }

    /// Reads a safetensors checkpoint written by [`Checkpoint::save`].
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be read, lacks the config, or holds non-`f32` tensors.
    pub fn load_safetensors(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let (_, metadata) = SafeTensors::read_metadata(&bytes)?;
        let config = metadata
            .metadata()
            .as_ref()
            .and_then(|m| m.get(CONFIG_KEY))
            .context("no config in the checkpoint metadata")?;
        let config = parse_config(config)?;

        let tensors = SafeTensors::deserialize(&bytes)?
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                ensure!(
                    view.dtype() == Dtype::F32,
                    "{name} is {:?}, not f32",
                    view.dtype()
                );
                Ok((
                    name,
                    Array {
                        shape: view.shape().to_vec(),
                        data: floats(view.data()),
                    },
                ))
            })
            .collect::<anyhow::Result<_>>()?;

        Ok(Self { config, tensors })
    }

    /// Writes the checkpoint as safetensors, with the config in the metadata.
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be written.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let c = self.config;
        let metadata = HashMap::from([(
            CONFIG_KEY.to_owned(),
            format!("{},{},{},{}", c.width, c.layers, c.heads, c.hidden),
        )]);
        let bytes: Vec<(String, Vec<u8>, Vec<usize>)> = self
            .tensors
            .iter()
            .map(|(name, array)| {
                let data = array.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                (name.clone(), data, array.shape.clone())
            })
            .collect();
        let views = bytes
            .iter()
            .map(|(name, data, shape)| {
                Ok((
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), data)?,
                ))
            })
            .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;

        let file = safetensors::serialize(views, Some(metadata))?;
        std::fs::write(path, file).with_context(|| format!("writing {}", path.display()))
    }

    /// The tensor called `name`.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint has no such tensor.
    pub fn tensor(&self, name: &str) -> anyhow::Result<&Array> {
        self.tensors
            .get(name)
            .with_context(|| format!("checkpoint has no tensor {name}"))
    }
}

/// `"96,4,4,256"` as a [`Config`].
fn parse_config(text: &str) -> anyhow::Result<Config> {
    let sizes = text
        .split(',')
        .map(|n| n.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("bad config {text:?}"))?;
    let [width, layers, heads, hidden] = sizes[..] else {
        bail!("config {text:?} needs four numbers");
    };

    Ok(Config {
        width,
        layers,
        heads,
        hidden,
    })
}

/// Little-endian `f32` values from raw bytes.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}
