//! Full-precision network weights, as saved by the original `train2.py`
//! (`torch.save({"cfg": {...}, "sd": state_dict})`).

use std::collections::BTreeMap;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, bail};
use candle_core::pickle::{Object, Stack, read_all_with_key};
use candle_core::{DType, Tensor};
use tiny::model::Config;

/// A network's shape and its named full-precision tensors.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Shape of the network.
    pub config: Config,
    /// State-dict tensors by their PyTorch names (`emb.weight`, `blocks.0.q.weight`, ...), as `f32`.
    pub tensors: BTreeMap<String, Tensor>,
}

impl Checkpoint {
    /// Reads a PyTorch checkpoint written by `train2.py` (`tiny2.pt`).
    ///
    /// # Errors
    ///
    /// Fails when the file is not such a checkpoint or a tensor cannot be read.
    pub fn load_pt(path: &Path) -> anyhow::Result<Self> {
        let config =
            read_config(path).with_context(|| format!("reading cfg of {}", path.display()))?;
        let tensors = read_all_with_key(path, Some("sd"))?
            .into_iter()
            .map(|(name, tensor)| Ok((name, tensor.to_dtype(DType::F32)?)))
            .collect::<candle_core::Result<_>>()?;

        Ok(Self { config, tensors })
    }

    /// The tensor called `name`.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint has no such tensor.
    pub fn tensor(&self, name: &str) -> anyhow::Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("checkpoint has no tensor {name}"))
    }
}

/// Reads the `cfg` dictionary (`D`, `L`, `H`, `HID`) from the pickle in the checkpoint zip.
fn read_config(path: &Path) -> anyhow::Result<Config> {
    let mut zip = zip::ZipArchive::new(BufReader::new(std::fs::File::open(path)?))?;
    let names = zip
        .file_names()
        .map(|name| name.map(std::borrow::Cow::into_owned))
        .collect::<Result<Vec<_>, _>>()?;
    let name = names
        .into_iter()
        .find(|name| name.ends_with("data.pkl"))
        .context("no data.pkl in the checkpoint")?;
    let mut stack = Stack::empty();
    stack.read_loop(&mut BufReader::new(zip.by_name(&name)?))?;

    let cfg = entry(stack.finalize()?, "cfg")?;
    let size = |key: &str| -> anyhow::Result<usize> {
        match entry(cfg.clone(), key)? {
            Object::Int(n) => Ok(usize::try_from(n)?),
            Object::Long(n) => Ok(usize::try_from(n)?),
            other => bail!("cfg.{key} is not a number: {other:?}"),
        }
    };

    Ok(Config {
        width: size("D")?,
        layers: size("L")?,
        heads: size("H")?,
        hidden: size("HID")?,
    })
}

/// The value stored under string key `key` in a pickled dict.
fn entry(dict: Object, key: &str) -> anyhow::Result<Object> {
    let Object::Dict(entries) = dict else {
        bail!("expected a dict holding {key}");
    };

    entries
        .into_iter()
        .find(|(k, _)| *k == Object::Unicode(key.to_owned()))
        .map(|(_, v)| v)
        .with_context(|| format!("no {key} in the dict"))
}
