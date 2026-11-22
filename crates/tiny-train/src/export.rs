//! Writes the version 2 `TINY.BIN` format that `tiny::model::Weights` reads,
//! rounding exactly like `train2.py export` does.

use anyhow::ensure;
use tiny::model::{CTX, VOCAB};

use crate::checkpoint::{Array, Checkpoint};

/// The `TINY.BIN` bytes of a checkpoint.
///
/// # Errors
///
/// Fails when a tensor is missing or has the wrong shape for the checkpoint's config.
pub fn to_bin(checkpoint: &Checkpoint) -> anyhow::Result<Vec<u8>> {
    let c = checkpoint.config;
    let (d, hid) = (c.width, c.hidden);
    let mut out = b"TNY2".to_vec();
    for size in [c.width, c.layers, c.heads, c.hidden] {
        out.extend_from_slice(&u16::try_from(size)?.to_le_bytes());
    }

    let table = |name: &str, rows: usize, cols: usize, out: &mut Vec<u8>| {
        write_table(out, checkpoint.tensor(name)?, rows, cols)
    };
    table("emb.weight", VOCAB, d, &mut out)?;
    table("pos.weight", CTX, d, &mut out)?;
    for l in 0..c.layers {
        let p = format!("blocks.{l}.");
        write_gains(&mut out, checkpoint.tensor(&format!("{p}n1.g"))?, d)?;
        for n in ["q", "k", "v", "o"] {
            table(&format!("{p}{n}.weight"), d, d, &mut out)?;
        }
        write_gains(&mut out, checkpoint.tensor(&format!("{p}n2.g"))?, d)?;
        table(&format!("{p}f1.weight"), hid, d, &mut out)?;
        table(&format!("{p}f2.weight"), d, hid, &mut out)?;
    }
    write_gains(&mut out, checkpoint.tensor("nf.g")?, d)?;

    Ok(out)
}

/// Appends a `rows x cols` matrix row by row: an `f32` scale, then one `i8` per weight.
fn write_table(out: &mut Vec<u8>, array: &Array, rows: usize, cols: usize) -> anyhow::Result<()> {
    for row in array.rows(rows, cols)? {
        let (scale, weights) = quantise_row(row);
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend(weights.iter().map(|w| w.to_le_bytes()[0]));
    }

    Ok(())
}

/// Appends `width` gains as plain `f32`.
fn write_gains(out: &mut Vec<u8>, array: &Array, width: usize) -> anyhow::Result<()> {
    ensure!(
        array.shape == [width],
        "expected {width} gains, got {:?}",
        array.shape
    );
    ensure!(
        array.data.len() == width,
        "{width} gains hold {} values",
        array.data.len()
    );
    for g in &array.data {
        out.extend_from_slice(&g.to_le_bytes());
    }

    Ok(())
}

/// One row's scale and 8-bit weights, rounded like NumPy in `train2.export`.
///
/// The scale is computed in `f64` (`float(abs(row).max()) / 127.0`, at least `1e-8`) and
/// stored as `f32`. The division `row / s` happens in `f32`, and `np.round` rounds half to even.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "NumPy casts the scale and the clipped values the same way"
)]
pub fn quantise_row(row: &[f32]) -> (f32, Vec<i8>) {
    let big = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = (f64::from(big) / 127.0).max(1e-8) as f32;
    let weights = row
        .iter()
        .map(|&v| (v / scale).round_ties_even().clamp(-127.0, 127.0) as i8)
        .collect();

    (scale, weights)
}
