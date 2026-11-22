//! Small numeric kernels, written to round exactly like `TINY.C` built with a
//! plain (non fast-math) C compiler: `f32` accumulation in index order, and
//! `f64` only where the C code calls `sqrt` or `exp`.

/// RMS-normalises `x`, multiplies by the gains `g` and writes the result to `y`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "C rounds double results to float the same way"
)]
pub fn norm(x: &[f32], g: &[f32], y: &mut [f32]) {
    let squares: f32 = x.iter().map(|v| v * v).sum();
    let mean = squares / count(x.len());
    let scale = (1.0 / (f64::from(mean) + 1e-5).sqrt()) as f32;

    for ((out, &v), &gain) in y.iter_mut().zip(x).zip(g) {
        *out = v * scale * gain;
    }
}

/// `e^x`, but exactly 0 below -60 (the C code avoids underflow traps that way).
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "C rounds double results to float the same way"
)]
pub fn safe_exp(x: f32) -> f32 {
    if x < -60.0 {
        0.0
    } else {
        f64::from(x).exp() as f32
    }
}

/// Quantises `src` into bytes with one shared scale, returning that scale.
///
/// Rounds half away from zero by adding ±0.5 and truncating, as `TINY.C` does.
#[expect(
    clippy::cast_possible_truncation,
    reason = "values are within ±127.5 by construction; `as` truncates like the C cast"
)]
pub fn quantise(dst: &mut [i8], src: &[f32]) -> f32 {
    let big = src.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-20);
    let k = 127.0 / big;

    for (out, &v) in dst.iter_mut().zip(src) {
        let v = v * k;
        let rounded = if v < 0.0 { v - 0.5 } else { v + 0.5 };
        *out = rounded as i8;
    }

    big / 127.0
}

/// `f32` dot product of floats with quantised values, in index order.
#[must_use]
pub fn dot(x: &[f32], q: &[i8]) -> f32 {
    x.iter().zip(q).map(|(&a, &b)| a * f32::from(b)).sum()
}

/// A length as `f32`, the way C converts an `int` divisor.
#[must_use]
pub fn count(n: usize) -> f32 {
    u16::try_from(n).map_or(f32::MAX, f32::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rounding is half away from zero and the scale maps the largest value to 127.
    #[test]
    fn quantise_matches_c_rounding() {
        let mut out = [0i8; 4];
        let scale = quantise(&mut out, &[1.0, -0.5, 0.25, -1.0]);

        assert_eq!(out, [127, -64, 32, -127]);
        assert!((scale - 1.0 / 127.0).abs() < 1e-9);
    }

    /// Very negative inputs give exactly zero.
    #[test]
    fn safe_exp_cuts_off() {
        assert_eq!(safe_exp(-61.0), 0.0);
        assert!((safe_exp(0.0) - 1.0).abs() < f32::EPSILON);
    }
}
