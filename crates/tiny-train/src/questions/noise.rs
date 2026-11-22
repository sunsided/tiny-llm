//! Typing noise (`typo` and `style` of `gen2.py`): typos, openers, case
//! changes, endings and stray spaces. Questions are ASCII, so byte indices
//! are Python's character indices.

use super::cap;
use super::tables::{KEYS, PREFIX, SUFFIX};
use crate::pyrandom::PyRandom;

/// `typo`: swaps, drops, doubles or mistypes one letter of a question with at
/// least four letters.
#[expect(
    clippy::many_single_char_names,
    reason = "names follow gen2.py for side-by-side checking"
)]
fn typo(q: &str, rng: &mut PyRandom) -> String {
    let b = q.as_bytes();
    let idx: Vec<usize> = (0..b.len())
        .filter(|&i| b[i].is_ascii_alphabetic())
        .collect();
    if idx.len() < 4 {
        return q.to_owned();
    }

    let i = *rng.choice(&idx);
    let k = rng.random();
    if k < 0.3 && i + 1 < b.len() {
        return format!(
            "{}{}{}{}",
            &q[..i],
            &q[i + 1..=i + 1],
            &q[i..=i],
            &q[i + 2..]
        );
    }

    if k < 0.55 {
        return format!("{}{}", &q[..i], &q[i + 1..]);
    }

    if k < 0.75 {
        return format!("{}{}", &q[..=i], &q[i..]);
    }

    let c = char::from(b[i].to_ascii_lowercase());
    let Some(row) = KEYS.iter().find(|r| r.contains(c)) else {
        return q.to_owned();
    };

    let j = row.find(c).unwrap_or_default();
    let nb = if rng.random() < 0.5 {
        j.saturating_sub(1)
    } else {
        (j + 1).min(row.len() - 1)
    };

    format!("{}{}{}", &q[..i], &row[nb..=nb], &q[i + 1..])
}

/// `style`: the noise every question gets; no typos when testing or when
/// `typos` is off.
pub(super) fn style(q: &str, rng: &mut PyRandom, test: bool, typos: bool) -> String {
    let mut q = q.to_owned();
    if typos && !test && rng.random() < 0.10 {
        q = typo(&q, rng);
    }

    if rng.random() < 0.18 {
        q = format!("{}{q}", rng.choice(PREFIX));
    }

    let r = rng.random();
    if r < 0.45 {
        q = q.to_lowercase();
    } else if r < 0.75 {
        q = cap(&q);
    } else if r < 0.81 {
        q = q.to_uppercase();
    }

    if rng.random() < 0.5 {
        q.push('?');
    } else if rng.random() < 0.08 {
        q.push_str(rng.choice(SUFFIX));
    }

    if rng.random() < 0.03 {
        q = format!("  {}", q.replacen(' ', "  ", 1));
    }

    q
}
