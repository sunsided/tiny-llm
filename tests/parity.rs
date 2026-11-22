//! Greedy answers must match the original `TINY.C` (built with gcc, plain
//! mode, creativity 0) letter for letter on the README sample questions.

use tiny::model::{Network, Weights};
use tiny::text::clean;
use tiny::think::answer;

/// `question<TAB>answer` lines produced by the C program.
const ORACLE: &str = include_str!("fixtures/greedy.tsv");

/// Every oracle question gets exactly the oracle answer.
#[test]
fn greedy_answers_match_tiny_c() {
    let mut net = Network::new(Weights::from_bytes(tiny::BUILTIN_MODEL).expect("built-in model"));
    let mut rng = rand::rng();

    let mismatches: Vec<String> = ORACLE
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .filter_map(|(q, want)| {
            let got = answer(&mut net, &clean(q), 0, &mut rng);
            (got != want).then(|| format!("{q:?}: got {got:?}, want {want:?}"))
        })
        .collect();

    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    assert_eq!(ORACLE.lines().count(), 44);
}

/// The released file parses with the expected shape and size.
#[test]
fn builtin_shape() {
    let config = Weights::from_bytes(tiny::BUILTIN_MODEL)
        .expect("built-in model")
        .config;

    assert_eq!(
        (config.width, config.layers, config.heads, config.hidden),
        (96, 4, 4, 256)
    );
    assert_eq!(config.parameter_count(), 366_528);
}

/// A truncated or extended file is rejected.
#[test]
fn rejects_wrong_length() {
    let bytes = tiny::BUILTIN_MODEL;

    assert!(Weights::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    assert!(Weights::from_bytes(&[bytes, &[0]].concat()).is_err());
    assert!(Weights::from_bytes(b"TNY1").is_err());
}
