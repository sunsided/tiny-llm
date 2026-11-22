//! The Rust generator reproduces `gen2.py` of TINY 2.1 line for line.
//!
//! Fixtures, made with Python 3.11 from inside `reference-tiny-2.1/training`
//! (`-I` drops the current directory from `sys.path`, hence the insert):
//!
//! - `eval.jsonl`: the logic of `train2.eval_set(60, 123)`, printed as JSON
//!   lines `{"q", "refs", "kind"}`.
//! - `samples-seed1.tsv`:
//!   `python3 -I -c "import sys; sys.path.insert(0, '.'); import random, gen2; rng = random.Random(1)
//!   [print(k, q, a, sep='\t') for q, a, k in (gen2.sample(rng) for _ in range(20000))]"`
//! - `samples-test-seed7.tsv`:
//!   `python3 -I -c "import sys; sys.path.insert(0, '.'); import random, gen2; rng = random.Random(7)
//!   [print(k, q, a, sep='\t') for n, _, _ in gen2.GENS for q, a, k in (gen2.sample(rng, True, n) for _ in range(50))]"`

use std::path::Path;

use tiny_train::pyrandom::PyRandom;
use tiny_train::questions::{KINDS, Kit, Sample};

/// Loads the kit from the bundled copy of the original fact files.
fn kit() -> Kit {
    Kit::load(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/facts"
    )))
    .expect("facts")
}

/// Reads a fixture file.
fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read_to_string(path).expect("fixture")
}

/// Asserts equal line lists, reporting the first difference with both sides.
fn assert_lines(want: &[String], got: &[String]) {
    if let Some(i) = (0..want.len().min(got.len())).find(|&i| want[i] != got[i]) {
        let before = if i > 0 { want[i - 1].as_str() } else { "" };
        panic!(
            "line {} differs (previous: {before:?})\n  python: {:?}\n  rust:   {:?}",
            i + 1,
            want[i],
            got[i]
        );
    }

    assert_eq!(want.len(), got.len(), "line count");
}

/// A sample as the fixture's `kind<TAB>question<TAB>answer` line.
fn line(s: &Sample) -> String {
    format!("{}\t{}\t{}", s.kind, s.question, s.answer)
}

/// `eval_set(60, 123)` is the frozen evaluation set of the original kit.
#[test]
fn eval_set_matches_python() {
    let got: Vec<String> = kit()
        .eval_set(60, 123)
        .into_iter()
        .map(|e| serde_json::json!({"q": e.question, "refs": e.refs, "kind": e.kind}).to_string())
        .collect();
    let want: Vec<String> = fixture("eval.jsonl")
        .lines()
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l)
                .expect("json")
                .to_string()
        })
        .collect();
    assert_lines(&want, &got);
}

/// 20000 training samples from seed 1 match Python.
#[test]
fn training_samples_match_python() {
    let kit = kit();
    let mut rng = PyRandom::new(1);
    let got: Vec<String> = (0..20_000)
        .map(|_| line(&kit.sample(&mut rng, false, None)))
        .collect();
    let want: Vec<String> = fixture("samples-seed1.tsv")
        .lines()
        .map(str::to_owned)
        .collect();
    assert_lines(&want, &got);
}

/// 50 test samples of every kind from seed 7 match Python.
#[test]
fn test_samples_per_kind_match_python() {
    let kit = kit();
    let mut rng = PyRandom::new(7);
    let got: Vec<String> = KINDS
        .iter()
        .flat_map(|(name, _)| {
            (0..50)
                .map(|_| line(&kit.sample(&mut rng, true, Some(name))))
                .collect::<Vec<_>>()
        })
        .collect();
    let want: Vec<String> = fixture("samples-test-seed7.tsv")
        .lines()
        .map(str::to_owned)
        .collect();
    assert_lines(&want, &got);
}

/// A facts directory without `geo.tsv` fails to load with an error naming the
/// missing relation, instead of panicking later in a generator.
#[test]
fn missing_relation_is_an_error() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/facts");
    let dir = std::env::temp_dir().join(format!("tiny-train-facts-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    for entry in std::fs::read_dir(&source).expect("facts") {
        let path = entry.expect("entry").path();
        if path.file_name().is_some_and(|n| n != "geo.tsv") {
            std::fs::copy(&path, dir.join(path.file_name().expect("name"))).expect("copy");
        }
    }

    let error = Kit::load(&dir).map(|_| ()).expect_err("geo.tsv is missing");
    std::fs::remove_dir_all(&dir).expect("cleanup");

    assert!(error.to_string().contains("capital"), "{error}");
}
