//! Scoring answers like `train2.py`'s `correct()`, and evaluating a network
//! on the frozen 780-question test set of the original kit.
//!
//! A reference answer counts as matched when every key word of it appears in
//! the generated answer. Key words are the reference's words minus a stop list
//! and minus words already in the question. Yes/no answers must also start
//! with the same yes or no, and "I do not know" answers must start like one
//! and repeat the made-up name from the question.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::Context;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Deserialize;
use tiny::model::Network;
use tiny::think::answer;

/// Words that never count as key words of a reference answer.
const STOP: &[&str] = &[
    "a", "an", "the", "it", "is", "are", "was", "of", "in", "on", "and", "to", "its", "it's",
    "they", "there", "people", "speak", "use", "money", "capital", "city", "called", "comes",
    "after", "before", "has", "have", "says", "say", "lives", "eats", "one", "two", "answer",
    "number", "letter", "letters", "starts", "ends", "with", "month", "day",
];

/// How "I do not know" answers start.
const UNKNOWN_START: &[&str] = &["i do not", "i have never", "sorry", "hmm"];

/// One question of a test set with its accepted answers.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Item {
    /// The question, as fed to the network (not cleaned).
    #[serde(rename = "q")]
    pub question: String,
    /// Accepted reference answers; any one of them may match.
    pub refs: Vec<String>,
    /// Generator that made the question (`facts`, `tables`, `unknown`, ...).
    pub kind: String,
}

/// Per-kind and total results of an evaluation.
#[derive(Debug, Clone, Default)]
pub struct Score {
    /// Correct and total answers per kind, in kind name order.
    pub by_kind: BTreeMap<String, (usize, usize)>,
    /// The questions answered wrongly: (kind, question, answer, first reference).
    pub misses: Vec<(String, String, String, String)>,
}

impl Score {
    /// Share of correct answers over all kinds, 0..=1.
    #[must_use]
    pub fn total(&self) -> f64 {
        let (right, all) = self
            .by_kind
            .values()
            .fold((0, 0), |(r, a), &(right, all)| (r + right, a + all));

        ratio(right, all)
    }

    /// `kind 61%  kind 40% ...` like the original's eval line.
    #[must_use]
    pub fn line(&self) -> String {
        let parts: Vec<String> = self
            .by_kind
            .iter()
            .map(|(kind, &(right, all))| format!("{kind} {:.0}%", 100.0 * ratio(right, all)))
            .collect();

        parts.join("  ")
    }
}

/// Reads a test set stored as JSON lines (`{"q", "refs", "kind"}`).
///
/// # Errors
///
/// Fails when the file cannot be read or a line is not such an object.
pub fn load(path: &Path) -> anyhow::Result<Vec<Item>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    text.lines()
        .enumerate()
        .map(|(i, line)| serde_json::from_str(line).with_context(|| format!("line {}", i + 1)))
        .collect()
}

/// Answers every question greedily (creativity 0) and scores the answers.
#[must_use]
pub fn evaluate(net: &mut Network, items: &[Item]) -> Score {
    let mut rng = StdRng::seed_from_u64(0);
    let mut score = Score::default();

    for item in items {
        let generated = answer(net, &item.question, 0, &mut rng);
        let ok = correct(&item.question, &generated, &item.refs, &item.kind);
        let entry = score.by_kind.entry(item.kind.clone()).or_default();
        entry.0 += usize::from(ok);
        entry.1 += 1;
        if !ok {
            let first = item.refs.first().cloned().unwrap_or_default();
            score
                .misses
                .push((item.kind.clone(), item.question.clone(), generated, first));
        }
    }

    score
}

/// Whether `generated` answers `question` as well as one of `refs` (`train2.correct`).
#[must_use]
pub fn correct(question: &str, generated: &str, refs: &[String], kind: &str) -> bool {
    let first = refs.first().map(|r| r.to_lowercase()).unwrap_or_default();
    if kind == "unknown" && starts_unknown(&first) {
        return correct_unknown(question, generated, &first);
    }

    let gen_words = words(generated);
    let gen_set: HashSet<&str> = gen_words.iter().map(String::as_str).collect();
    let question_set: HashSet<String> = words(question).into_iter().collect();

    refs.iter().any(|r| {
        let ref_words = words(r);
        if is_yes_no(&ref_words) && gen_words.first() != ref_words.first() {
            return false;
        }

        key_words(&ref_words, &question_set)
            .iter()
            .all(|w| gen_set.contains(w.as_str()))
    })
}

/// An "I do not know" answer must start like one and repeat the question's
/// words that also appear in the reference (the made-up name).
fn correct_unknown(question: &str, generated: &str, reference: &str) -> bool {
    if !starts_unknown(&generated.to_lowercase()) {
        return false;
    }

    let question_words = words(question);
    let gen_words = words(generated);

    words(reference)
        .iter()
        .filter(|w| question_words.contains(w))
        .all(|w| gen_words.contains(w))
}

/// Key words of a reference: no stop words and nothing from the question;
/// failing that, no stop words; failing that, all words.
fn key_words(ref_words: &[String], question: &HashSet<String>) -> Vec<String> {
    let not_stop = |w: &&String| !STOP.contains(&w.as_str());
    let key: Vec<String> = ref_words
        .iter()
        .filter(not_stop)
        .filter(|w| !question.contains(*w))
        .cloned()
        .collect();
    if !key.is_empty() {
        return key;
    }

    let key: Vec<String> = ref_words.iter().filter(not_stop).cloned().collect();
    if key.is_empty() {
        ref_words.to_vec()
    } else {
        key
    }
}

/// Whether a reference starts with "yes" or "no".
fn is_yes_no(ref_words: &[String]) -> bool {
    matches!(ref_words.first().map(String::as_str), Some("yes" | "no"))
}

/// Whether lower-case text starts like an "I do not know" answer.
fn starts_unknown(lower: &str) -> bool {
    UNKNOWN_START.iter().any(|s| lower.starts_with(s))
}

/// The runs of `[a-z0-9]` in the lower-cased text (`re.findall(r"[a-z0-9]+", s.lower())`).
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit()))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

/// `right / all`, 0 for an empty set.
fn ratio(right: usize, all: usize) -> f64 {
    let to_f64 = |n: usize| f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    if all == 0 {
        0.0
    } else {
        to_f64(right) / to_f64(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Owned reference list.
    fn refs(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Key words must all appear; words from the question are not required.
    #[test]
    fn key_words_decide() {
        let r = refs(&["The capital of Peru is Lima."]);

        assert!(correct(
            "what is the capital of peru",
            "Lima.",
            &r,
            "tables"
        ));
        assert!(!correct(
            "what is the capital of peru",
            "Quito.",
            &r,
            "tables"
        ));
    }

    /// Yes/no references need the same first word.
    #[test]
    fn yes_no_must_agree() {
        let r = refs(&["No, a whale is a mammal."]);

        assert!(correct(
            "is a whale a fish",
            "No, a whale is a mammal.",
            &r,
            "yesno"
        ));
        assert!(!correct(
            "is a whale a fish",
            "Yes, a whale is a mammal.",
            &r,
            "yesno"
        ));
    }

    /// Unknown answers must decline and repeat the made-up name.
    #[test]
    fn unknown_needs_the_name() {
        let r = refs(&["I do not know a country called Zorbia."]);

        assert!(correct(
            "capital of zorbia",
            "I do not know a country called Zorbia.",
            &r,
            "unknown"
        ));
        assert!(!correct(
            "capital of zorbia",
            "I do not know.",
            &r,
            "unknown"
        ));
        assert!(!correct("capital of zorbia", "Paris.", &r, "unknown"));
    }
}
