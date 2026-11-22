//! Letters and spelling (`g_letters`) and "I do not know" for made-up names
//! (`g_unknown`) of `gen2.py`.

use super::tables::{ENDS, ORD, SYL};
use super::{Kit, cap, fmt, words};
use crate::pyrandom::PyRandom;

/// The kinds of letter question, repeated by weight.
const KINDS: &[&str] = &[
    "first", "last", "spell", "spell", "back", "back", "caps", "after", "before", "pos", "count",
];

/// The first letter of a word.
const FIRST_Q: &[&str] = &[
    "what is the first letter of {w}",
    "what letter does {w} start with",
    "first letter of {w}",
    "which letter does {w} begin with",
];

/// Answers of the first letter.
const FIRST_A: &[&str] = &["{L}.", "{W} starts with {L}."];

/// The last letter of a word.
const LAST_Q: &[&str] = &[
    "what is the last letter of {w}",
    "what letter does {w} end with",
    "last letter of {w}",
];

/// Answers of the last letter.
const LAST_A: &[&str] = &["{L}.", "{W} ends with {L}."];

/// Spelling a word.
const SPELL_Q: &[&str] = &[
    "spell {w}",
    "how do you spell {w}",
    "can you spell {w}",
    "spell the word {w}",
];

/// Spelling a word backwards.
const BACK_Q: &[&str] = &[
    "spell {w} backwards",
    "what is {w} backwards",
    "write {w} backwards",
    "reverse the word {w}",
    "say {w} backwards",
];

/// Writing a word in capitals.
const CAPS_Q: &[&str] = &[
    "write {w} in capital letters",
    "write {w} in big letters",
    "{w} in upper case",
    "make {w} uppercase",
];

/// The length of a word.
const COUNT_Q: &[&str] = &[
    "how many letters are in {w}",
    "how many letters does {w} have",
    "how long is the word {w}",
];

/// The letter after another.
const AFTER_Q: &[&str] = &[
    "what letter comes after {l}",
    "which letter is after {l}",
    "what comes after {l} in the alphabet",
    "next letter after {l}",
];

/// The letter before another.
const BEFORE_Q: &[&str] = &[
    "what letter comes before {l}",
    "which letter is before {l}",
    "what comes before {l} in the alphabet",
];

/// The position of a letter, or the letter at a position (`{n}`).
const POS_Q: &[&str] = &[
    "what number is the letter {l} in the alphabet",
    "where is {l} in the alphabet",
    "which letter of the alphabet is {l}",
    "what is the {n} letter of the alphabet",
];

/// The kinds of unknown question, repeated by weight.
const UNKNOWN_KINDS: &[&str] = &[
    "country",
    "country",
    "animal",
    "word",
    "person",
    "gibberish",
    "gibberish",
];

/// Questions about a made-up country.
const COUNTRY_Q: &[&str] = &[
    "what is the capital of {W}",
    "which continent is {W} in",
    "what language is spoken in {W}",
    "what is the currency of {W}",
    "where is {W}",
];

/// Answers for a made-up country.
const COUNTRY_A: &[&str] = &[
    "I do not know a country called {W}.",
    "I have never heard of {W}.",
];

/// Questions about a made-up animal.
const ANIMAL_Q: &[&str] = &[
    "how many legs does a {w} have",
    "what does a {w} eat",
    "what is a baby {w} called",
    "what sound does a {w} make",
    "where does a {w} live",
];

/// Answers for a made-up animal.
const ANIMAL_A: &[&str] = &[
    "I do not know an animal called a {w}.",
    "I have never heard of a {w}.",
];

/// Questions about a made-up word.
const WORD_Q: &[&str] = &[
    "what is the opposite of {w}",
    "what is the plural of {w}",
    "what does {w} mean",
    "what is a {w}",
];

/// Questions about a made-up person.
const PERSON_Q: &[&str] = &[
    "who is {W} {V}",
    "who was {W} {V}",
    "what did {W} {V} invent",
];

/// Answers for a made-up person.
const PERSON_A: &[&str] = &[
    "I do not know who that is.",
    "Sorry, I do not know that person.",
];

/// Answers for gibberish.
const GIBBERISH_A: &[&str] = &[
    "Sorry, I did not understand. Please ask a question.",
    "I do not understand. Try again?",
    "Hmm? Please ask me a question.",
];

/// Letters of gibberish: `ascii_lowercase + '  ;,.'`.
const GIBBERISH: &[u8] = b"abcdefghijklmnopqrstuvwxyz  ;,.";

/// The alphabet, `string.ascii_lowercase`.
const ALPHA: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

impl Kit {
    /// `g_letters`: first and last letters, spelling, capitals, length and
    /// the alphabet.
    pub(super) fn g_letters(&self, rng: &mut PyRandom, test: bool) -> (String, String) {
        let k = *rng.choice(KINDS);
        let pool = if test {
            &self.split.words.1
        } else {
            &self.split.words.0
        };

        let w = rng.choice(pool).as_str();
        let wc = cap(w);
        let upper = w.to_uppercase();
        match k {
            "first" | "last" => {
                let (qs, ans, i) = if k == "first" {
                    (FIRST_Q, FIRST_A, 0)
                } else {
                    (LAST_Q, LAST_A, upper.len() - 1)
                };

                let q = fmt(rng.choice(qs), &[("w", w)]);
                (q, fmt(rng.choice(ans), &[("L", &upper[i..=i]), ("W", &wc)]))
            }
            "spell" => {
                let letters: Vec<String> = upper.chars().map(String::from).collect();
                (
                    fmt(rng.choice(SPELL_Q), &[("w", w)]),
                    format!("{}.", letters.join("-")),
                )
            }
            "back" => {
                let back: String = w.chars().rev().collect();
                (
                    fmt(rng.choice(BACK_Q), &[("w", w)]),
                    format!("{}.", cap(&back)),
                )
            }
            "caps" => (fmt(rng.choice(CAPS_Q), &[("w", w)]), format!("{upper}.")),
            "count" => {
                let q = fmt(rng.choice(COUNT_Q), &[("w", w)]);
                let n = words(i64::try_from(w.len()).expect("short word"));
                let opts = [format!("{}.", cap(&n)), format!("{wc} has {n} letters.")];
                (q, rng.choice(&opts).clone())
            }
            _ => alphabet(k, rng),
        }
    }

    /// `g_unknown`: made-up countries, animals, words, people and gibberish.
    pub(super) fn g_unknown(&self, rng: &mut PyRandom) -> (String, String) {
        let k = *rng.choice(UNKNOWN_KINDS);
        let w = self.fake(rng, k == "country");
        let wc = cap(&w);
        match k {
            "country" => {
                let t = rng.choice(COUNTRY_Q);
                let q = fmt(t, &[("W", if rng.random() < 0.5 { &wc } else { &w })]);
                (q, fmt(rng.choice(COUNTRY_A), &[("W", &wc)]))
            }
            "animal" => {
                let q = fmt(rng.choice(ANIMAL_Q), &[("w", &w)]);
                (q, fmt(rng.choice(ANIMAL_A), &[("w", &w)]))
            }
            "word" => (
                fmt(rng.choice(WORD_Q), &[("w", &w)]),
                format!("I do not know the word {w}."),
            ),
            "person" => {
                let t = rng.choice(PERSON_Q);
                let v = cap(&self.fake(rng, false));
                (
                    fmt(t, &[("W", &wc), ("V", &v)]),
                    (*rng.choice(PERSON_A)).to_owned(),
                )
            }
            _ => {
                let n = rng.randint(3, 14);
                let raw: String = (0..n).map(|_| char::from(*rng.choice(GIBBERISH))).collect();
                let q = raw.trim();
                let q = if q.is_empty() { "xq" } else { q };
                (q.to_owned(), (*rng.choice(GIBBERISH_A)).to_owned())
            }
        }
    }

    /// `fake`: a made-up word of 4 to 11 letters that is no real name.
    fn fake(&self, rng: &mut PyRandom, country: bool) -> String {
        loop {
            let n = rng.randint(2, 3) - i64::from(country);
            let mut w: String = (0..n).map(|_| *rng.choice(SYL)).collect();
            if country && rng.random() < 0.7 {
                w.push_str(rng.choice(ENDS));
            }

            if !self.real.contains(&w) && (4..=11).contains(&w.len()) {
                return w;
            }
        }
    }
}

/// The alphabet kinds of `g_letters`; `after z` and `before a` become
/// position questions.
fn alphabet(k: &str, rng: &mut PyRandom) -> (String, String) {
    let i = rng.randrange(26);
    let lower = char::from(ALPHA[i]).to_string();
    let upper = lower.to_uppercase();
    let next = |j: usize| format!("{}.", char::from(ALPHA[j].to_ascii_uppercase()));
    if k == "after" && i < 25 {
        let t = rng.choice(AFTER_Q);
        let l = if rng.random() < 0.5 { &lower } else { &upper };
        return (fmt(t, &[("l", l)]), next(i + 1));
    }

    if k == "before" && i > 0 {
        let t = rng.choice(BEFORE_Q);
        let l = if rng.random() < 0.5 { &lower } else { &upper };
        return (fmt(t, &[("l", l)]), next(i - 1));
    }

    let q = *rng.choice(POS_Q);
    if q.contains("{n}") {
        return (fmt(q, &[("n", ORD[i + 1])]), format!("{upper}."));
    }

    (
        fmt(q, &[("l", &upper)]),
        format!("{upper} is letter number {}.", i + 1),
    )
}
