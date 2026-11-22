//! The question generator of TINY 2.1 (`gen2.py`), ported so that the same
//! seed gives exactly the same stream of (question, answer) pairs.
//!
//! Every generator draws from [`PyRandom`] in the same order as the Python
//! original, with the same retry loops and filters; only then are the streams
//! identical. The fact files are pure ASCII (checked on load), so Python's
//! string methods reduce to their ASCII forms.

mod letters;
mod load;
mod noise;
mod numbers;
mod reverse;
mod tables;
mod world;
mod yesno;

use std::collections::{HashMap, HashSet};

use crate::pyrandom::PyRandom;

/// Longest question plus answer: with two newlines they fill 128 letters.
const MAXLEN: usize = 126;

/// The generators and their share of the training data (`gen2.GENS`).
pub const KINDS: [(&str, f64); 12] = [
    ("facts", 0.23),
    ("tables", 0.31),
    ("landlocked", 0.015),
    ("borders", 0.035),
    ("reverse", 0.05),
    ("reverse2", 0.03),
    ("yesno", 0.08),
    ("combo", 0.04),
    ("numbers", 0.10),
    ("numbers2", 0.02),
    ("letters", 0.06),
    ("unknown", 0.03),
];

/// Kinds whose questions never get typos (`style(..., typos=False)`).
const NO_TYPOS: [&str; 3] = ["letters", "yesno", "numbers2"];

/// One generated training or test pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample {
    /// The question, with typing noise applied.
    pub question: String,
    /// The answer.
    pub answer: String,
    /// The generator that made it, one of [`KINDS`].
    pub kind: &'static str,
}

/// One question of the frozen evaluation set, with every accepted answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalItem {
    /// The question.
    pub question: String,
    /// The reference answers.
    pub refs: Vec<String>,
    /// The generator that made it, one of [`KINDS`].
    pub kind: &'static str,
}

/// A one-off fact from `facts/*.txt`.
struct Fact {
    /// Phrasings for training.
    train_q: Vec<String>,
    /// The phrasing kept back for testing (empty with fewer than three).
    test_q: Vec<String>,
    /// Accepted answers.
    answers: Vec<String>,
}

/// (subject, answer) pairs in file order, duplicates kept.
type Pairs = Vec<(String, String)>;

/// A table relation the `tables` generator asks about.
struct Relation {
    /// The relation name, e.g. `capital`.
    name: String,
    /// Question templates; the last one is for testing only.
    questions: Vec<String>,
    /// Answer templates.
    answers: Vec<String>,
}

/// The answer-to-subject tables of `g_reverse` and `g_reverse2`.
struct Reverse {
    /// `list(CAP_REV)` in insertion order: capitals of exactly one country.
    cap_rev_keys: Vec<String>,
    /// `sorted(CAP_REV.items())`.
    cap_rev: Pairs,
    /// `sorted(LANG_REV.items())`.
    lang_rev: Vec<(String, Vec<String>)>,
    /// `sorted(CUR_REV.items())`.
    cur_rev: Vec<(String, Vec<String>)>,
    /// `sorted(SOUND_REV.items())`.
    sound_rev: Vec<(String, Vec<String>)>,
    /// `sorted(BABY_REV.items())`.
    baby_rev: Vec<(String, Vec<String>)>,
    /// `sorted(PLURAL_REV.items())`.
    plural_rev: Pairs,
    /// `sorted(PAST_REV.items())`.
    past_rev: Pairs,
    /// `sorted(MONTHNUM.items())`.
    monthnum: Pairs,
    /// `sorted(SYM_REV.items())`.
    sym_rev: Pairs,
    /// `ATOM_REV.items()` sorted by atomic number.
    atom_rev: Pairs,
    /// `sorted(DOM_REV.items())`.
    dom_rev: Pairs,
    /// `sorted(STATE_REV.items())`.
    state_rev: Pairs,
    /// `sorted(MEANING.items())`: foreign word to English meanings.
    meaning: Vec<(String, Vec<String>)>,
}

/// A train and a test half, both sorted.
type Split<T> = (Vec<T>, Vec<T>);

/// The deterministic train/test splits of `gen2.py`.
struct Splits {
    /// `YN_COUNTRIES_TRAIN` and `YN_COUNTRIES_TEST`.
    yn_countries: Split<String>,
    /// `ADD_TRAIN` and `ADD_TEST`.
    add: Split<(i64, i64)>,
    /// `SUB_TRAIN` and `SUB_TEST`.
    sub: Split<(i64, i64)>,
    /// `MUL_TRAIN` and `MUL_TEST`.
    mul: Split<(i64, i64)>,
    /// `WORDS_TRAIN` and `WORDS_TEST`.
    words: Split<String>,
    /// `ROMAN_TRAIN` and `ROMAN_TEST`.
    roman: Split<i64>,
    /// `SQ_TRAIN` and `SQ_TEST`.
    sq: Split<i64>,
}

/// The loaded fact files and every table `gen2.py` derives from them.
///
/// Field names follow the Python globals; lists are in the order Python
/// iterates them, so draws pick the same element.
pub struct Kit {
    /// `TXT`: the one-off facts.
    txt: Vec<Fact>,
    /// `REL`: relations in first-seen order with their pairs.
    rel: Vec<(String, Pairs)>,
    /// `D`: relation to subject to answer, the last pair winning.
    d: HashMap<String, HashMap<String, String>>,
    /// `REL_NAMES` with their templates.
    rels: Vec<Relation>,
    /// `REL_W` in the order of `rels`.
    rel_weights: Vec<f64>,
    /// `BORDERS`: country to neighbours.
    borders: HashMap<String, Vec<String>>,
    /// `dict(REL['plural'])` for `plural_of`.
    plural: HashMap<String, String>,
    /// The tables of the reverse questions.
    rev: Reverse,
    /// `CLASSES`.
    classes: Vec<String>,
    /// `CONTS`.
    conts: Vec<String>,
    /// `CAPS_BY_CONT`, keyed by `D['continent'].get(country)`.
    caps_by_cont: HashMap<Option<String>, Vec<String>>,
    /// `LANGS`.
    langs: Vec<String>,
    /// `sorted(set(D['sound'].values()))`.
    sounds: Vec<String>,
    /// `sorted(set(D['opposite'].values()))`.
    opposites: Vec<String>,
    /// `sorted(set(D['legs']) & set(D['sound']))`.
    legs_sound: Vec<String>,
    /// `sorted(set(D['baby']) & set(D['class']))`.
    baby_class: Vec<String>,
    /// `COUNTRIES`.
    countries: Vec<String>,
    /// The train/test splits.
    split: Splits,
    /// `REAL`: words a made-up word must not be.
    real: HashSet<String>,
}

impl Kit {
    /// `gen2.sample`: draws generators (or uses `kind`) until a styled pair
    /// passes the length and printable-ASCII filters.
    ///
    /// # Panics
    /// If `kind` is not one of [`KINDS`].
    pub fn sample(&self, rng: &mut PyRandom, test: bool, kind: Option<&str>) -> Sample {
        let weights = KINDS.map(|(_, w)| w);
        loop {
            let i = match kind {
                Some(k) => KINDS
                    .iter()
                    .position(|(n, _)| *n == k)
                    .unwrap_or_else(|| panic!("unknown kind {k:?}")),
                None => rng.choices_index(&weights),
            };

            let name = KINDS[i].0;
            let (q, a) = self.generate(i, rng, test);
            let q = noise::style(&q, rng, test, !NO_TYPOS.contains(&name));
            let printable = q.bytes().chain(a.bytes()).all(|c| (32..=126).contains(&c));
            if q.len() <= 80 && q.len() + a.len() <= MAXLEN && printable {
                return Sample {
                    question: q,
                    answer: a,
                    kind: name,
                };
            }
        }
    }

    /// Runs generator `i` of [`KINDS`].
    fn generate(&self, i: usize, rng: &mut PyRandom, test: bool) -> (String, String) {
        match i {
            0 => self.g_txt(rng, test),
            1 => self.g_rel(rng, test),
            2 => self.g_landlocked(rng, test),
            3 => self.g_borders(rng, test),
            4 => self.g_reverse(rng, test),
            5 => self.g_reverse2(rng),
            6 => self.g_yesno(rng, test),
            7 => self.g_combo(rng, test),
            8 => self.g_numbers(rng, test),
            9 => self.g_numbers2(rng, test),
            10 => self.g_letters(rng, test),
            _ => self.g_unknown(rng),
        }
    }

    /// `train2.eval_set`: the held-out phrasing of up to 120 shuffled one-off
    /// facts, then `n_per_kind` distinct test questions of every other kind.
    #[must_use]
    pub fn eval_set(&self, n_per_kind: usize, seed: u64) -> Vec<EvalItem> {
        let mut rng = PyRandom::new(seed);
        let mut out: Vec<EvalItem> = self
            .txt
            .iter()
            .filter_map(|f| {
                Some(EvalItem {
                    question: f.test_q.first()?.clone(),
                    refs: f.answers.clone(),
                    kind: "facts",
                })
            })
            .collect();
        rng.shuffle(&mut out);
        out.truncate(120);

        for (name, _) in &KINDS[1..] {
            let mut asked = HashSet::new();
            let mut count = 0;
            while count < n_per_kind {
                let sample = self.sample(&mut rng, true, Some(*name));
                if asked.insert(sample.question.to_lowercase()) {
                    out.push(EvalItem {
                        question: sample.question,
                        refs: vec![sample.answer],
                        kind: sample.kind,
                    });
                    count += 1;
                }
            }
        }

        out
    }

    /// `REL[r]`, empty for an unknown relation like Python's `defaultdict`.
    fn rel(&self, r: &str) -> &[(String, String)] {
        self.rel
            .iter()
            .find(|(n, _)| n == r)
            .map_or(&[], |(_, p)| p.as_slice())
    }

    /// `D[r]`: subject to answer.
    ///
    /// # Panics
    /// If the relation is missing (Python raises `KeyError`).
    fn d(&self, r: &str) -> &HashMap<String, String> {
        &self.d[r]
    }

    /// `D[r][s]`.
    fn dv(&self, r: &str, s: &str) -> &str {
        &self.d(r)[s]
    }

    /// `plural_of`: the table plural, else the English rule.
    fn plural_of(&self, w: &str) -> String {
        if let Some(p) = self.plural.get(w) {
            return p.clone();
        }

        let b = w.as_bytes();
        if w.ends_with('y') && b.len() >= 2 && !b"aeiou".contains(&b[b.len() - 2]) {
            return format!("{}ies", &w[..w.len() - 1]);
        }

        naive_plural(w)
    }

    /// `fill`: a relation template with every field of subject and answer.
    fn fill(&self, t: &str, s: &str, a: &str) -> String {
        let art_a = art(a);
        fmt(
            t,
            &[
                ("s", s),
                ("S", &cap(s)),
                ("a", a),
                ("A", &cap(a)),
                ("art", &art_a),
                ("Art", &cap(&art_a)),
                ("p", &self.plural_of(s)),
            ],
        )
    }
}

/// Python's `str.format` with named fields only (no specs, no `{{`), which
/// is all `gen2.py` uses.
///
/// # Panics
/// On a field missing from `args` (Python raises `KeyError`).
fn fmt(t: &str, args: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(t.len() + 16);
    let mut rest = t;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let close = open + rest[open..].find('}').expect("unclosed field");
        let name = &rest[open + 1..close];
        let (_, value) = args
            .iter()
            .find(|(k, _)| *k == name)
            .unwrap_or_else(|| panic!("no field {name} for {t:?}"));
        out.push_str(value);
        rest = &rest[close + 1..];
    }

    out.push_str(rest);
    out
}

/// `cap`: first letter upper case, the rest untouched.
fn cap(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map_or_else(String::new, |f| f.to_uppercase().chain(c).collect())
}

/// `art`: the word with `a` or `an`.
fn art(w: &str) -> String {
    let l = w.to_lowercase();
    let a_words = ["ewe", "uni", "use", "eu", "one", "ur"];
    let vowel = l.starts_with(['a', 'e', 'i', 'o', 'u']) || l.starts_with("hour");
    if a_words.iter().any(|p| l.starts_with(p)) || !vowel {
        format!("a {w}")
    } else {
        format!("an {w}")
    }
}

/// The plural by appending `s` or `es` only.
fn naive_plural(s: &str) -> String {
    let es = ["s", "x", "ch", "sh"].iter().any(|e| s.ends_with(e));
    format!("{s}{}", if es { "es" } else { "s" })
}

/// `words`: a number below 1000 in English words (1000 and up: one thousand).
fn words(n: i64) -> String {
    let i = |n: i64| usize::try_from(n).expect("non-negative number");
    match n {
        ..20 => tables::ONES[i(n)].to_owned(),
        20..100 => {
            let ones = if n % 10 == 0 {
                String::new()
            } else {
                format!("-{}", tables::ONES[i(n % 10)])
            };

            format!("{}{ones}", tables::TENS[i(n / 10)])
        }
        100..1000 => {
            let rest = if n % 100 == 0 {
                String::new()
            } else {
                format!(" and {}", words(n % 100))
            };

            format!("{} hundred{rest}", tables::ONES[i(n / 100)])
        }
        _ => "one thousand".to_owned(),
    }
}

/// `list_and`: `a, b and c`.
fn list_and<S: AsRef<str>>(xs: &[S]) -> String {
    let xs: Vec<&str> = xs.iter().map(AsRef::as_ref).collect();
    match xs.split_last() {
        Some((last, [])) => (*last).to_owned(),
        Some((last, init)) => format!("{} and {last}", init.join(", ")),
        None => String::new(),
    }
}

/// `pick_q`: the last phrasing when testing, else any other.
fn pick_q<'a, S: AsRef<str>>(qs: &'a [S], rng: &mut PyRandom, test: bool) -> &'a str {
    if test {
        qs[qs.len() - 1].as_ref()
    } else {
        rng.choice(&qs[..qs.len() - 1]).as_ref()
    }
}

/// `other`: draws until `keep` accepts the element (Python compares with
/// one value to avoid).
fn other<'a, T>(items: &'a [T], keep: impl Fn(&T) -> bool, rng: &mut PyRandom) -> &'a T {
    loop {
        let x = rng.choice(items);
        if keep(x) {
            return x;
        }
    }
}

/// `split`: a deterministic train/test split, both halves sorted.
fn split<T: Ord + Clone>(mut items: Vec<T>, frac: f64, seed: u64) -> Split<T> {
    items.sort();
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "mirrors Python's int(len(items) * frac) on small lists"
    )]
    let k = ((items.len() as f64 * frac) as usize).max(1);
    let mut test = PyRandom::new(seed).sample(&items, k);
    test.sort();
    test.dedup();
    let train = items
        .into_iter()
        .filter(|x| test.binary_search(x).is_err())
        .collect();
    (train, test)
}
