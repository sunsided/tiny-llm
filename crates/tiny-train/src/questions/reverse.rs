//! The reverse generators of `gen2.py`: given the answer of a table relation,
//! name the subject (`g_reverse`, `g_reverse2`).

use super::{Kit, art, cap, fmt, list_and, pick_q};
use crate::pyrandom::PyRandom;

/// The kinds of reverse question, repeated by weight.
const REVERSE_KINDS: &[&str] = &[
    "cap", "cap", "cap", "lang", "cur", "sound", "baby", "plural", "past", "month",
];

/// Which country has this capital.
const CAP_Q: &[&str] = &[
    "{a} is the capital of which country",
    "which country has the capital {a}",
    "what country is {a} the capital of",
    "{a} is in which country",
    "where is {a}",
    "in which country is {a}",
    "which country is {a} in",
    "{a} is the capital of what",
    "of which country is {a} the capital",
];

/// Answers naming the country of a capital.
const CAP_A: &[&str] = &[
    "{s}.",
    "{a} is in {s}.",
    "{a} is the capital of {s}.",
    "It is in {s}.",
];

/// Where a language is spoken.
const LANG_Q: &[&str] = &[
    "where do people speak {a}",
    "in which country do people speak {a}",
    "which country speaks {a}",
    "where is {a} spoken",
    "who speaks {a}",
    "in which countries is {a} spoken",
];

/// Where a currency is used.
const CUR_Q: &[&str] = &[
    "which country uses the {a}",
    "where do people pay with the {a}",
    "where is the {a} used",
    "which country has the {a}",
    "the {a} is the money of which country",
    "what country uses the {a}",
];

/// Which animal makes a sound.
const SOUND_Q: &[&str] = &[
    "which animal says {a}",
    "what animal says {a}",
    "who says {a}",
    "what animal goes {a}",
    "which animal makes the sound {a}",
    "what says {a}",
];

/// Whose baby this is; `{x}` has its article.
const BABY_Q: &[&str] = &[
    "{x} is the baby of which animal",
    "whose baby is called {x}",
    "which animal has a baby called {x}",
    "what animal does {x} grow up to be",
    "{x} is a baby what",
    "what is {x} the baby of",
];

/// The singular of a plural.
const PLURAL_Q: &[&str] = &[
    "what is the singular of {a}",
    "what is one of {a} called",
    "singular of {a}",
    "{a} is the plural of what",
    "what is the singular form of {a}",
];

/// Answers naming the singular.
const PLURAL_A: &[&str] = &["{S}.", "One {s}, two {a}.", "It is {s}."];

/// The present of a past tense.
const PAST_Q: &[&str] = &[
    "{a} is the past tense of which word",
    "what is the present tense of {a}",
    "{a} is the past of what",
    "which word has the past tense {a}",
    "what is {a} the past tense of",
];

/// Answers naming the present.
const PAST_A: &[&str] = &["{S}.", "The past of {s} is {a}.", "It is {s}."];

/// The position of a month.
const MONTH_Q: &[&str] = &[
    "which month of the year is {a}",
    "what number month is {a}",
    "where is {a} in the year",
    "is {a} the first month",
    "which month number is {a}",
];

/// The kinds of second-batch reverse question, repeated by weight.
const REVERSE2_KINDS: &[&str] = &["sym", "sym", "atom", "dom", "state", "mean"];

/// Which element has a symbol.
const SYM_Q: &[&str] = &[
    "which element has the symbol {a}",
    "what element is {a}",
    "{a} is the symbol for what",
    "what does the symbol {a} stand for",
    "which element is {a}",
];

/// Answers naming the element of a symbol.
const SYM_A: &[&str] = &["{S}.", "{a} is {s}.", "It is {s}."];

/// Which element has an atomic number.
const ATOM_Q: &[&str] = &[
    "what is element number {a}",
    "which element has atomic number {a}",
    "element {a}",
    "what element has {a} protons",
];

/// Answers naming the element of an atomic number.
const ATOM_A: &[&str] = &["{S}.", "Element {a} is {s}."];

/// Which country has an internet domain.
const DOM_Q: &[&str] = &[
    "which country uses the domain {a}",
    "{a} is the internet domain of which country",
    "what country has websites ending in {a}",
    "whose domain is {a}",
];

/// Which state has a capital.
const STATE_Q: &[&str] = &[
    "{a} is the capital of which state",
    "which state has the capital {a}",
    "what state is {a} the capital of",
];

/// Answers naming the state.
const STATE_A: &[&str] = &["{s}.", "{a} is the capital of {s}."];

/// What a foreign word means.
const MEAN_Q: &[&str] = &[
    "what does {x} mean",
    "what does {x} mean in english",
    "translate {x} into english",
    "what is {x} in english",
];

/// Answers with the meanings.
const MEAN_A: &[&str] = &["It means {m}.", "{X} means {m}."];

/// `list_and` of animals with articles, the first capitalised.
fn animals(ss: &[String]) -> String {
    let named: Vec<String> = ss
        .iter()
        .enumerate()
        .map(|(i, x)| if i == 0 { cap(&art(x)) } else { art(x) })
        .collect();
    format!("{}.", list_and(&named))
}

/// Up to three names, else the first two and `more`.
fn names(ss: &[String], more: &str) -> String {
    if ss.len() <= 3 {
        list_and(ss)
    } else {
        list_and(&[ss[0].as_str(), ss[1].as_str(), more])
    }
}

/// A reverse question from a pair list: `{a}` asked, `{s}`/`{S}` answered.
fn pair_q(
    pairs: &[(String, String)],
    qs: &[&str],
    ans: &[&str],
    rng: &mut PyRandom,
    test: bool,
) -> (String, String) {
    let (a, s) = rng.choice(pairs);
    let q = fmt(pick_q(qs, rng, test), &[("a", a)]);
    let ans = fmt(
        rng.choice::<&str>(ans),
        &[("a", a), ("s", s), ("S", &cap(s))],
    );
    (q, ans)
}

impl Kit {
    /// `g_reverse`: capitals, languages, currencies, sounds, babies,
    /// plurals, pasts and months backwards.
    pub(super) fn g_reverse(&self, rng: &mut PyRandom, test: bool) -> (String, String) {
        match *rng.choice(REVERSE_KINDS) {
            "cap" => pair_q(&self.rev.cap_rev, CAP_Q, CAP_A, rng, test),
            "lang" => {
                let (a, ss) = rng.choice(&self.rev.lang_rev);
                let q = fmt(pick_q(LANG_Q, rng, test), &[("a", a)]);
                (q, format!("In {}.", names(ss, "other countries")))
            }
            "cur" => {
                let (a, ss) = rng.choice(&self.rev.cur_rev);
                let q = fmt(pick_q(CUR_Q, rng, test), &[("a", a)]);
                (q, format!("{}.", names(ss, "many others")))
            }
            "sound" => {
                let (a, ss) = rng.choice(&self.rev.sound_rev);
                (fmt(pick_q(SOUND_Q, rng, test), &[("a", a)]), animals(ss))
            }
            "baby" => {
                let (a, ss) = rng.choice(&self.rev.baby_rev);
                (
                    fmt(pick_q(BABY_Q, rng, test), &[("x", &art(a))]),
                    animals(ss),
                )
            }
            "plural" => pair_q(&self.rev.plural_rev, PLURAL_Q, PLURAL_A, rng, test),
            "past" => pair_q(&self.rev.past_rev, PAST_Q, PAST_A, rng, test),
            _ => {
                let (a, s) = rng.choice(&self.rev.monthnum);
                let q = fmt(pick_q(MONTH_Q, rng, test), &[("a", a)]);
                (q, format!("{a} is the {s} month."))
            }
        }
    }

    /// `g_reverse2`: element symbols and numbers, domains, state capitals and
    /// foreign words backwards; no phrasings are held out.
    pub(super) fn g_reverse2(&self, rng: &mut PyRandom) -> (String, String) {
        let simple = |pairs: &[(String, String)], qs: &[&str], ans: &[&str], rng: &mut PyRandom| {
            let (a, s) = rng.choice(pairs);
            let q = fmt(rng.choice::<&str>(qs), &[("a", a)]);
            (
                q,
                fmt(
                    rng.choice::<&str>(ans),
                    &[("a", a), ("s", s), ("S", &cap(s))],
                ),
            )
        };

        match *rng.choice(REVERSE2_KINDS) {
            "sym" => simple(&self.rev.sym_rev, SYM_Q, SYM_A, rng),
            "atom" => simple(&self.rev.atom_rev, ATOM_Q, ATOM_A, rng),
            "dom" => {
                let (a, s) = rng.choice(&self.rev.dom_rev);
                (fmt(rng.choice::<&str>(DOM_Q), &[("a", a)]), format!("{s}."))
            }
            "state" => simple(&self.rev.state_rev, STATE_Q, STATE_A, rng),
            _ => {
                let (x, es) = rng.choice(&self.rev.meaning);
                let q = fmt(rng.choice::<&str>(MEAN_Q), &[("x", x)]);
                let m = es.join(" or ");
                (
                    q,
                    fmt(
                        rng.choice::<&str>(MEAN_A),
                        &[("x", x), ("X", &cap(x)), ("m", &m)],
                    ),
                )
            }
        }
    }
}
