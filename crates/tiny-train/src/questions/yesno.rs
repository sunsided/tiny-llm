//! `g_yesno` of `gen2.py`: yes/no questions, true or false with equal odds,
//! where a false claim is a plausible wrong value.

use super::{Kit, art, fmt, naive_plural, other};
use crate::pyrandom::PyRandom;

/// The kinds of yes/no question, repeated by weight.
const KINDS: &[&str] = &[
    "cap", "cap", "cap", "cont", "cont", "lang", "class", "class", "legs", "sound", "opp",
    "plural", "past", "days",
];

/// Is this the capital.
const CAP_Q: &[&str] = &[
    "is {x} the capital of {s}",
    "is the capital of {s} {x}",
    "is {x} the capital city of {s}",
    "{x} is the capital of {s}, right",
    "is {s}'s capital {x}",
];

/// Is the country in this continent.
const CONT_Q: &[&str] = &[
    "is {s} in {x}",
    "is {s} a country in {x}",
    "is {s} part of {x}",
    "{s} is in {x}, right",
    "does {s} lie in {x}",
];

/// Is this language spoken there.
const LANG_Q: &[&str] = &[
    "do people speak {x} in {s}",
    "is {x} spoken in {s}",
    "do they speak {x} in {s}",
    "is the language of {s} {x}",
];

/// Is the animal of this class; `{X}` has its article.
const CLASS_Q: &[&str] = &[
    "is a {s} a {x}",
    "is a {s} {X}",
    "are {s}s {x}s",
    "is the {s} a {x}",
];

/// Wrong leg counts.
const LEG_COUNTS: &[&str] = &["two", "four", "six", "eight", "ten"];

/// Does the animal have this many legs.
const LEGS_Q: &[&str] = &[
    "does a {s} have {x} legs",
    "has a {s} got {x} legs",
    "do {s}s have {x} legs",
];

/// Does the animal make this sound.
const SOUND_Q: &[&str] = &["does a {s} say {x}", "does a {s} go {x}", "do {s}s say {x}"];

/// Is this the opposite.
const OPP_Q: &[&str] = &[
    "is {x} the opposite of {s}",
    "is the opposite of {s} {x}",
    "are {s} and {x} opposites",
];

/// Is this the plural.
const PLURAL_Q: &[&str] = &[
    "is the plural of {s} {x}",
    "is {x} the plural of {s}",
    "one {s}, two {x}, right",
];

/// Is this the past tense.
const PAST_Q: &[&str] = &[
    "is the past tense of {s} {x}",
    "is {x} the past of {s}",
    "yesterday i {x}, is that right",
];

/// Does the month have this many days.
const DAYS_Q: &[&str] = &[
    "does {s} have {x} days",
    "are there {x} days in {s}",
    "has {s} got {x} days",
];

/// A question template with `{s}`/`{x}` and its answer for true and false.
type Claim<'a> = (&'a [&'a str], &'a str, &'a str);

impl Kit {
    /// `g_yesno`.
    #[expect(
        clippy::many_single_char_names,
        reason = "names follow gen2.py for side-by-side checking"
    )]
    pub(super) fn g_yesno(&self, rng: &mut PyRandom, test: bool) -> (String, String) {
        let k = *rng.choice(KINDS);
        let yes = rng.random() < 0.5;
        match k {
            "cap" | "cont" | "lang" => self.yn_country(k, yes, rng, test),
            "class" => {
                let (s, a) = rng.choice(self.rel("class"));
                let x = if yes {
                    a.as_str()
                } else {
                    other(&self.classes, |c| c != a, rng)
                };

                let q = fmt(
                    rng.choice::<&str>(CLASS_Q),
                    &[("s", s), ("x", x), ("X", &art(x))],
                );
                let t = if x == a {
                    "Yes, a {s} is {A}."
                } else {
                    "No, a {s} is {A}."
                };

                (q, fmt(t, &[("s", s), ("A", &art(a))]))
            }
            "legs" => self.yn_legs(yes, rng),
            "sound" => {
                let (s, a) = rng.choice(self.rel("sound"));
                let x = if yes {
                    a.as_str()
                } else {
                    other(&self.sounds, |c| c != a, rng)
                };

                claim(
                    s,
                    a,
                    x,
                    (SOUND_Q, "Yes, a {s} says {a}.", "No, a {s} says {a}."),
                    rng,
                )
            }
            "opp" => {
                let (s, a) = rng.choice(self.rel("opposite"));
                let x = if yes {
                    a.as_str()
                } else {
                    other(&self.opposites, |c| c != a, rng)
                };

                claim(
                    s,
                    a,
                    x,
                    (
                        OPP_Q,
                        "Yes, {x} is the opposite of {s}.",
                        "No, the opposite of {s} is {a}.",
                    ),
                    rng,
                )
            }
            "plural" | "past" => self.yn_form(k == "plural", yes, rng),
            _ => self.yn_days(yes, rng),
        }
    }

    /// The country kinds of `g_yesno`: capital, continent or language.
    fn yn_country(&self, k: &str, yes: bool, rng: &mut PyRandom, test: bool) -> (String, String) {
        let mut pool: Vec<&String> = self.yn_pool(test).iter().collect();
        let language = self.d("language");
        if k == "lang" {
            pool.retain(|c| language.contains_key(*c));
            if pool.is_empty() {
                pool = language.keys().collect();
                pool.sort();
            }
        }

        let s = rng.choice(&pool).as_str();
        match k {
            "cap" => {
                let a = self.dv("capital", s);
                let x = if yes {
                    a
                } else {
                    self.wrong_capital(s, a, rng)
                };

                claim(
                    s,
                    a,
                    x,
                    (
                        CAP_Q,
                        "Yes, {a} is the capital of {s}.",
                        "No, the capital of {s} is {a}.",
                    ),
                    rng,
                )
            }
            "cont" => {
                let a = self.dv("continent", s);
                let x = if yes {
                    a
                } else {
                    other(&self.conts, |c| c != a, rng)
                };

                claim(
                    s,
                    a,
                    x,
                    (CONT_Q, "Yes, {s} is in {a}.", "No, {s} is in {a}."),
                    rng,
                )
            }
            _ => self.yn_lang(s, yes, rng),
        }
    }

    /// The language kind of `g_yesno`: a true claim names one of several
    /// languages, a false one any other language.
    #[expect(
        clippy::many_single_char_names,
        reason = "names follow gen2.py for side-by-side checking"
    )]
    fn yn_lang(&self, s: &str, yes: bool, rng: &mut PyRandom) -> (String, String) {
        let a = self.dv("language", s);
        let mine: Vec<String> = a
            .replace(", ", " and ")
            .split(" and ")
            .map(str::to_owned)
            .collect();
        let x = if yes {
            rng.choice(&mine).as_str()
        } else {
            let others: Vec<&String> = self.langs.iter().filter(|l| !mine.contains(l)).collect();
            rng.choice(&others).as_str()
        };

        let q = fmt(rng.choice::<&str>(LANG_Q), &[("x", x), ("s", s)]);
        let t = if mine.iter().any(|m| m == x) {
            "Yes, people in {s} speak {a}."
        } else {
            "No, people in {s} speak {a}."
        };

        (q, fmt(t, &[("a", a), ("s", s)]))
    }

    /// A wrong capital, from the same continent when it has more than one.
    fn wrong_capital(&self, s: &str, a: &str, rng: &mut PyRandom) -> &str {
        let same = self.d("continent").get(s).cloned();
        match self.caps_by_cont.get(&same) {
            Some(caps) if caps.len() > 1 => other::<String>(caps, |c| c != a, rng),
            _ => other::<String>(&self.rev.cap_rev_keys, |c| c != a, rng),
        }
    }

    /// The legs kind of `g_yesno`; legless and many-legged animals are
    /// always asked falsely.
    fn yn_legs(&self, yes: bool, rng: &mut PyRandom) -> (String, String) {
        let (s, a) = rng.choice(self.rel("legs"));
        if a == "none" || a == "many" {
            let none = a == "none";
            let q = format!(
                "does a {s} have {} legs",
                if none { "four" } else { "eight" }
            );
            let ans = if none {
                format!("No, a {s} has no legs.")
            } else {
                format!("No, a {s} has very many legs.")
            };

            return (q, ans);
        }

        let x = if yes {
            a.as_str()
        } else {
            *other(LEG_COUNTS, |c| c != a, rng)
        };

        claim(
            s,
            a,
            x,
            (
                LEGS_Q,
                "Yes, a {s} has {a} legs.",
                "No, a {s} has {a} legs.",
            ),
            rng,
        )
    }

    /// The plural and past kinds of `g_yesno`: false claims use the naive
    /// regular form.
    fn yn_form(&self, plural: bool, yes: bool, rng: &mut PyRandom) -> (String, String) {
        let (s, a) = rng.choice(self.rel(if plural { "plural" } else { "past" }));
        let naive = if plural {
            naive_plural(s)
        } else {
            format!("{s}{}", if s.ends_with('e') { "d" } else { "ed" })
        };

        let fallback = format!("{s}{}", if plural { "ies" } else { "s" });
        let x = if yes {
            a.clone()
        } else if naive == *a {
            fallback
        } else {
            naive
        };

        let c: Claim = if plural {
            (
                PLURAL_Q,
                "Yes, one {s}, two {a}.",
                "No, the plural of {s} is {a}.",
            )
        } else {
            (
                PAST_Q,
                "Yes, the past of {s} is {a}.",
                "No, the past of {s} is {a}.",
            )
        };

        claim(s, a, &x, c, rng)
    }

    /// The days kind of `g_yesno`; February has a fixed question.
    fn yn_days(&self, yes: bool, rng: &mut PyRandom) -> (String, String) {
        let (s, a) = rng.choice(self.rel("month_days"));
        if s == "February" {
            return (
                "does february have 30 days".to_owned(),
                "No, February has 28 days, or 29 in a leap year.".to_owned(),
            );
        }

        let x = if yes {
            a.as_str()
        } else if a == "31" {
            "30"
        } else {
            "31"
        };

        claim(
            s,
            a,
            x,
            (DAYS_Q, "Yes, {s} has {a} days.", "No, {s} has {a} days."),
            rng,
        )
    }
}

/// Asks whether `x` is the answer for `s` and answers with the true `a`.
fn claim(s: &str, a: &str, x: &str, (qs, yes, no): Claim, rng: &mut PyRandom) -> (String, String) {
    let args = [("s", s), ("a", a), ("x", x)];
    let q = fmt(rng.choice::<&str>(qs), &args);
    (q, fmt(if x == a { yes } else { no }, &args))
}
