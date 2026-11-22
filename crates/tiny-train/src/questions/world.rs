//! The fact generators of `gen2.py`: one-off facts, table relations,
//! landlocked countries, borders and two-part questions.

use super::tables::{GENERIC, GENERIC_Q, SWAPS};
use super::{Kit, Relation, art, cap, fmt, list_and, other, pick_q, words};
use crate::pyrandom::PyRandom;

/// `is it landlocked` phrasings.
const LANDLOCKED_Q: &[&str] = &[
    "is {s} landlocked",
    "is {s} a landlocked country",
    "is {s} cut off from the sea",
    "does {s} have no coast",
    "is {s} surrounded by land",
];

/// `does it have a coast` phrasings.
const COAST_Q: &[&str] = &[
    "does {s} have a coast",
    "does {s} have a coastline",
    "is {s} by the sea",
    "does {s} touch the sea",
    "can you see the sea in {s}",
    "does {s} have a sea coast",
];

/// Questions for the list of neighbours.
const BORDERS_Q: &[&str] = &[
    "which countries border {s}",
    "what are the neighbours of {s}",
    "which countries are next to {s}",
    "what countries does {s} border",
    "name the neighbours of {s}",
    "neighbours of {s}",
    "what countries are around {s}",
    "what are the neighbors of {s}",
];

/// Answers for an island without land neighbours.
const NO_BORDERS_A: &[&str] = &[
    "None. {s} has no land borders.",
    "{s} has no land neighbours.",
];

/// Questions for the number of neighbours.
const BORDER_COUNT_Q: &[&str] = &[
    "how many countries border {s}",
    "how many neighbours does {s} have",
    "how many countries are next to {s}",
];

/// Yes/no questions whether two countries share a border.
const BORDER_YN_Q: &[&str] = &[
    "does {s} border {t}",
    "is {s} next to {t}",
    "are {s} and {t} neighbours",
    "does {s} share a border with {t}",
    "is {t} a neighbour of {s}",
];

/// The kinds of two-part question, repeated by weight.
const COMBO_KINDS: &[&str] = &[
    "capcur", "capcont", "caplang", "twocap", "twocap", "twocont", "animal2", "animal3",
];

/// Capital and continent: the fallback two-part question.
const CAPCONT_Q: &[&str] = &[
    "where is {s} and what is its capital",
    "which continent is {s} in and what is its capital",
    "tell me the continent and capital of {s}",
];

/// Capital and currency.
const CAPCUR_Q: &[&str] = &[
    "what is the capital and the currency of {s}",
    "tell me the capital and money of {s}",
    "what are the capital and currency of {s}",
    "capital and currency of {s}",
];

/// Capital and language.
const CAPLANG_Q: &[&str] = &[
    "what is the capital of {s} and what language do they speak",
    "capital and language of {s}",
    "tell me the capital and language of {s}",
];

/// Capitals of two countries.
const TWOCAP_Q: &[&str] = &[
    "what are the capitals of {s} and {t}",
    "capitals of {s} and {t}",
    "what is the capital of {s} and of {t}",
    "tell me the capitals of {s} and {t}",
];

/// Answers for the capitals of two countries.
const TWOCAP_A: &[&str] = &[
    "{a} and {b}.",
    "The capital of {s} is {a} and of {t} it is {b}.",
];

/// Continents of two countries.
const TWOCONT_Q: &[&str] = &[
    "which continents are {s} and {t} in",
    "where are {s} and {t}",
    "on which continents are {s} and {t}",
];

/// Legs and sound of an animal.
const ANIMAL2_Q: &[&str] = &[
    "how many legs does {ax} have and what does it say",
    "what does {ax} say and how many legs does it have",
];

/// Class and baby of an animal.
const ANIMAL3_Q: &[&str] = &[
    "what kind of animal is {ax} and what is its baby called",
    "what is a baby {x} called and what kind of animal is it",
];

/// `reword`: one or two automatic rewordings from [`SWAPS`].
fn reword(q: &str, rng: &mut PyRandom) -> String {
    let mut q = q.to_owned();
    for _ in 0..rng.randint(1, 2) {
        let opts: Vec<&(&str, &str)> = SWAPS.iter().filter(|(a, _)| q.contains(a)).collect();
        if opts.is_empty() {
            break;
        }

        let (a, b) = **rng.choice(&opts);
        q = q.replacen(a, b, 1);
    }

    q
}

impl Kit {
    /// `g_txt`: a one-off fact, sometimes reworded when training.
    pub(super) fn g_txt(&self, rng: &mut PyRandom, test: bool) -> (String, String) {
        loop {
            let f = rng.choice(&self.txt);
            let qs = if test { &f.test_q } else { &f.train_q };
            if qs.is_empty() {
                continue;
            }

            let mut q = rng.choice(qs).clone();
            if !test && rng.random() < 0.35 {
                q = reword(&q, rng);
            }

            return (q, rng.choice(&f.answers).clone());
        }
    }

    /// `g_rel`: a weighted table relation and one of its pairs.
    pub(super) fn g_rel(&self, rng: &mut PyRandom, test: bool) -> (String, String) {
        let r = &self.rels[rng.choices_index(&self.rel_weights)];
        let (s, a) = rng.choice(self.rel(&r.name));
        self.rel_qa(r, s, a, rng, test)
    }

    /// `rel_qa`: question and answer for one pair of a relation.
    #[expect(
        clippy::many_single_char_names,
        reason = "names follow gen2.py for side-by-side checking"
    )]
    fn rel_qa(
        &self,
        r: &Relation,
        s: &str,
        a: &str,
        rng: &mut PyRandom,
        test: bool,
    ) -> (String, String) {
        let generic = GENERIC.iter().find(|(n, _)| *n == r.name).map(|(_, l)| *l);
        let q = match generic {
            Some(l) if !test && rng.random() < 0.2 => {
                let x = art(s);
                fmt(rng.choice(GENERIC_Q), &[("l", l), ("x", &x)])
            }
            _ => self.fill(pick_q(&r.questions, rng, test), s, a),
        };

        let t = rng.randbelow(r.answers.len());
        let special: Option<[&str; 3]> = match (r.name.as_str(), a, s) {
            ("legs", "none", _) => Some(["None.", "A {s} has no legs.", "It has no legs."]),
            ("legs", "many", _) => Some([
                "Very many.",
                "A {s} has very many legs.",
                "It has very many legs.",
            ]),
            ("month_days", _, "February") => Some([
                "28, or 29 in a leap year.",
                "February has 28 days, or 29 in a leap year.",
                "It has 28 days, or 29 in a leap year.",
            ]),
            _ => None,
        };

        let a2 = match special {
            Some(alt) => fmt(alt[t], &[("s", s)]),
            None => self.fill(&r.answers[t], s, a),
        };

        (q, a2)
    }

    /// `g_landlocked`: yes/no about a coast, asked both ways round.
    pub(super) fn g_landlocked(&self, rng: &mut PyRandom, test: bool) -> (String, String) {
        let (s, a) = rng.choice(self.rel("landlocked"));
        let yes = a == "yes";
        let args = [("s", s.as_str())];
        if rng.random() < 0.5 {
            let q = fmt(pick_q(LANDLOCKED_Q, rng, test), &args);
            let t = if yes {
                "Yes, {s} is landlocked. It has no coast."
            } else {
                "No, {s} has a coast."
            };

            return (q, fmt(t, &args));
        }

        let q = fmt(pick_q(COAST_Q, rng, test), &args);
        let t = if yes {
            "No, {s} is landlocked."
        } else {
            "Yes, {s} has a coast."
        };

        (q, fmt(t, &args))
    }

    /// The yes/no country pool: `YN_COUNTRIES_TEST` or `_TRAIN`.
    pub(super) fn yn_pool(&self, test: bool) -> &[String] {
        if test {
            &self.split.yn_countries.1
        } else {
            &self.split.yn_countries.0
        }
    }

    /// `g_borders`: neighbours, their number, or whether two countries touch.
    #[expect(
        clippy::many_single_char_names,
        reason = "names follow gen2.py for side-by-side checking"
    )]
    pub(super) fn g_borders(&self, rng: &mut PyRandom, test: bool) -> (String, String) {
        let pool: Vec<&String> = self
            .yn_pool(test)
            .iter()
            .filter(|c| self.borders.contains_key(*c))
            .collect();
        let s: &str = rng.choice(&pool);
        let nb = &self.borders[s];
        let k = rng.random();
        let args = [("s", s)];
        if k < 0.45 {
            let q = fmt(rng.choice(BORDERS_Q), &args);
            if nb.is_empty() {
                return (q, fmt(rng.choice(NO_BORDERS_A), &args));
            }

            if nb.len() <= 4 {
                return (q, format!("{}.", list_and(nb)));
            }

            return (
                q,
                format!("{} and {} others.", nb[..3].join(", "), words(len(nb) - 3)),
            );
        }

        if k < 0.6 {
            let q = fmt(rng.choice(BORDER_COUNT_Q), &args);
            let n = nb.len();
            let t = if n == 0 {
                "{s} has no land neighbours."
            } else {
                let many = if n > 1 {
                    "{s} borders {n} countries."
                } else {
                    "{s} borders one country."
                };

                *rng.choice(&["{N}.", many])
            };

            let w = words(len(nb));
            return (q, fmt(t, &[("s", s), ("n", &w), ("N", &cap(&w))]));
        }

        self.border_yn(s, nb, rng)
    }

    /// The yes/no part of `g_borders`: a neighbour, or a country of the same
    /// continent that is not one.
    fn border_yn(&self, s: &str, nb: &[String], rng: &mut PyRandom) -> (String, String) {
        let yes = !nb.is_empty() && rng.random() < 0.5;
        let t: &str = if yes {
            rng.choice(nb).as_str()
        } else {
            let cont = self.d("continent");
            let same: Vec<&String> = self
                .countries
                .iter()
                .filter(|c| cont.get(*c) == cont.get(s) && *c != s && !nb.contains(*c))
                .collect();
            if same.is_empty() {
                rng.choice(&self.countries).as_str()
            } else {
                rng.choice(&same).as_str()
            }
        };

        let args = [("s", s), ("t", t)];
        let q = fmt(rng.choice(BORDER_YN_Q), &args);
        let a = if nb.iter().any(|c| c == t) {
            "Yes, {s} borders {t}."
        } else {
            "No, {s} does not border {t}."
        };

        (q, fmt(a, &args))
    }

    /// `g_combo`: two facts in one question.
    pub(super) fn g_combo(&self, rng: &mut PyRandom, test: bool) -> (String, String) {
        let pool = self.yn_pool(test);
        let k = *rng.choice(COMBO_KINDS);
        let s = rng.choice(pool).as_str();
        let args = [("s", s)];
        let cap_of = |c: &str| self.dv("capital", c);
        if k == "capcur" && self.d("currency").contains_key(s) {
            let q = fmt(rng.choice(CAPCUR_Q), &args);
            let a = format!(
                "The capital is {} and the money is the {}.",
                cap_of(s),
                self.dv("currency", s)
            );
            return (q, a);
        }

        if k == "caplang" && self.d("language").contains_key(s) {
            let q = fmt(rng.choice(CAPLANG_Q), &args);
            return (
                q,
                format!(
                    "The capital is {} and people speak {}.",
                    cap_of(s),
                    self.dv("language", s)
                ),
            );
        }

        match k {
            "twocap" => {
                let t = other(pool, |c| c != s, rng).as_str();
                let args = [("s", s), ("t", t), ("a", cap_of(s)), ("b", cap_of(t))];
                let q = fmt(rng.choice(TWOCAP_Q), &args);
                (q, fmt(rng.choice(TWOCAP_A), &args))
            }
            "twocont" => {
                let t = other(pool, |c| c != s, rng).as_str();
                let q = fmt(rng.choice(TWOCONT_Q), &[("s", s), ("t", t)]);
                let cont = |c: &str| self.dv("continent", c);
                (
                    q,
                    format!("{s} is in {} and {t} is in {}.", cont(s), cont(t)),
                )
            }
            "animal2" | "animal3" => self.combo_animal(k == "animal2", rng),
            _ => {
                let q = fmt(rng.choice(CAPCONT_Q), &args);
                let a = format!(
                    "{s} is in {}. The capital is {}.",
                    self.dv("continent", s),
                    cap_of(s)
                );
                (q, a)
            }
        }
    }

    /// The animal parts of `g_combo`: legs and sound, or class and baby.
    fn combo_animal(&self, legs: bool, rng: &mut PyRandom) -> (String, String) {
        let both = if legs {
            &self.legs_sound
        } else {
            &self.baby_class
        };

        let x = rng.choice(both).as_str();
        let ax = art(x);
        let q = fmt(
            rng.choice(if legs { ANIMAL2_Q } else { ANIMAL3_Q }),
            &[("x", x), ("ax", &ax)],
        );
        let a = if legs {
            format!(
                "{} has {} legs and says {}.",
                cap(&ax),
                self.dv("legs", x),
                self.dv("sound", x)
            )
        } else {
            let c = art(self.dv("class", x));
            format!(
                "{} is {c} and its baby is {}.",
                cap(&ax),
                art(self.dv("baby", x))
            )
        };

        (q, a)
    }
}

/// A list length as the `i64` that [`words`] takes.
fn len<T>(xs: &[T]) -> i64 {
    i64::try_from(xs.len()).expect("short list")
}
