//! The number generators of `gen2.py`: sums, counting, comparing, number
//! words (`g_numbers`), Roman numerals and squares (`g_numbers2`).

use super::tables::ROMAN;
use super::{Kit, cap, fmt, other, words};
use crate::pyrandom::PyRandom;

/// The kinds of number question, repeated by weight.
const KINDS: &[&str] = &[
    "add",
    "add",
    "add",
    "sub",
    "sub",
    "mul",
    "mul",
    "div",
    "next",
    "prev",
    "odd",
    "big",
    "small",
    "words",
    "fromwords",
    "count",
    "double",
    "half",
];

/// Operators of addition, repeated by weight.
const ADD_OPS: &[&str] = &["plus", "plus", "+", "add"];

/// Operators of subtraction.
const SUB_OPS: &[&str] = &["minus", "minus", "-", "take away"];

/// Operators of multiplication.
const MUL_OPS: &[&str] = &["times", "times", "x", "*", "multiplied by"];

/// Operators of division.
const DIV_OPS: &[&str] = &["divided by", "divided by", "/"];

/// Questions with the operator `add`.
const ADD_Q: &[&str] = &["add {A} and {B}", "what is {A} add {B}", "add {A} to {B}"];

/// Questions with a symbol operator and digits.
const SYMBOL_Q: &[&str] = &[
    "{A}{o}{B}",
    "{A} {o} {B}",
    "what is {A} {o} {B}",
    "what is {A}{o}{B}",
    "{A} {o} {B} =",
    "how much is {A} {o} {B}",
];

/// Questions with a word operator.
const WORD_Q: &[&str] = &[
    "what is {A} {o} {B}",
    "{A} {o} {B}",
    "how much is {A} {o} {B}",
    "can you tell me what {A} {o} {B} is",
    "what does {A} {o} {B} make",
];

/// Other phrasings of subtraction.
const SUB_Q: &[&str] = &[
    "what is {B} less than {A}",
    "take {B} from {A}",
    "subtract {B} from {A}",
];

/// Other phrasings of multiplication.
const MUL_Q: &[&str] = &["what is {A} lots of {B}", "{A} groups of {B}"];

/// Answers of a sum.
const SUM_A: &[&str] = &[
    "{r}.",
    "{a} {o} {b} is {r}.",
    "It is {r}.",
    "The answer is {r}.",
];

/// Numbers just below a round one, for `next`.
const NEXT_EDGES: &[i64] = &[9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 199, 299, 499, 999];

/// Round numbers, for `prev`.
const PREV_EDGES: &[i64] = &[10, 20, 30, 40, 50, 100, 200, 500, 999];

/// The next number.
const NEXT_Q: &[&str] = &[
    "what number comes after {n}",
    "what comes after {n}",
    "which number is after {n}",
    "what is one more than {n}",
    "what is {n} plus one",
    "what is the next number after {n}",
    "count on from {n}",
];

/// Answers of the next number.
const NEXT_A: &[&str] = &["{r}.", "{r} comes after {n}.", "It is {r}."];

/// The previous number.
const PREV_Q: &[&str] = &[
    "what number comes before {n}",
    "what comes before {n}",
    "which number is before {n}",
    "what is one less than {n}",
    "what is {n} minus one",
];

/// Answers of the previous number.
const PREV_A: &[&str] = &["{r}.", "{r} comes before {n}.", "It is {r}."];

/// Even or odd.
const ODD_Q: &[&str] = &[
    "is {n} even or odd",
    "is {n} odd or even",
    "is {n} an even number",
    "is {n} odd",
    "is {n} even",
    "even or odd: {n}",
];

/// The bigger of two numbers.
const BIG_Q: &[&str] = &[
    "which is bigger, {a} or {b}",
    "which is more, {a} or {b}",
    "is {a} bigger than {b}",
    "which number is larger, {a} or {b}",
    "{a} or {b}, which is bigger",
];

/// The smaller of two numbers.
const SMALL_Q: &[&str] = &[
    "which is smaller, {a} or {b}",
    "which is less, {a} or {b}",
    "is {a} smaller than {b}",
    "which number is smaller, {a} or {b}",
];

/// A number in words.
const WORDS_Q: &[&str] = &[
    "how do you write {n} in words",
    "write {n} in words",
    "how do you say {n}",
    "spell the number {n}",
    "{n} in words",
];

/// Words as a number.
const FROMWORDS_Q: &[&str] = &[
    "what number is {w}",
    "write {w} in numbers",
    "how do you write {w} as a number",
    "{w} as a number",
];

/// Counting up.
const COUNT_Q: &[&str] = &[
    "count from {a} to {b}",
    "count up from {a} to {b}",
    "what are the numbers from {a} to {b}",
];

/// Doubling.
const DOUBLE_Q: &[&str] = &[
    "what is double {n}",
    "double {n}",
    "what is twice {n}",
    "what is {n} doubled",
];

/// Answers of doubling.
const DOUBLE_A: &[&str] = &["{r}.", "Double {n} is {r}."];

/// Halving.
const HALF_Q: &[&str] = &[
    "what is half of {n}",
    "half of {n}",
    "what is {n} halved",
    "what is half {n}",
];

/// Answers of halving.
const HALF_A: &[&str] = &["{r}.", "Half of {n} is {r}."];

/// The kinds of `g_numbers2`, repeated by weight.
const KINDS2: &[&str] = &["roman", "roman", "unroman", "square"];

/// A number in Roman numerals.
const ROMAN_Q: &[&str] = &[
    "what is {n} in roman numerals",
    "write {n} in roman numerals",
    "{n} in roman numerals",
    "how do you write {n} in roman numerals",
    "roman numeral for {n}",
];

/// Answers in Roman numerals.
const ROMAN_A: &[&str] = &["{r}.", "{n} is {r} in Roman numerals."];

/// Roman numerals as a number.
const UNROMAN_Q: &[&str] = &[
    "what number is {r}",
    "what is the roman numeral {r}",
    "what does {r} mean in roman numerals",
    "{r} in numbers",
];

/// Answers of Roman numerals as a number.
const UNROMAN_A: &[&str] = &["{n}.", "{r} is {n}."];

/// Squares and square roots.
const SQUARE_Q: &[&str] = &[
    "what is {n} squared",
    "{n} squared",
    "what is {n} times itself",
    "what is the square of {n}",
    "what is the square root of {m}",
];

/// Answers of a square root.
const ROOT_A: &[&str] = &["{n}.", "The square root of {m} is {n}."];

/// Answers of a square.
const SQUARE_A: &[&str] = &["{m}.", "{n} squared is {m}."];

/// `num`: a number in words or digits.
fn num(n: i64, wordy: bool) -> String {
    if wordy { words(n) } else { n.to_string() }
}

/// `roman`: Roman numerals of a positive number.
fn roman(mut n: i64) -> String {
    let mut out = String::new();
    for &(v, r) in ROMAN {
        while n >= i64::from(v) {
            out.push_str(r);
            n -= i64::from(v);
        }
    }

    out
}

/// Picks one of the train or test halves of a split.
fn half<T>(split: &(Vec<T>, Vec<T>), test: bool) -> &[T] {
    if test { &split.1 } else { &split.0 }
}

impl Kit {
    /// `g_numbers`.
    pub(super) fn g_numbers(&self, rng: &mut PyRandom, test: bool) -> (String, String) {
        let k = *rng.choice(KINDS);
        let wordy = rng.random() < 0.25;
        match k {
            "add" | "sub" | "mul" | "div" => self.sum(k, wordy, rng, test),
            "next" => {
                let n = if rng.random() < 0.7 {
                    rng.randint(0, 998)
                } else {
                    *rng.choice(NEXT_EDGES)
                };

                let n = n.min(998);
                step(n, n + 1, wordy, (NEXT_Q, NEXT_A), rng)
            }
            "prev" => {
                let n = if rng.random() < 0.7 {
                    rng.randint(1, 999)
                } else {
                    *rng.choice(PREV_EDGES)
                };

                step(n, n - 1, wordy, (PREV_Q, PREV_A), rng)
            }
            "odd" => {
                let n = rng.randint(0, 999);
                let q = fmt(
                    rng.choice::<&str>(ODD_Q),
                    &[("n", &num(n, wordy && n < 100))],
                );
                (
                    q,
                    format!("{n} is {}.", if n % 2 == 0 { "even" } else { "odd" }),
                )
            }
            "big" | "small" => compare(k == "big", rng),
            "words" | "fromwords" => number_words(k == "words", rng),
            "count" => count(rng),
            _ => scale(k == "double", wordy, rng),
        }
    }

    /// The four sums of `g_numbers`, from the held-out operand pairs.
    #[expect(
        clippy::many_single_char_names,
        reason = "names follow gen2.py for side-by-side checking"
    )]
    fn sum(&self, k: &str, wordy: bool, rng: &mut PyRandom, test: bool) -> (String, String) {
        let (a, b, r, ops, sym) = match k {
            "add" => {
                let &(a, b) = rng.choice(half(&self.split.add, test));
                (a, b, a + b, ADD_OPS, "plus")
            }
            "sub" => {
                let &(a, b) = rng.choice(half(&self.split.sub, test));
                (a, b, a - b, SUB_OPS, "minus")
            }
            "mul" => {
                let &(a, b) = rng.choice(half(&self.split.mul, test));
                (a, b, a * b, MUL_OPS, "times")
            }
            _ => {
                let &(b, r) = rng.choice(half(&self.split.mul, test));
                (b * r, b, r, DIV_OPS, "divided by")
            }
        };

        let op = *rng.choice(ops);
        let (x, y) = (num(a, wordy), num(b, wordy));
        let symbol = "+-x*/".contains(op);
        let mut q = if op == "add" {
            fmt(rng.choice::<&str>(ADD_Q), &[("A", &x), ("B", &y)])
        } else if symbol && !wordy {
            fmt(
                rng.choice::<&str>(SYMBOL_Q),
                &[("A", &x), ("B", &y), ("o", op)],
            )
        } else {
            let o = if symbol { sym } else { op };
            fmt(
                rng.choice::<&str>(WORD_Q),
                &[("A", &x), ("B", &y), ("o", o)],
            )
        };

        if k == "sub" && rng.random() < 0.2 {
            q = fmt(rng.choice::<&str>(SUB_Q), &[("A", &x), ("B", &y)]);
        }

        if k == "mul" && rng.random() < 0.15 {
            q = fmt(rng.choice::<&str>(MUL_Q), &[("A", &x), ("B", &y)]);
        }

        let args = [
            ("r", r.to_string()),
            ("a", a.to_string()),
            ("b", b.to_string()),
        ];
        let args = [
            ("r", args[0].1.as_str()),
            ("a", &args[1].1),
            ("b", &args[2].1),
            ("o", sym),
        ];
        (q, fmt(rng.choice::<&str>(SUM_A), &args))
    }

    /// `g_numbers2`: Roman numerals both ways, squares and square roots.
    pub(super) fn g_numbers2(&self, rng: &mut PyRandom, test: bool) -> (String, String) {
        let k = *rng.choice(KINDS2);
        if k == "square" {
            let n = *rng.choice(half(&self.split.sq, test));
            let (ns, ms) = (n.to_string(), (n * n).to_string());
            let args = [("n", ns.as_str()), ("m", &ms)];
            let q = fmt(rng.choice::<&str>(SQUARE_Q), &args);
            let ans = if q.contains("root") { ROOT_A } else { SQUARE_A };
            return (q, fmt(rng.choice::<&str>(ans), &args));
        }

        let n = *rng.choice(half(&self.split.roman, test));
        let (ns, r) = (n.to_string(), roman(n));
        let args = [("n", ns.as_str()), ("r", &r)];
        let (qs, ans) = if k == "roman" {
            (ROMAN_Q, ROMAN_A)
        } else {
            (UNROMAN_Q, UNROMAN_A)
        };

        let q = fmt(rng.choice::<&str>(qs), &args);
        (q, fmt(rng.choice::<&str>(ans), &args))
    }
}

/// `next` and `prev`: the neighbour `r` of `n`; only numbers below 100 are
/// asked in words.
fn step(
    n: i64,
    r: i64,
    wordy: bool,
    (qs, ans): (&[&str], &[&str]),
    rng: &mut PyRandom,
) -> (String, String) {
    let q = fmt(rng.choice::<&str>(qs), &[("n", &num(n, wordy && n < 100))]);
    (
        q,
        fmt(
            rng.choice::<&str>(ans),
            &[("r", &r.to_string()), ("n", &n.to_string())],
        ),
    )
}

/// `big` and `small`: the second number is near the first half the time.
fn compare(big: bool, rng: &mut PyRandom) -> (String, String) {
    let a = rng.randint(0, 999);
    let pool: Vec<i64> = if rng.random() < 0.5 {
        ((a - 30).max(0)..=(a + 30).min(999)).collect()
    } else {
        (0..1000).collect()
    };

    let b = *other(&pool, |&x| x != a, rng);
    let args = [("a", a.to_string()), ("b", b.to_string())];
    let args = [("a", args[0].1.as_str()), ("b", &args[1].1)];
    let q = fmt(rng.choice::<&str>(if big { BIG_Q } else { SMALL_Q }), &args);
    let ans = match (q.starts_with("is "), big) {
        (false, true) => format!("{} is bigger.", a.max(b)),
        (false, false) => format!("{} is smaller.", a.min(b)),
        (true, true) if a > b => format!("Yes, {a} is bigger than {b}."),
        (true, true) => format!("No, {a} is smaller than {b}."),
        (true, false) if a < b => format!("Yes, {a} is smaller than {b}."),
        (true, false) => format!("No, {a} is bigger than {b}."),
    };

    (q, ans)
}

/// `count`: up from `a` to `b`, or down when the second draw says so.
fn count(rng: &mut PyRandom) -> (String, String) {
    let a = rng.randint(0, 95);
    let b = a + rng.randint(2, 9);
    let q = fmt(
        rng.choice::<&str>(COUNT_Q),
        &[("a", &a.to_string()), ("b", &b.to_string())],
    );
    let join = |xs: Vec<i64>| -> String {
        let s: Vec<String> = xs.iter().map(i64::to_string).collect();
        format!("{}.", s.join(", "))
    };

    if rng.random() < 0.3 {
        return (
            format!("count down from {b} to {a}"),
            join((a..=b).rev().collect()),
        );
    }

    (q, join((a..=b).collect()))
}

/// `words` and `fromwords`: a number to words or back.
fn number_words(to_words: bool, rng: &mut PyRandom) -> (String, String) {
    if to_words {
        let n = if rng.random() < 0.4 {
            rng.randint(0, 999)
        } else {
            rng.randint(0, 100)
        };

        let q = fmt(rng.choice::<&str>(WORDS_Q), &[("n", &n.to_string())]);
        return (q, format!("{}.", cap(&words(n))));
    }

    let n = rng.randint(0, 100);
    (
        fmt(rng.choice::<&str>(FROMWORDS_Q), &[("w", &words(n))]),
        format!("{n}."),
    )
}

/// `double` and `half` of a number up to 50 (half: of an even one up to 100).
fn scale(double: bool, wordy: bool, rng: &mut PyRandom) -> (String, String) {
    let (n, r, qs, ans) = if double {
        let n = rng.randint(0, 50);
        (n, 2 * n, DOUBLE_Q, DOUBLE_A)
    } else {
        let n = 2 * rng.randint(0, 50);
        (n, n / 2, HALF_Q, HALF_A)
    };

    let q = fmt(rng.choice::<&str>(qs), &[("n", &num(n, wordy))]);
    let a = fmt(
        rng.choice::<&str>(ans),
        &[("r", &r.to_string()), ("n", &n.to_string())],
    );
    (q, a)
}
