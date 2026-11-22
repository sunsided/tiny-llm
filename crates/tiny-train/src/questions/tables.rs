//! The phrasing tables of `gen2.py`: question and answer templates per table
//! relation, plus the word lists the noise and "unknown" generators draw from.
//!
//! Templates use Python `str.format` fields: `{s}` subject, `{S}` subject
//! capitalised, `{a}` answer, `{A}` answer capitalised, `{art}`/`{Art}` answer
//! with its article, `{p}` plural of the subject. The last question of every
//! list is kept back for the test set.

/// Question and answer templates per relation, in `REL_T`'s dict order
/// (without the `word_*` relations, which [`LANG_Q`] and [`LANG_A`] make).
pub(super) const REL_T: &[(&str, &[&str], &[&str])] = &[
    (
        "capital",
        &[
            "what is the capital of {s}",
            "what is the capital city of {s}",
            "capital of {s}",
            "what is {s}'s capital",
            "which city is the capital of {s}",
            "name the capital of {s}",
            "tell me the capital of {s}",
            "{s} capital",
            "do you know the capital of {s}",
            "the capital of {s} is",
            "what city is the capital of {s}",
            "capital city of {s}",
            "where is the government of {s}",
            "what is the main city of {s}",
            "which city is {s}'s capital",
        ],
        &[
            "{a}.",
            "It is {a}.",
            "The capital is {a}.",
            "The capital of {s} is {a}.",
        ],
    ),
    (
        "continent",
        &[
            "which continent is {s} in",
            "what continent is {s} in",
            "on which continent is {s}",
            "where is {s}",
            "in which part of the world is {s}",
            "{s} is in which continent",
            "continent of {s}",
            "which continent does {s} belong to",
            "is {s} in europe or asia",
            "where in the world is {s}",
            "what continent is {s} on",
        ],
        &["{a}.", "In {a}.", "It is in {a}.", "{s} is in {a}."],
    ),
    (
        "language",
        &[
            "what language is spoken in {s}",
            "what language do people speak in {s}",
            "which language do they speak in {s}",
            "what is the language of {s}",
            "what do people speak in {s}",
            "language of {s}",
            "what language do they use in {s}",
            "what language do people in {s} speak",
            "which language is used in {s}",
            "what is the main language of {s}",
        ],
        &[
            "{a}.",
            "They speak {a}.",
            "People there speak {a}.",
            "In {s} people speak {a}.",
        ],
    ),
    (
        "currency",
        &[
            "what is the currency of {s}",
            "what money do people use in {s}",
            "what money is used in {s}",
            "which currency is used in {s}",
            "currency of {s}",
            "what do people pay with in {s}",
            "what is the money of {s} called",
            "money of {s}",
            "what money do they have in {s}",
            "what do they pay with in {s}",
            "what is the money in {s}",
        ],
        &[
            "The {a}.",
            "It is the {a}.",
            "They use the {a}.",
            "The money in {s} is the {a}.",
        ],
    ),
    (
        "people",
        &[
            "what are people from {s} called",
            "what do you call people from {s}",
            "what nationality are people from {s}",
            "what is someone from {s} called",
            "people from {s} are called what",
            "what is the nationality of {s}",
            "what do we call a person from {s}",
            "what nationality is someone from {s}",
        ],
        &["{a}.", "They are called {a}.", "The nationality is {a}."],
    ),
    (
        "month_after",
        &[
            "which month comes after {s}",
            "what month comes after {s}",
            "what is the month after {s}",
            "what month follows {s}",
            "month after {s}",
            "after {s} comes which month",
            "what is the next month after {s}",
            "which month is after {s}",
            "which month is next after {s}",
        ],
        &["{a}.", "It is {a}.", "{a} comes after {s}."],
    ),
    (
        "month_before",
        &[
            "which month comes before {s}",
            "what month comes before {s}",
            "what is the month before {s}",
            "month before {s}",
            "before {s} comes which month",
            "which month is before {s}",
            "what month is just before {s}",
            "which month is just before {s}",
        ],
        &["{a}.", "It is {a}.", "{a} comes before {s}."],
    ),
    (
        "month_days",
        &[
            "how many days are in {s}",
            "how many days does {s} have",
            "how many days has {s}",
            "how long is {s} in days",
            "days in {s}",
            "number of days in {s}",
            "{s} has how many days",
            "how many days are there in {s}",
        ],
        &["{A}.", "{s} has {a} days.", "It has {a} days."],
    ),
    (
        "month_number",
        &[
            "what is the {s} month of the year",
            "which month is the {s} month",
            "what is the {s} month",
            "name the {s} month of the year",
            "the {s} month is called what",
            "what month is {s}",
            "which is the {s} month of the year",
        ],
        &["{a}.", "It is {a}.", "The {s} month is {a}."],
    ),
    (
        "day_after",
        &[
            "which day comes after {s}",
            "what day comes after {s}",
            "what is the day after {s}",
            "what day follows {s}",
            "day after {s}",
            "if today is {s} what is tomorrow",
            "what day is after {s}",
            "today is {s}, what day is tomorrow",
            "which day is next after {s}",
        ],
        &["{a}.", "It is {a}.", "{a} comes after {s}."],
    ),
    (
        "day_before",
        &[
            "which day comes before {s}",
            "what day comes before {s}",
            "what is the day before {s}",
            "day before {s}",
            "if today is {s} what was yesterday",
            "what day is before {s}",
            "today is {s}, what was yesterday",
            "which day is just before {s}",
        ],
        &["{a}.", "It is {a}.", "{a} comes before {s}."],
    ),
    (
        "opposite",
        &[
            "what is the opposite of {s}",
            "opposite of {s}",
            "what is the opposite word for {s}",
            "tell me the opposite of {s}",
            "what word means the opposite of {s}",
            "the opposite of {s} is what",
            "what's the opposite of {s}",
            "antonym of {s}",
            "what is the antonym of {s}",
            "give me the opposite of {s}",
        ],
        &["{A}.", "It is {a}.", "The opposite of {s} is {a}."],
    ),
    (
        "plural",
        &[
            "what is the plural of {s}",
            "plural of {s}",
            "what do you call more than one {s}",
            "how do you say many of {s}",
            "one {s} and two what",
            "what is the plural form of {s}",
            "one {s}, two",
            "what's the plural of {s}",
            "give me the plural of {s}",
        ],
        &["{A}.", "It is {a}.", "One {s}, two {a}."],
    ),
    (
        "past",
        &[
            "what is the past tense of {s}",
            "past tense of {s}",
            "what is the past of {s}",
            "how do you say {s} in the past",
            "today i {s} and yesterday i what",
            "what is the past form of {s}",
            "what's the past tense of {s}",
            "yesterday i ... ({s})",
            "give me the past tense of {s}",
        ],
        &["{A}.", "It is {a}.", "The past of {s} is {a}."],
    ),
    (
        "class",
        &[
            "what kind of animal is a {s}",
            "what type of animal is a {s}",
            "what group does a {s} belong to",
            "what is the animal group of a {s}",
            "animal group of a {s}",
            "is a {s} a mammal or what",
            "what is a {s}",
            "a {s} is what kind of animal",
            "what animal group is a {s} in",
        ],
        &["A {a}.", "It is a {a}.", "A {s} is a {a}."],
    ),
    (
        "legs",
        &[
            "how many legs does a {s} have",
            "how many legs has a {s}",
            "a {s} has how many legs",
            "number of legs of a {s}",
            "what is the number of legs of a {s}",
            "legs of a {s}",
            "how many legs do {s}s have",
            "count the legs of a {s}",
            "how many feet does a {s} have",
            "how many legs are on a {s}",
        ],
        &["{A}.", "A {s} has {a} legs.", "It has {a} legs."],
    ),
    (
        "baby",
        &[
            "what is a baby {s} called",
            "what do you call a baby {s}",
            "what is the baby of a {s} called",
            "a baby {s} is called what",
            "what is the baby name of a {s}",
            "baby name of a {s}",
            "what is a young {s} called",
            "baby {s}",
            "what is the name for a baby {s}",
            "what do we call a baby {s}",
        ],
        &[
            "{Art}.",
            "It is called {art}.",
            "A baby {s} is called {art}.",
        ],
    ),
    (
        "food",
        &[
            "what does a {s} eat",
            "what food does a {s} eat",
            "what is the food of a {s}",
            "food of a {s}",
            "what do {s}s eat",
            "what does a {s} like to eat",
            "what is a {s}'s food",
            "tell me the food of a {s}",
            "what does a {s} feed on",
        ],
        &["{A}.", "It eats {a}.", "A {s} eats {a}."],
    ),
    (
        "home",
        &[
            "where does a {s} live",
            "what is the home of a {s}",
            "where do {s}s live",
            "home of a {s}",
            "where can you find a {s}",
            "where is the home of a {s}",
            "tell me the home of a {s}",
            "where does a {s} make its home",
            "where would you find a {s}",
        ],
        &["{A}.", "In {a}.", "A {s} lives in {a}."],
    ),
    (
        "sound",
        &[
            "what sound does a {s} make",
            "what does a {s} say",
            "what noise does a {s} make",
            "how does a {s} sound",
            "sound of a {s}",
            "what sound do {s}s make",
            "a {s} says what",
            "what is the sound of a {s}",
            "what noise do {s}s make",
        ],
        &["{A}!", "A {s} says {a}.", "It says {a}."],
    ),
    (
        "subregion",
        &[
            "what region is {s} in",
            "in which region is {s}",
            "which region of the world is {s} in",
            "region of {s}",
            "what is the region of {s}",
            "which world region is {s} part of",
            "where exactly is {s}",
        ],
        &["{a}.", "It is in {a}.", "{s} is in {a}."],
    ),
    (
        "domain",
        &[
            "what is the internet domain of {s}",
            "what domain does {s} use",
            "internet code of {s}",
            "what is the web domain of {s}",
            "what do websites in {s} end with",
            "domain of {s}",
            "what is the country code domain of {s}",
        ],
        &["{a}.", "It is {a}.", "Websites in {s} end with {a}."],
    ),
    (
        "symbol",
        &[
            "what is the symbol for {s}",
            "what is the chemical symbol of {s}",
            "symbol of {s}",
            "how do you write {s} in chemistry",
            "what is the element symbol for {s}",
            "what letters stand for {s}",
            "chemical symbol for {s}",
        ],
        &["{a}.", "It is {a}.", "The symbol for {s} is {a}."],
    ),
    (
        "atomic",
        &[
            "what is the atomic number of {s}",
            "atomic number of {s}",
            "how many protons does {s} have",
            "what number element is {s}",
            "which number is {s} in the periodic table",
            "what is the element number of {s}",
        ],
        &["{a}.", "It is {a}.", "{S} has atomic number {a}."],
    ),
    (
        "kind",
        &[
            "what kind of element is {s}",
            "what type of element is {s}",
            "is {s} a metal or a nonmetal",
            "what sort of element is {s}",
            "which kind of element is {s}",
        ],
        &["{A}.", "{S} is {a}.", "It is {a}."],
    ),
    (
        "phase",
        &[
            "is {s} a solid, liquid or gas",
            "what state is {s} at room temperature",
            "is {s} solid liquid or gas",
            "what is {s} like at room temperature",
            "is {s} a solid or a gas",
            "what state of matter is {s}",
        ],
        &["{A}.", "It is a {a}.", "{S} is a {a} at room temperature."],
    ),
    (
        "synonym",
        &[
            "what is another word for {s}",
            "what is a synonym for {s}",
            "another word for {s}",
            "synonym of {s}",
            "what word means the same as {s}",
            "what means the same as {s}",
            "give me another word for {s}",
        ],
        &["{A}.", "It is {a}.", "Another word for {s} is {a}."],
    ),
    (
        "comparative",
        &[
            "what is the comparative of {s}",
            "comparative of {s}",
            "how do you say more {s}",
            "what is the comparative form of {s}",
            "{s}, then what",
        ],
        &["{A}.", "It is {a}.", "The comparative of {s} is {a}."],
    ),
    (
        "superlative",
        &[
            "what is the superlative of {s}",
            "superlative of {s}",
            "how do you say the most {s}",
            "what is the superlative form of {s}",
            "{s}, ..., what is the last one",
        ],
        &["{A}.", "It is {a}.", "The superlative of {s} is {a}."],
    ),
    (
        "group",
        &[
            "what is a group of {p} called",
            "what do you call a group of {p}",
            "a group of {p} is called what",
            "group of {p}",
            "what is the word for a group of {p}",
            "what is the name for a group of {p}",
        ],
        &[
            "{Art}.",
            "It is called {art}.",
            "A group of {p} is called {art}.",
        ],
    ),
    (
        "male",
        &[
            "what is a male {s} called",
            "what do you call a male {s}",
            "male {s}",
            "a boy {s} is called what",
            "what is the name for a male {s}",
        ],
        &["{Art}.", "A male {s} is called {art}."],
    ),
    (
        "female",
        &[
            "what is a female {s} called",
            "what do you call a female {s}",
            "female {s}",
            "a girl {s} is called what",
            "what is the name for a female {s}",
        ],
        &["{Art}.", "A female {s} is called {art}."],
    ),
    (
        "state_capital",
        &[
            "what is the capital of the state of {s}",
            "what is the capital of {s}",
            "capital of {s}",
            "which city is the capital of {s}",
            "what is {s}'s capital",
            "state capital of {s}",
            "what is the state capital of {s}",
        ],
        &["{a}.", "It is {a}.", "The capital of {s} is {a}."],
    ),
];

/// The languages of the `word_*` relations, appended to [`REL_T`] in this order.
pub(super) const LANGUAGES: &[&str] = &["French", "German", "Spanish", "Italian"];

/// Translation questions; `{L}` is replaced by the language when the kit loads.
pub(super) const LANG_Q: &[&str] = &[
    "how do you say {s} in {L}",
    "what is {s} in {L}",
    "{s} in {L}",
    "what is the {L} word for {s}",
    "translate {s} into {L}",
    "how do you say {s} in {L} language",
    "say {s} in {L}",
];

/// Translation answers; `{L}` is replaced like in [`LANG_Q`].
pub(super) const LANG_A: &[&str] = &["{A}.", "In {L} it is {a}.", "{S} in {L} is {a}."];

/// The generic animal questions of version 1: `{l}` label, `{x}` animal with article.
pub(super) const GENERIC_Q: &[&str] = &[
    "what is the {l} of {x}",
    "{l} of {x}",
    "tell me the {l} of {x}",
    "{x}: {l}",
    "do you know the {l} of {x}",
    "what is the {l} for {x}",
];

/// Openers the noise puts before a question.
pub(super) const PREFIX: &[&str] = &[
    "hey tiny, ",
    "tiny, ",
    "please tell me ",
    "can you tell me ",
    "i want to know ",
    "quick question: ",
    "hello, ",
    "hi tiny, ",
    "tell me ",
    "i wonder ",
    "ok, ",
    "so ",
    "excuse me, ",
    "question: ",
    "do you know ",
    "hey, ",
    "could you tell me ",
];

/// Endings the noise adds after a question.
pub(super) const SUFFIX: &[&str] = &[
    " please", " tiny", ", please", " thanks", "?!", "??", "!", " :)", " ?",
];

/// Syllables of made-up words.
pub(super) const SYL: &[&str] = &[
    "ka", "lo", "mi", "zu", "ber", "tan", "vel", "or", "quo", "shi", "dra", "pex", "nim", "gol",
    "fra", "wen", "tri", "xa", "bul", "sno", "ri", "ya", "mox", "ple",
];

/// Endings of made-up country names.
pub(super) const ENDS: &[&str] = &[
    "ia", "land", "stan", "ovia", "ania", "istan", "onia", "ica", "esia", "burg", "ar", "o",
];

/// Ordinal words, indexed by number.
pub(super) const ORD: &[&str] = &[
    "zeroth",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
    "twenty-first",
    "twenty-second",
    "twenty-third",
    "twenty-fourth",
    "twenty-fifth",
    "twenty-sixth",
];

/// Number words below twenty.
pub(super) const ONES: &[&str] = &[
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

/// Number words of the tens, indexed by `n / 10`.
pub(super) const TENS: &[&str] = &[
    "zero", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// The labels of the generic animal questions per relation.
pub(super) const GENERIC: &[(&str, &str)] = &[
    ("class", "animal group"),
    ("legs", "number of legs"),
    ("baby", "baby name"),
    ("food", "food"),
    ("home", "home"),
];

/// Automatic rewordings of one-off facts: first occurrence of `.0` becomes `.1`.
pub(super) const SWAPS: &[(&str, &str)] = &[
    ("what is ", "what's "),
    ("which ", "what "),
    ("what ", "which "),
    ("biggest", "largest"),
    ("largest", "biggest"),
    ("smallest", "tiniest"),
    ("colour", "color"),
    ("color", "colour"),
    ("how many", "what number of"),
    ("do we ", "do you "),
    ("do you ", "do we "),
    ("what are ", "name "),
    ("name the ", "what are the "),
    ("what do we call ", "what is the name for "),
    (" the ", " a "),
    ("is called", "is named"),
    ("who ", "which person "),
    ("hottest", "warmest"),
    ("coldest", "chilliest"),
    ("tallest", "highest"),
    ("highest", "tallest"),
    ("closest", "nearest"),
    ("nearest", "closest"),
    ("make", "create"),
    ("in the world", "on earth"),
    ("on earth", "in the world"),
];

/// Keyboard rows for neighbour-key typos.
pub(super) const KEYS: &[&str] = &["qwertyuiop", "asdfghjkl", "zxcvbnm"];

/// Roman numeral values, largest first.
pub(super) const ROMAN: &[(u32, &str)] = &[
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
];
