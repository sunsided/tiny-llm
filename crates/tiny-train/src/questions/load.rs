//! Reading `facts/*.txt` and `facts/*.tsv` and deriving the tables of
//! `gen2.py`'s module level, in Python's iteration orders.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

use super::{Fact, Kit, Pairs, Relation, Reverse, Splits, split, tables};

/// Words added to the spelling pool on top of the table words.
const EXTRA_WORDS: &str = "red blue green yellow black white orange purple pink brown apple banana \
    bread milk water house school book pencil table chair window door tree flower sun moon star \
    river mountain ocean computer keyboard mouse screen disk tiny hello friend";

/// Relations whose subjects join the spelling pool.
const POOL_SUBJECTS: [&str; 6] = ["class", "legs", "baby", "sound", "food", "home"];

/// Relations whose subjects and answers join the spelling pool.
const POOL_BOTH: [&str; 3] = ["opposite", "plural", "past"];

impl Kit {
    /// Loads the fact files from `dir` (the original's `facts/`) and builds
    /// every derived table.
    ///
    /// # Errors
    /// If a file cannot be read, is not ASCII, or has a malformed line.
    pub fn load(dir: &Path) -> Result<Self> {
        let txt = load_txt(dir)?;
        let rel = load_tsv(dir)?;
        let borders = load_borders(dir)?;
        let get = |r: &str| rel.iter().find(|(n, _)| n == r).map_or(&[][..], |(_, p)| p);
        let d: HashMap<String, HashMap<String, String>> = rel
            .iter()
            .map(|(r, p)| (r.clone(), p.iter().cloned().collect()))
            .collect();
        let (rels, rel_weights) = relations(&rel);
        let rev = reverse_tables(&get, &rel);
        let countries: Vec<String> = BTreeSet::from_iter(d["capital"].keys().cloned())
            .into_iter()
            .collect();
        let words = word_pool(&get, get("capital"));
        let real_words = countries
            .iter()
            .chain(&rev.cap_rev_keys)
            .map(|x| x.to_lowercase())
            .chain(words.iter().cloned())
            .collect();
        let values = |r: &str| -> Vec<String> {
            BTreeSet::from_iter(d[r].values().cloned())
                .into_iter()
                .collect()
        };

        let keys_both = |a: &str, b: &str| -> Vec<String> {
            BTreeSet::from_iter(d[a].keys().filter(|k| d[b].contains_key(*k)).cloned())
                .into_iter()
                .collect()
        };

        Ok(Self {
            classes: values("class"),
            conts: values("continent"),
            caps_by_cont: caps_by_cont(get("capital"), &d["continent"]),
            langs: langs(&d["language"]),
            sounds: values("sound"),
            opposites: values("opposite"),
            legs_sound: keys_both("legs", "sound"),
            baby_class: keys_both("baby", "class"),
            split: splits(countries.clone(), words),
            countries,
            real: real_words,
            plural: get("plural").iter().cloned().collect(),
            rev,
            rels,
            rel_weights,
            borders,
            d,
            rel,
            txt,
        })
    }
}

/// The reverse tables: `CAP_REV`, `LANG_REV`, ... and `MEANING`.
fn reverse_tables<'a>(
    get: &impl Fn(&str) -> &'a [(String, String)],
    rel: &[(String, Pairs)],
) -> Reverse {
    let single = |g: Vec<(String, Vec<String>)>| -> Pairs {
        g.into_iter()
            .filter(|(_, ss)| ss.len() == 1)
            .map(|(a, mut ss)| (a, ss.remove(0)))
            .collect()
    };

    let cap_rev = single(group(get("capital")));
    let changed = |r: &str| -> Pairs {
        sorted(
            reverse(get(r))
                .into_iter()
                .filter(|(a, s)| a != s)
                .collect(),
        )
    };

    let mut atom_rev = reverse(get("atomic"));
    atom_rev.sort_by_key(|(a, _)| a.parse::<i64>().unwrap_or(i64::MAX));
    Reverse {
        cap_rev_keys: cap_rev.iter().map(|(a, _)| a.clone()).collect(),
        cap_rev: sorted(cap_rev),
        lang_rev: sorted_groups(get("language")),
        cur_rev: sorted_groups(get("currency")),
        sound_rev: sorted_groups(get("sound")),
        baby_rev: sorted_groups(get("baby")),
        plural_rev: changed("plural"),
        past_rev: changed("past"),
        monthnum: sorted(reverse(get("month_number"))),
        sym_rev: sorted(reverse(get("symbol"))),
        atom_rev,
        dom_rev: sorted(single(group(get("domain")))),
        state_rev: sorted(reverse(get("state_capital"))),
        meaning: meaning(rel),
    }
}

/// The splits with `gen2.py`'s fractions and seeds.
fn splits(countries: Vec<String>, words: Vec<String>) -> Splits {
    Splits {
        yn_countries: split(countries, 0.08, 3),
        add: split(pairs(0..21, |_| 0..21), 0.1, 11),
        sub: split(pairs(0..21, |a| 0..=a), 0.1, 12),
        mul: split(pairs(1..13, |_| 1..13), 0.08, 13),
        words: split(words, 0.1, 21),
        roman: split((1..101).chain(1900..2031).collect(), 0.08, 31),
        sq: split((0..21).collect(), 0.1, 32),
    }
}

/// The fact files of one extension in `glob`'s sorted order (no dot files).
fn files(dir: &Path, ext: &str) -> Result<Vec<(String, String)>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<_>>()?;
    paths.retain(|p| {
        p.extension().is_some_and(|e| e == ext)
            && p.file_name()
                .is_some_and(|n| !n.to_string_lossy().starts_with('.'))
    });
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let text =
                std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
            ensure!(text.is_ascii(), "{} is not ASCII", p.display());
            let name = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            Ok((name, text))
        })
        .collect()
}

/// `load_txt`: `q1 | q2 | q3 => a1 | a2` lines; with three or more
/// phrasings the last is kept for testing.
fn load_txt(dir: &Path) -> Result<Vec<Fact>> {
    let mut facts = Vec::new();
    for (name, text) in files(dir, "txt")? {
        for line in text.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let Some((qs, ans)) = line.split_once("=>").filter(|(_, a)| !a.contains("=>")) else {
                bail!("{name}: expected one `=>` in {line:?}");
            };

            let list =
                |s: &str| -> Vec<String> { s.split('|').map(|x| x.trim().to_owned()).collect() };
            let mut train_q = list(qs);
            let test_q = if train_q.len() >= 3 {
                train_q.split_off(train_q.len() - 1)
            } else {
                Vec::new()
            };

            facts.push(Fact {
                train_q,
                test_q,
                answers: list(ans),
            });
        }
    }

    Ok(facts)
}

/// The three tab-separated fields of a TSV line.
fn fields<'a>(name: &str, line: &'a str) -> Result<(&'a str, &'a str, &'a str)> {
    match line.split('\t').collect::<Vec<_>>()[..] {
        [r, s, a] => Ok((r, s, a)),
        _ => bail!("{name}: expected three fields in {line:?}"),
    }
}

/// `load_tsv`: relation rows in first-seen relation order.
fn load_tsv(dir: &Path) -> Result<Vec<(String, Pairs)>> {
    let mut rel: Vec<(String, Pairs)> = Vec::new();
    for (name, text) in files(dir, "tsv")? {
        for line in text.lines() {
            let (r, s, a) = fields(&name, line)?;
            if (name == "geo.tsv" && (r == "language" || r == "currency")) || r == "borders" {
                continue;
            }

            let s = if name == "elements.tsv" {
                s.to_lowercase()
            } else {
                s.to_owned()
            };

            let pair = (s, a.to_owned());
            match rel.iter_mut().find(|(n, _)| n == r) {
                Some((_, p)) => p.push(pair),
                None => rel.push((r.to_owned(), vec![pair])),
            }
        }
    }

    Ok(rel)
}

/// `BORDERS` from `geo3.tsv`: `-` means no land neighbours.
fn load_borders(dir: &Path) -> Result<HashMap<String, Vec<String>>> {
    let path = dir.join("geo3.tsv");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let mut borders = HashMap::new();
    for line in text.lines() {
        let (r, s, a) = fields("geo3.tsv", line)?;
        if r == "borders" {
            let nb = if a == "-" {
                Vec::new()
            } else {
                a.split('|').map(str::to_owned).collect()
            };

            borders.insert(s.to_owned(), nb);
        }
    }

    Ok(borders)
}

/// `REL_NAMES` with templates and `REL_W` weights (`len ** 0.75`).
fn relations(rel: &[(String, Pairs)]) -> (Vec<Relation>, Vec<f64>) {
    let owned = |xs: &[&str]| -> Vec<String> { xs.iter().map(|&x| x.to_owned()).collect() };
    let lang =
        |xs: &[&str], l: &str| -> Vec<String> { xs.iter().map(|x| x.replace("{L}", l)).collect() };
    let all = tables::REL_T
        .iter()
        .map(|(n, q, a)| Relation {
            name: (*n).to_owned(),
            questions: owned(q),
            answers: owned(a),
        })
        .chain(tables::LANGUAGES.iter().map(|l| Relation {
            name: format!("word_{l}"),
            questions: lang(tables::LANG_Q, l),
            answers: lang(tables::LANG_A, l),
        }));
    let mut rels = Vec::new();
    let mut weights = Vec::new();
    for r in all {
        if let Some((_, p)) = rel.iter().find(|(n, _)| *n == r.name) {
            #[expect(clippy::cast_precision_loss, reason = "table sizes are small")]
            weights.push((p.len() as f64).powf(0.75));
            rels.push(r);
        }
    }

    (rels, weights)
}

/// `group`: answer to subjects, in first-seen answer order.
fn group(pairs: &[(String, String)]) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    let mut index = HashMap::new();
    for (s, a) in pairs {
        let i = *index.entry(a.clone()).or_insert_with(|| {
            out.push((a.clone(), Vec::new()));
            out.len() - 1
        });
        out[i].1.push(s.clone());
    }

    out
}

/// `sorted(group(...).items())`.
fn sorted_groups(pairs: &[(String, String)]) -> Vec<(String, Vec<String>)> {
    let mut g = group(pairs);
    g.sort_by(|x, y| x.0.cmp(&y.0));
    g
}

/// `{a: s for s, a in pairs}` in insertion order, the last subject winning.
fn reverse(pairs: &[(String, String)]) -> Pairs {
    let mut out: Pairs = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for (s, a) in pairs {
        if let Some(&i) = index.get(a) {
            out[i].1.clone_from(s);
        } else {
            index.insert(a.clone(), out.len());
            out.push((a.clone(), s.clone()));
        }
    }

    out
}

/// `sorted(...)` of a list of pairs.
fn sorted(mut pairs: Pairs) -> Pairs {
    pairs.sort();
    pairs
}

/// `MEANING`: lower-cased foreign word to its English words, over the
/// `word_*` relations in `REL` order.
fn meaning(rel: &[(String, Pairs)]) -> Vec<(String, Vec<String>)> {
    let mut m: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (_, pairs) in rel.iter().filter(|(r, _)| r.starts_with("word_")) {
        for (e, x) in pairs {
            let es = m.entry(x.to_lowercase()).or_default();
            if !es.contains(e) {
                es.push(e.clone());
            }
        }
    }

    m.into_iter().collect()
}

/// `CAPS_BY_CONT`: capitals by the continent of their country.
fn caps_by_cont(
    capital: &[(String, String)],
    continent: &HashMap<String, String>,
) -> HashMap<Option<String>, Vec<String>> {
    let mut m: HashMap<Option<String>, Vec<String>> = HashMap::new();
    for (s, c) in capital {
        m.entry(continent.get(s).cloned())
            .or_default()
            .push(c.clone());
    }

    m
}

/// `LANGS`: every single language named in `D['language']`.
fn langs(language: &HashMap<String, String>) -> Vec<String> {
    let set: BTreeSet<String> = language
        .values()
        .flat_map(|v| {
            v.replace(", ", " and ")
                .split(" and ")
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect();
    set.into_iter().collect()
}

/// `[(a, b) for a in outer for b in inner(a)]`.
fn pairs<I: IntoIterator<Item = i64>>(
    outer: std::ops::Range<i64>,
    inner: impl Fn(i64) -> I,
) -> Vec<(i64, i64)> {
    outer
        .flat_map(|a| inner(a).into_iter().map(move |b| (a, b)))
        .collect()
}

/// `word_pool`: lower-case words of 2 to 9 letters for the letter games.
fn word_pool<'a>(
    get: &impl Fn(&str) -> &'a [(String, String)],
    capital: &[(String, String)],
) -> Vec<String> {
    let mut w: BTreeSet<String> = BTreeSet::new();
    for r in POOL_SUBJECTS {
        w.extend(get(r).iter().map(|(s, _)| s.clone()));
    }

    for r in POOL_BOTH {
        w.extend(get(r).iter().flat_map(|(s, a)| [s.clone(), a.clone()]));
    }

    w.extend(capital.iter().map(|(_, c)| c.to_lowercase()));
    w.extend(EXTRA_WORDS.split_whitespace().map(str::to_owned));
    // ASCII only (checked on load): isalpha() and islower() mean a-z only.
    w.into_iter()
        .filter(|x| (2..=9).contains(&x.len()) && x.bytes().all(|b| b.is_ascii_lowercase()))
        .collect()
}
