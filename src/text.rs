//! Letters ↔ tokens, and clean-up of typed questions.

/// Token that ends a question and an answer.
pub const NEWLINE: usize = 1;

/// Longest question, in letters.
pub const MAX_QUESTION: usize = 72;

/// Token of a printable ASCII letter (`' '..='~'` → 2..=96).
///
/// # Panics
///
/// Panics if `c` is not printable ASCII; [`clean`] guarantees that.
#[must_use]
pub fn token(c: u8) -> usize {
    assert!(is_printable(c), "{c} is not a printable letter");
    usize::from(c - 30)
}

/// Letter of a token from 2..=96, `None` for padding and newline.
#[must_use]
pub fn letter(token: usize) -> Option<char> {
    let byte = u8::try_from(token + 30).ok()?;
    (token > NEWLINE && is_printable(byte)).then_some(char::from(byte))
}

/// Cleans a typed line: printable letters only, at most [`MAX_QUESTION`] of them,
/// no trailing spaces, and all-capitals text in lower case (the network knows it better).
#[must_use]
pub fn clean(line: &str) -> String {
    let mut q: String = line
        .bytes()
        .filter(|&c| is_printable(c))
        .take(MAX_QUESTION)
        .map(char::from)
        .collect();
    q.truncate(q.trim_end_matches(' ').len());

    let has_lower = q.bytes().any(|c| c.is_ascii_lowercase());
    if !has_lower {
        q.make_ascii_lowercase();
    }

    q
}

/// Whether `q` is the command word `word`, ignoring case.
#[must_use]
pub fn is_word(q: &str, word: &str) -> bool {
    q.eq_ignore_ascii_case(word)
}

/// Whether `c` is in the network's alphabet.
fn is_printable(c: u8) -> bool {
    (32..=126).contains(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Control characters go, trailing spaces go, CAPS LOCK text is lowered.
    #[test]
    fn clean_like_tiny_c() {
        assert_eq!(clean("WHAT IS 7 TIMES 8?  \n"), "what is 7 times 8?");
        assert_eq!(clean("Is a Whale\ta fish"), "Is a Whalea fish");
        assert_eq!(clean(&"x".repeat(100)).len(), MAX_QUESTION);
    }

    /// Tokens and letters round-trip; newline has no letter.
    #[test]
    fn tokens_round_trip() {
        assert_eq!(token(b' '), 2);
        assert_eq!(letter(token(b'~')), Some('~'));
        assert_eq!(letter(NEWLINE), None);
    }
}
