//! The conversation history and how it wraps into screen lines.

use ratatui::text::{Line, Span};

use super::palette::{self, Attr, Scheme};

/// Lines of the help panel shown by F1 and HELP.
const HELP: [&str; 7] = [
    "Ask a question and press Enter.",
    "Try: capital of peru · is a whale a fish",
    "     what is 7 times 8 · spell elephant",
    "     which countries border estonia",
    "     how do you say thank you in spanish",
    "F2 sound · F3 voice · F4 creativity",
    "↑ last question · Esc quit · CLS clears",
];

/// Width of the `" YOU » "` / `"TINY » "` labels; wrapped text lines up after them.
const LABEL_WIDTH: usize = 7;

/// One item of the conversation.
#[derive(Debug, Clone)]
pub enum Entry {
    /// Dim informational text, like the greeting.
    Note(&'static str),
    /// The help panel.
    Help,
    /// A question and its (possibly unfinished) answer.
    Turn(Turn),
}

/// A question and TINY's answer to it.
#[derive(Debug, Clone)]
pub struct Turn {
    /// The cleaned question.
    pub question: String,
    /// Answer letters written so far.
    pub answer: String,
    /// Where the network is with this turn.
    pub state: TurnState,
    /// Byte range of the answer word being spoken, highlighted.
    pub spoken: Option<std::ops::Range<usize>>,
}

/// Progress of a [`Turn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState {
    /// Reading the question: a scanner sweeps.
    Reading {
        /// Share of the question read, 0..=100.
        percent: usize,
        /// Animation counter of the scanner.
        spin: usize,
    },
    /// Writing the answer; the newest letter glows.
    Writing {
        /// Random glyph shown for one frame where the next letter will appear.
        flicker: Option<char>,
    },
    /// Finished.
    Done,
}

/// The whole conversation.
#[derive(Debug, Clone, Default)]
pub struct Chat {
    /// Entries, oldest first.
    entries: Vec<Entry>,
}

impl Chat {
    /// Adds an entry at the bottom.
    pub fn push(&mut self, entry: Entry) {
        self.entries.push(entry);
    }

    /// Forgets everything (CLS).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// The newest turn, if the last entry is one.
    pub fn last_turn(&mut self) -> Option<&mut Turn> {
        match self.entries.last_mut() {
            Some(Entry::Turn(turn)) => Some(turn),
            _ => None,
        }
    }

    /// All entries as screen lines of at most `width` cells.
    #[must_use]
    pub fn lines(&self, width: usize, scheme: Scheme) -> Vec<Line<'static>> {
        let mut out = Vec::new();
        for entry in &self.entries {
            match entry {
                Entry::Note(text) => out.push(Line::styled(*text, scheme.style(palette::DIM))),
                Entry::Help => help_lines(&mut out, scheme),
                Entry::Turn(turn) => turn_lines(&mut out, turn, width, scheme),
            }
        }

        out
    }
}

/// Appends the help panel.
fn help_lines(out: &mut Vec<Line<'static>>, scheme: Scheme) {
    out.push(Line::default());
    out.push(Line::from(Span::styled(
        " HELP ",
        scheme.style(palette::PANEL),
    )));
    for text in HELP {
        out.push(Line::styled(
            format!(" {text}"),
            scheme.style(palette::TEXT),
        ));
    }
}

/// Appends a blank line, the question and the answer.
fn turn_lines(out: &mut Vec<Line<'static>>, turn: &Turn, width: usize, scheme: Scheme) {
    let question: Vec<_> = turn
        .question
        .chars()
        .map(|c| (c, palette::QUESTION))
        .collect();

    out.push(Line::default());
    wrap(out, (" YOU » ", palette::YOU), &question, width, scheme);
    wrap(
        out,
        ("TINY » ", palette::FRAME),
        &answer_cells(turn),
        width,
        scheme,
    );
}

/// The coloured letters after `TINY »`, depending on the turn's state.
fn answer_cells(turn: &Turn) -> Vec<(char, Attr)> {
    let mut cells: Vec<_> = turn.answer.chars().map(|c| (c, palette::ANSWER)).collect();
    if let Some(word) = &turn.spoken {
        for cell in cells.get_mut(word.clone()).into_iter().flatten() {
            cell.1 = palette::SPEAK;
        }
    }

    match turn.state {
        TurnState::Reading { percent, spin } => {
            cells.extend(scanner(spin));
            cells.extend(
                format!(" reading {percent}%")
                    .chars()
                    .map(|c| (c, palette::DIM)),
            );
        }
        TurnState::Writing { flicker } => {
            if let Some(last) = cells.last_mut() {
                last.1 = palette::BRIGHT;
            }

            cells.extend(flicker.map(|c| (c, palette::FRAME)));
        }
        TurnState::Done => {}
    }

    cells
}

/// Ten cells of a light bouncing back and forth with a fading tail.
fn scanner(spin: usize) -> impl Iterator<Item = (char, Attr)> {
    let head = match spin % 18 {
        h if h >= 9 => 17 - h,
        h => h,
    };

    (0..10usize).map(move |k| match head.abs_diff(k) {
        0 => ('█', palette::BRIGHT),
        1 => ('▓', palette::YOU),
        2 => ('▒', palette::MAGENTA),
        _ => ('·', palette::DIM),
    })
}

/// Appends `label` and `cells`, wrapping letter by letter at `width` with
/// continuation lines indented to line up after the label.
fn wrap(
    out: &mut Vec<Line<'static>>,
    label: (&'static str, Attr),
    cells: &[(char, Attr)],
    width: usize,
    scheme: Scheme,
) {
    let per_line = width.saturating_sub(LABEL_WIDTH).max(1);
    let mut chunks = cells.chunks(per_line);
    let first = chunks.next().unwrap_or(&[]);

    let mut line = vec![Span::styled(label.0, scheme.style(label.1))];
    line.extend(spans(first, scheme));
    out.push(Line::from(line));

    for chunk in chunks {
        let mut line = vec![Span::raw(" ".repeat(LABEL_WIDTH))];
        line.extend(spans(chunk, scheme));
        out.push(Line::from(line));
    }
}

/// One span per coloured cell.
fn spans(cells: &[(char, Attr)], scheme: Scheme) -> impl Iterator<Item = Span<'static>> + '_ {
    cells
        .iter()
        .map(move |&(c, attr)| Span::styled(c.to_string(), scheme.style(attr)))
}
