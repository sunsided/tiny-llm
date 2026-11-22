//! DOS text attributes and the three colour schemes.
//!
//! The interface is drawn with DOS-style attributes (low nibble foreground,
//! high nibble background, 16 colours, bright backgrounds allowed) so the
//! colour choices of `TINY.C` carry over unchanged. A [`Scheme`] turns them
//! into terminal colours.

use ratatui::style::{Color, Style};

/// A DOS text attribute: `background << 4 | foreground`.
pub type Attr = u8;

/// Electric cyan: frames, `TINY »`.
pub const FRAME: Attr = 11;
/// Dark cyan: secondary frames.
pub const FRAME2: Attr = 3;
/// Cyan on blue: title bar.
pub const TITLE: Attr = 0x1B;
/// White on blue: title bar highlights.
pub const TITLE2: Attr = 0x1F;
/// Green on blue: the "online" lamp.
pub const LAMP: Attr = 0x1A;
/// White on dark cyan: panel names.
pub const PANEL: Attr = 0x3F;
/// Neon pink: `YOU »` and the input prompt.
pub const YOU: Attr = 13;
/// Dark magenta: scanner tail.
pub const MAGENTA: Attr = 5;
/// Dark blue: logo shadow.
pub const SHADOW: Attr = 1;
/// Amber: question text, unsure confidence.
pub const QUESTION: Attr = 14;
/// Neon mint: answer text, sure confidence.
pub const ANSWER: Attr = 10;
/// Light blue: logo middle row.
pub const BLUE: Attr = 9;
/// White: the newest answer letter, bright text.
pub const BRIGHT: Attr = 15;
/// Grey-blue: labels and hints.
pub const DIM: Attr = 8;
/// Light grey: normal text.
pub const TEXT: Attr = 7;
/// White on green: the word being spoken.
pub const SPEAK: Attr = 0x2F;
/// Status bar.
pub const BAR: Attr = 0x30;
/// Status bar key names.
pub const BAR_KEY: Attr = 0x3F;
/// Status bar values.
pub const BAR_VALUE: Attr = 0x3E;
/// Hot pink-red: guessing.
pub const HOT: Attr = 12;

/// How attributes become terminal colours.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// The neon VGA palette of TINY 2.1, as 24-bit colour.
    Neon,
    /// The terminal's own 16 colours.
    Classic,
    /// Greys only.
    Mono,
}

/// The neon VGA DAC values (0..=63 per channel) for the 16 text colours.
const NEON: [[u8; 3]; 16] = [
    [1, 1, 7],
    [5, 7, 22],
    [0, 26, 18],
    [0, 20, 30],
    [40, 4, 12],
    [28, 6, 40],
    [42, 26, 4],
    [40, 45, 54],
    [13, 16, 28],
    [18, 34, 63],
    [18, 63, 42],
    [10, 57, 63],
    [63, 22, 34],
    [63, 18, 58],
    [63, 54, 14],
    [60, 62, 63],
];

/// The 16 CGA colours in DOS order, as named terminal colours.
const CLASSIC: [Color; 16] = [
    Color::Black,
    Color::Blue,
    Color::Green,
    Color::Cyan,
    Color::Red,
    Color::Magenta,
    Color::Yellow,
    Color::Gray,
    Color::DarkGray,
    Color::LightBlue,
    Color::LightGreen,
    Color::LightCyan,
    Color::LightRed,
    Color::LightMagenta,
    Color::LightYellow,
    Color::White,
];

impl Scheme {
    /// Terminal style of a DOS attribute.
    #[must_use]
    pub fn style(self, attr: Attr) -> Style {
        let (fg, bg) = match self {
            Self::Mono => mono(attr),
            _ => (attr & 15, attr >> 4),
        };

        Style::new().fg(self.color(fg)).bg(self.color(bg))
    }

    /// Terminal colour of DOS colour index `index` (0..=15).
    fn color(self, index: u8) -> Color {
        let i = usize::from(index & 15);
        match self {
            Self::Neon => {
                let [r, g, b] = NEON[i].map(|v| v * 4 + v / 16);
                Color::Rgb(r, g, b)
            }
            Self::Classic | Self::Mono => CLASSIC[i],
        }
    }
}

/// Grey mapping of `TINY.C`: any background becomes black on grey; dim text
/// grey, bright colours white, everything else grey.
fn mono(attr: Attr) -> (u8, u8) {
    let (fg, bg) = (attr & 15, (attr >> 4) & 7);
    if bg != 0 {
        (0, 7)
    } else if fg >= 9 {
        (15, 0)
    } else {
        (7, 0)
    }
}
