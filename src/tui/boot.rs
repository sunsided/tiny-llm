//! The boot sequence: digital rain, the logo assembling out of noise, the
//! weights "loading" layer by layer, then SYSTEM READY.
//!
//! It draws onto its own 80 x 25 [`Canvas`] the way `TINY.C` draws onto the
//! DOS screen (cells stay until overwritten), and that canvas is copied into
//! the middle of the terminal every frame.
//!
//! ```mermaid
//! stateDiagram-v2
//!     [*] --> Rain
//!     Rain --> Logo
//!     Logo --> Sweep
//!     Sweep --> Loading
//!     Loading --> Ready
//!     Ready --> Waiting
//!     Rain --> Waiting: any key
//!     Logo --> Waiting: any key
//!     Loading --> Waiting: any key
//! ```

use rand::RngExt;
use rand::rngs::ThreadRng;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use super::palette::{self, Attr, Scheme};
use crate::sound::{Sound, Tone, effects};

/// Canvas width in cells.
const WIDTH: u16 = 80;
/// Canvas height in cells.
const HEIGHT: u16 = 25;

/// The TINY logo; every `#` becomes two full blocks.
const LOGO: [&[u8; 24]; 5] = [
    b"#####  ###  #   #  #   #",
    b"  #     #   ##  #   # # ",
    b"  #     #   # # #    #  ",
    b"  #     #   #  ##    #  ",
    b"  #    ###  #   #    #  ",
];
/// Colour of each logo row.
const LOGO_COLORS: [Attr; 5] = [11, 11, palette::BLUE, palette::YOU, palette::YOU];
/// Left edge of the logo.
const LOGO_X: u16 = 15;
/// Top edge of the logo.
const LOGO_Y: u16 = 1;

/// Frames of digital rain.
const RAIN_FRAMES: usize = 46;
/// First row of the loading log.
const LOG_Y: u16 = 16;

/// Where the boot sequence is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Falling letters (one rain frame per two ticks); the counter is the tick.
    Rain(usize),
    /// Logo cells settling; the counter is the frame number.
    Logo(usize),
    /// A light sweeping over the logo; the counter is its column.
    Sweep(u16),
    /// Weight tables "loading"; the counter is tables done.
    Loading(usize),
    /// SYSTEM READY blinking; the counter is the blink number.
    Ready(usize),
    /// Finished, waiting for a key.
    Waiting,
}

/// A fixed grid of coloured letters.
#[derive(Debug, Clone)]
struct Canvas {
    /// Row-major cells, `WIDTH * HEIGHT` long.
    cells: Vec<(char, Attr)>,
}

impl Canvas {
    /// A blank canvas.
    fn new() -> Self {
        Self {
            cells: vec![(' ', palette::TEXT); usize::from(WIDTH * HEIGHT)],
        }
    }

    /// Sets one cell; positions off the canvas are ignored.
    fn put(&mut self, x: u16, y: u16, c: char, attr: Attr) {
        if x < WIDTH && y < HEIGHT {
            self.cells[usize::from(y * WIDTH + x)] = (c, attr);
        }
    }

    /// Writes `text` from `(x, y)` rightwards.
    fn text(&mut self, x: u16, y: u16, text: &str, attr: Attr) {
        for (dx, c) in (0..).zip(text.chars()) {
            self.put(x + dx, y, c, attr);
        }
    }

    /// Copies the canvas into the middle of `area`.
    fn blit(&self, area: Rect, buf: &mut Buffer, scheme: Scheme) {
        let x0 = area.x + area.width.saturating_sub(WIDTH) / 2;
        let y0 = area.y + area.height.saturating_sub(HEIGHT) / 2;
        let rows = self.cells.chunks_exact(usize::from(WIDTH));

        for (y, row) in (y0..area.bottom()).zip(rows) {
            for (x, &(c, attr)) in (x0..area.right()).zip(row) {
                buf[(x, y)].set_char(c).set_style(scheme.style(attr));
            }
        }
    }
}

/// The boot sequence's state.
#[derive(Debug, Clone)]
pub struct Boot {
    /// What is on screen.
    canvas: Canvas,
    /// Current stage.
    stage: Stage,
    /// Rain head row of every even column (may be negative: not yet visible).
    rain: Vec<(i32, i32)>,
    /// Logo cells already settled.
    settled: [[bool; 24]; 5],
    /// Number of weight tables in the network.
    tables: usize,
    /// Number of layers in the network.
    layers: usize,
    /// Next free row of the loading log.
    log_y: u16,
    /// Random numbers for the effects.
    rng: ThreadRng,
    /// Sounds of the current frame, played at the end of [`Boot::tick`].
    cues: Vec<Tone>,
}

impl Boot {
    /// Starts the sequence for a network with `layers` blocks.
    #[must_use]
    pub fn new(layers: usize, tables: usize) -> Self {
        let mut rng = rand::rng();
        let rain = (0..WIDTH / 2)
            .map(|_| (-rng.random_range(0..25), rng.random_range(1..=2)))
            .collect();

        Self {
            canvas: Canvas::new(),
            stage: Stage::Rain(0),
            rain,
            settled: [[false; 24]; 5],
            tables,
            layers,
            log_y: LOG_Y,
            rng,
            cues: Vec::new(),
        }
    }

    /// Whether the sequence is over and waiting for a key.
    #[must_use]
    pub fn is_waiting(&self) -> bool {
        self.stage == Stage::Waiting
    }

    /// Advances one frame (about 15 ms) and plays its sounds.
    pub fn tick(&mut self, sound: &Sound) {
        self.stage = match self.stage {
            Stage::Rain(t) if t < 2 * RAIN_FRAMES => {
                if t % 2 == 0 {
                    self.rain_frame(t / 2);
                }

                Stage::Rain(t + 1)
            }
            Stage::Rain(_) => {
                self.canvas = Canvas::new();
                Stage::Logo(0)
            }
            Stage::Logo(f) => self.logo_frame(f),
            Stage::Sweep(x) => self.sweep_frame(x),
            Stage::Loading(done) => self.loading_frame(done),
            Stage::Ready(n) => self.ready_frame(n),
            Stage::Waiting => Stage::Waiting,
        };

        sound.play(&self.cues);
        self.cues.clear();
    }

    /// Jumps straight to the finished screen.
    pub fn skip(&mut self) {
        self.canvas = Canvas::new();
        self.log_y = LOG_Y;
        self.draw_logo();
        self.draw_titles();
        for done in 1..=self.tables {
            self.draw_progress(done);
        }

        self.draw_ready(true);
        self.draw_prompt();
        self.stage = Stage::Waiting;
    }

    /// Draws the canvas into the middle of `area`.
    pub fn render(&self, area: Rect, buf: &mut Buffer, scheme: Scheme) {
        self.canvas.blit(area, buf, scheme);
    }

    /// Rain frame `f`: each column's head moves down, its tail is erased.
    fn rain_frame(&mut self, f: usize) {
        if f.is_multiple_of(4) {
            let hz = self.rng.random_range(3000.0..6000.0);
            self.cues.extend(effects::rain(hz));
        }

        let trail = if f < RAIN_FRAMES - 6 {
            palette::ANSWER
        } else {
            palette::DIM
        };
        for (x, (head, speed)) in (0..).step_by(2).zip(self.rain.clone()) {
            let (behind, ahead) = (self.glyph(), self.glyph());
            self.put_row(x, head - 1, behind, trail);
            self.put_row(x, head, ahead, palette::BRIGHT);
            self.put_row(x, head - 8, ' ', palette::TEXT);

            let next = head + speed;
            let next = if next > 32 {
                -self.rng.random_range(0..10)
            } else {
                next
            };
            self.rain[usize::from(x / 2)].0 = next;
        }
    }

    /// A random printable letter.
    fn glyph(&mut self) -> char {
        char::from(self.rng.random_range(33..123u8))
    }

    /// Puts a cell at a row that may be off screen.
    fn put_row(&mut self, x: u16, y: i32, c: char, attr: Attr) {
        if let Ok(y) = u16::try_from(y) {
            self.canvas.put(x, y, c, attr);
        }
    }

    /// One frame of the logo settling: five cells land, some noise flickers.
    fn logo_frame(&mut self, f: usize) -> Stage {
        if f.is_multiple_of(3) {
            let settled = self.settled.iter().flatten().filter(|&&s| s).count();
            self.cues.extend(effects::settle(to_u16(settled)));
        }

        for _ in 0..5 {
            let (y, x) = self.random_logo_cell();
            if is_logo(y, x) && !self.settled[y][x] {
                self.settled[y][x] = true;
                self.draw_logo_cell(y, x, LOGO_COLORS[y]);
            }
        }

        for _ in 0..6 {
            let (y, x) = self.random_logo_cell();
            if is_logo(y, x) && !self.settled[y][x] {
                let noise = ['░', '▒', '▓'][self.rng.random_range(0..3)];
                let (cx, cy) = logo_pos(y, x);
                self.canvas
                    .text(cx, cy, &format!("{noise}{noise}"), palette::DIM);
            }
        }

        let all_settled = (0..5).all(|y| (0..24).all(|x| !is_logo(y, x) || self.settled[y][x]));
        if all_settled || f > 400 {
            self.draw_logo();
            return Stage::Sweep(0);
        }

        Stage::Logo(f + 1)
    }

    /// One frame of the light sweep over the logo.
    fn sweep_frame(&mut self, x: u16) -> Stage {
        let col = usize::from(x);
        for (y, &attr) in LOGO_COLORS.iter().enumerate() {
            if col < 24 && is_logo(y, col) {
                self.draw_logo_cell(y, col, palette::BRIGHT);
            }

            if (2..26).contains(&col) && is_logo(y, col - 2) {
                self.draw_logo_cell(y, col - 2, attr);
            }
        }

        if x < 25 {
            Stage::Sweep(x + 1)
        } else {
            self.draw_titles();
            Stage::Loading(0)
        }
    }

    /// One weight table "loads" every other frame.
    fn loading_frame(&mut self, done: usize) -> Stage {
        if done >= 2 * self.tables {
            self.cues.extend(effects::boot());
            return Stage::Ready(0);
        }

        if done % 2 == 1 {
            self.draw_progress(done / 2 + 1);
        }

        Stage::Loading(done + 1)
    }

    /// SYSTEM READY blinks six times, eight frames each.
    fn ready_frame(&mut self, n: usize) -> Stage {
        self.draw_ready(n / 8 % 2 == 1);
        if n < 48 {
            return Stage::Ready(n + 1);
        }

        self.draw_prompt();
        Stage::Waiting
    }

    /// A random logo cell as (row, column).
    fn random_logo_cell(&mut self) -> (usize, usize) {
        (self.rng.random_range(0..5), self.rng.random_range(0..24))
    }

    /// Draws one logo cell, with its shadow unless another cell covers it.
    fn draw_logo_cell(&mut self, y: usize, x: usize, attr: Attr) {
        let (cx, cy) = logo_pos(y, x);
        if !(y < 4 && x < 23 && is_logo(y + 1, x + 1)) {
            self.canvas.text(cx + 2, cy + 1, "▒▒", palette::SHADOW);
        }

        self.canvas.text(cx, cy, "██", attr);
    }

    /// Draws the complete logo.
    fn draw_logo(&mut self) {
        for (y, &attr) in LOGO_COLORS.iter().enumerate() {
            for x in (0..24).filter(|&x| is_logo(y, x)) {
                self.draw_logo_cell(y, x, attr);
            }
        }
    }

    /// Draws the texts under the logo.
    fn draw_titles(&mut self) {
        self.canvas
            .text(23, 7, "N E U R A L    I N T E R F A C E", palette::BRIGHT);
        self.canvas
            .text(28, 8, "version 2.1  ·  Rust edition", palette::DIM);
        self.canvas.text(9, 10, &"─".repeat(60), palette::FRAME2);
        self.canvas.text(
            21,
            12,
            "LOADING NEURAL WEIGHTS FROM TINY.BIN",
            palette::FRAME,
        );
    }

    /// Draws the loading bar at `done` tables and, at layer boundaries, a log line.
    fn draw_progress(&mut self, done: usize) {
        let filled = 36 * done / self.tables;
        self.canvas.put(21, 14, '▐', palette::FRAME);
        for (x, i) in (22..).zip(0..36) {
            let attr = match i {
                i if i >= filled => palette::DIM,
                i if i * 3 < filled => palette::FRAME,
                _ => palette::ANSWER,
            };
            self.canvas
                .put(x, 14, if i < filled { '█' } else { '·' }, attr);
        }

        self.canvas.put(58, 14, '▌', palette::FRAME);
        let percent = format!(" {:3}%", 100 * done / self.tables);
        self.canvas.text(59, 14, &percent, palette::BRIGHT);

        if done == 2 || (done > 2 && (done - 2).is_multiple_of(6)) {
            let what = if done == 2 {
                "letters + places".to_owned()
            } else {
                format!("layer {} of {}", (done - 2) / 6, self.layers)
            };
            self.log_line(&what);
        }
    }

    /// Writes `► what ······ ONLINE` on the next log row.
    fn log_line(&mut self, what: &str) {
        let y = self.log_y;
        let dots = 22usize.saturating_sub(what.chars().count());
        self.canvas.text(25, y, "► ", palette::YOU);
        self.canvas.text(27, y, &format!("{what} "), palette::TEXT);
        self.canvas.text(
            28 + to_u16(what.chars().count()),
            y,
            &"·".repeat(dots),
            palette::DIM,
        );
        self.canvas.text(49, y, " ONLINE", palette::ANSWER);
        self.log_y += 1;
        self.cues.extend(effects::online(self.log_y));
    }

    /// Draws SYSTEM READY, bright or mint.
    fn draw_ready(&mut self, bright: bool) {
        let attr = if bright {
            palette::BRIGHT
        } else {
            palette::ANSWER
        };
        self.canvas
            .text(29, self.log_y + 1, "► SYSTEM  READY ◄", attr);
    }

    /// Draws the "press any key" hint.
    fn draw_prompt(&mut self) {
        self.canvas
            .text(28, self.log_y + 3, "press any key to begin", palette::DIM);
    }
}

/// Whether logo cell (row `y`, column `x`) is filled.
fn is_logo(y: usize, x: usize) -> bool {
    LOGO[y][x] == b'#'
}

/// Canvas position of logo cell (row `y`, column `x`).
fn logo_pos(y: usize, x: usize) -> (u16, u16) {
    (LOGO_X + 2 * to_u16(x), LOGO_Y + to_u16(y))
}

/// A small count as a canvas coordinate.
fn to_u16(n: usize) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}
