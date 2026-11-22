//! The full-screen neon interface: boot sequence, conversation, neural
//! monitor, input line and status bar, redrawn every frame.

mod app;
mod boot;
mod chat;
mod monitor;
mod palette;

use std::io;
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event, KeyEventKind};
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph, Widget};
use ratatui::{DefaultTerminal, Frame};

use crate::sound::{Sound, effects};
use app::{App, Screen};
use palette::{Attr, Scheme};
use tiny::model::Network;
use tiny::think::Creativity;

/// Time between animation frames.
const FRAME: Duration = Duration::from_millis(15);

/// How long the goodbye sweep plays before the program ends.
const GOODBYE: Duration = Duration::from_millis(400);

/// Width of the neural monitor panel, frame included.
const MONITOR_WIDTH: u16 = 22;

/// How the interface starts.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Initial creativity, 0..=9.
    pub creativity: Creativity,
    /// Whether TINY asks itself sample questions.
    pub demo: bool,
    /// Colour scheme.
    pub scheme: Scheme,
    /// Start with all sound off.
    pub quiet: bool,
}

impl From<&crate::Cli> for Options {
    fn from(cli: &crate::Cli) -> Self {
        let scheme = match (cli.mono, cli.classic) {
            (true, _) => Scheme::Mono,
            (_, true) => Scheme::Classic,
            _ => Scheme::Neon,
        };

        Self {
            creativity: cli.creativity,
            demo: cli.demo,
            scheme,
            quiet: cli.quiet,
        }
    }
}

/// Runs the interface until the user quits. The terminal is restored on
/// return and on panic.
///
/// # Errors
///
/// Fails when the terminal cannot be drawn to or read from.
pub fn run(net: Network, options: Options) -> io::Result<()> {
    let sound = Sound::open(!options.quiet);
    let mut app = App::new(net, options, sound);
    let mut terminal = ratatui::init();
    let result = event_loop(&mut terminal, &mut app);
    ratatui::restore();

    if app.sound.enabled && app.sound.is_available() {
        app.sound.play(&effects::goodbye());
        std::thread::sleep(GOODBYE);
    }

    result
}

/// Draws, handles keys until the next frame is due, advances, repeats.
fn event_loop(terminal: &mut DefaultTerminal, app: &mut App) -> io::Result<()> {
    while !app.quit {
        terminal.draw(|frame| render(app, frame))?;
        let due = Instant::now() + FRAME;

        while let Some(left) = due.checked_duration_since(Instant::now()) {
            if !event::poll(left)? {
                break;
            }

            if let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                app.on_key(key);
            }
        }

        app.tick();
    }

    Ok(())
}

/// Draws the current screen.
fn render(app: &App, frame: &mut Frame) {
    let area = frame.area();
    frame
        .buffer_mut()
        .set_style(area, app.scheme.style(palette::TEXT));

    match &app.screen {
        Screen::Boot(boot) => boot.render(area, frame.buffer_mut(), app.scheme),
        Screen::Chat => render_chat(app, frame),
    }
}

/// Draws the chat screen: title, conversation, monitor, input, status.
fn render_chat(app: &App, frame: &mut Frame) {
    let [title, main, input, status] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(5),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    let [talk, monitor] =
        Layout::horizontal([Constraint::Min(20), Constraint::Length(MONITOR_WIDTH)]).areas(main);
    let scheme = app.scheme;

    title_bar(app, title, frame);
    let inner = panel(" CONVERSATION ", palette::FRAME, talk, frame, scheme);
    conversation(app, inner, frame);
    let inner = panel(" NEURAL MONITOR ", palette::FRAME2, monitor, frame, scheme);
    app.monitor
        .render(inner, frame.buffer_mut(), scheme, app.creativity);
    input_line(app, input, monitor.x, frame);
    status_bar(app, status, frame);
}

/// Draws a double-line frame with a tab-like title and returns its inside.
fn panel(title: &'static str, attr: Attr, area: Rect, frame: &mut Frame, scheme: Scheme) -> Rect {
    let block = Block::bordered()
        .border_type(BorderType::Double)
        .border_style(scheme.style(attr))
        .padding(Padding::horizontal(1))
        .title(Line::from(vec![
            Span::styled("╡", scheme.style(attr)),
            Span::styled(title, scheme.style(palette::PANEL)),
            Span::styled("╞", scheme.style(attr)),
        ]));
    let inner = block.inner(area);
    block.render(area, frame.buffer_mut());

    inner
}

/// Draws the newest conversation lines that fit, bottom-aligned.
fn conversation(app: &App, area: Rect, frame: &mut Frame) {
    let lines = app.chat.lines(usize::from(area.width), app.scheme);
    let skip = lines.len().saturating_sub(usize::from(area.height));

    Paragraph::new(lines.into_iter().skip(skip).collect::<Vec<_>>())
        .render(area, frame.buffer_mut());
}

/// Draws the top bar with the network name and size.
fn title_bar(app: &App, area: Rect, frame: &mut Frame) {
    let style = |attr| app.scheme.style(attr);
    let weights = thousands(app.net.config().parameter_count());
    let line = Line::from(vec![
        Span::styled(" ◄► ", style(palette::TITLE)),
        Span::styled("T I N Y", style(palette::TITLE2)),
        Span::styled(
            format!(" ·  NEURAL INTERFACE 2.1  ·  {weights} WEIGHTS   "),
            style(palette::TITLE),
        ),
        Span::styled("■", style(palette::LAMP)),
        Span::styled(" ONLINE", style(palette::TITLE)),
    ]);

    frame.buffer_mut().set_style(area, style(palette::TITLE));
    line.render(area, frame.buffer_mut());
}

/// Draws the ASK TINY box with the typed text and the key hints, and places the cursor.
fn input_line(app: &App, area: Rect, divider_x: u16, frame: &mut Frame) {
    let scheme = app.scheme;
    let inner = panel(" ASK TINY ", palette::FRAME2, area, frame, scheme);
    let buf = frame.buffer_mut();

    buf.set_string(divider_x, area.y, "╤", scheme.style(palette::FRAME2));
    buf.set_string(divider_x, inner.y, "│", scheme.style(palette::FRAME2));
    buf.set_string(
        divider_x,
        area.bottom() - 1,
        "╧",
        scheme.style(palette::FRAME2),
    );
    buf.set_string(
        divider_x + 2,
        inner.y,
        "↑ last  F1 help",
        scheme.style(palette::DIM),
    );
    buf.set_string(inner.x, inner.y, "►", scheme.style(palette::YOU));

    let room = usize::from(divider_x.saturating_sub(inner.x + 3));
    let shown = &app.input[app.input.len().saturating_sub(room)..];
    buf.set_string(inner.x + 2, inner.y, shown, scheme.style(palette::BRIGHT));

    if app.accepts_typing() {
        let x = inner.x + 2 + u16::try_from(shown.len()).unwrap_or(0);
        frame.set_cursor_position(Position::new(x, inner.y));
    }
}

/// Draws the bottom bar with key help and the latest timing.
fn status_bar(app: &App, area: Rect, frame: &mut Frame) {
    let style = |attr| app.scheme.style(attr);
    let line = Line::from(vec![
        Span::styled(" F1", style(palette::BAR_KEY)),
        Span::styled(" Help  ", style(palette::BAR)),
        Span::styled("F2", style(palette::BAR_KEY)),
        Span::styled(" Sound ", style(palette::BAR)),
        Span::styled(
            switch(app.sound.enabled && app.sound.is_available()),
            style(palette::BAR_VALUE),
        ),
        Span::styled("  F3", style(palette::BAR_KEY)),
        Span::styled(" Voice ", style(palette::BAR)),
        Span::styled(switch(app.sound.voice), style(palette::BAR_VALUE)),
        Span::styled("  F4", style(palette::BAR_KEY)),
        Span::styled(" Creat. ", style(palette::BAR)),
        Span::styled(app.creativity.to_string(), style(palette::BAR_VALUE)),
        Span::styled("  Esc", style(palette::BAR_KEY)),
        Span::styled(" Quit  ", style(palette::BAR)),
        Span::styled(app.status.clone(), style(palette::BAR)),
    ]);

    frame.buffer_mut().set_style(area, style(palette::BAR));
    line.render(area, frame.buffer_mut());
}

/// `ON` or `OFF`, padded to the same width.
fn switch(on: bool) -> &'static str {
    if on { "ON " } else { "OFF" }
}

/// `366528` → `"366,528"`.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }

        out.push(c);
    }

    out
}

#[cfg(test)]
mod tests {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::{KeyCode, KeyEvent};
    use tiny::model::Weights;

    use super::*;
    use crate::sound::engine::Engine;

    /// Types `text` and presses Enter.
    fn type_line(app: &mut App, text: &str) {
        for c in text.chars() {
            app.on_key(KeyEvent::from(KeyCode::Char(c)));
        }

        app.on_key(KeyEvent::from(KeyCode::Enter));
    }

    /// Renders `app` at 80 x 25 and returns the screen as text lines.
    fn screen(app: &App) -> String {
        let mut terminal = Terminal::new(TestBackend::new(80, 25)).expect("test terminal");
        terminal.draw(|frame| render(app, frame)).expect("draw");
        let buf = terminal.backend().buffer();

        (0..25)
            .map(|y| (0..80).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A fresh app at creativity 0.
    fn app(demo: bool) -> App {
        let net = Network::new(Weights::from_bytes(tiny::BUILTIN_MODEL).expect("model"));
        App::new(
            net,
            Options {
                creativity: 0,
                demo,
                scheme: Scheme::Neon,
                quiet: true,
            },
            Sound::silent(),
        )
    }

    /// The whole boot animation runs to its prompt without a key.
    #[test]
    fn boot_runs_to_the_end() {
        let mut app = app(false);
        for _ in 0..1000 {
            app.tick();
        }

        assert!(screen(&app).contains("press any key to begin"));
    }

    /// Demo mode boots, types, answers and moves on by itself.
    #[test]
    fn demo_plays_by_itself() {
        let mut app = app(true);
        let mut text = String::new();
        for _ in 0..3000 {
            app.tick();
            text = screen(&app);
            if text.contains("YOU » is a whale a fish") && text.contains("No, a whale is a mammal.")
            {
                break;
            }
        }

        assert!(
            text.contains("TINY » The capital of Estonia is Tallinn."),
            "{text}"
        );
        assert!(text.contains("No, a whale is a mammal."), "{text}");
    }

    /// The help panel fits the 80-column conversation.
    #[test]
    fn help_fits() {
        let mut app = app(false);
        app.on_key(KeyEvent::from(KeyCode::Char(' ')));
        app.on_key(KeyEvent::from(KeyCode::Char(' ')));
        app.on_key(KeyEvent::from(KeyCode::F(1)));

        let text = screen(&app);
        assert!(
            text.contains("F2 sound · F3 voice · F4 creativity"),
            "{text}"
        );
        assert!(
            text.contains("↑ last question · Esc quit · CLS clears"),
            "{text}"
        );
    }

    /// Past the boot screen with an offline voice, and the engine to render it.
    fn speaking_app() -> (App, Engine) {
        let (sound, engine) = Sound::offline();
        let net = Network::new(Weights::from_bytes(tiny::BUILTIN_MODEL).expect("model"));
        let options = Options {
            creativity: 0,
            demo: false,
            scheme: Scheme::Neon,
            quiet: false,
        };
        let mut app = App::new(net, options, sound);
        app.on_key(KeyEvent::from(KeyCode::Char(' ')));
        app.on_key(KeyEvent::from(KeyCode::Char(' ')));

        (app, engine)
    }

    /// The letters on screen with the "being spoken" highlight.
    fn highlighted(app: &App) -> String {
        let mut terminal = Terminal::new(TestBackend::new(80, 25)).expect("test terminal");
        terminal.draw(|frame| render(app, frame)).expect("draw");
        let speak = app.scheme.style(palette::SPEAK).bg;

        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .filter(|cell| Some(cell.bg) == speak)
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    /// Runs frames, with 15 ms of audio each, until `done` or a frame limit;
    /// returns every highlighted word seen, in order, without repeats.
    fn spoken_words(
        app: &mut App,
        engine: &mut Engine,
        done: impl Fn(&App) -> bool,
    ) -> Vec<String> {
        let mut audio = vec![0.0; 240];
        let mut seen: Vec<String> = Vec::new();
        for _ in 0..5000 {
            app.tick();
            engine.render(&mut audio, 1);
            let word = highlighted(app);
            if !word.is_empty() && seen.last() != Some(&word) {
                seen.push(word);
            }

            if done(app) {
                break;
            }
        }

        seen
    }

    /// The answer is read aloud word by word with a moving highlight, which
    /// disappears at the end.
    #[test]
    fn speaks_word_by_word() {
        let (mut app, mut engine) = speaking_app();
        type_line(&mut app, "what is the capital of france");
        let words = spoken_words(&mut app, &mut engine, |app| {
            app.accepts_typing() && !app.is_speaking() && app.status.contains("ltr")
        });

        assert_eq!(words, ["The", "capital", "of", "France", "is", "Paris."]);
        assert_eq!(highlighted(&app), "");
    }

    /// A key stops the voice and still types; the next answer is spoken in full.
    #[test]
    fn key_stops_voice_then_speaks_again() {
        let (mut app, mut engine) = speaking_app();
        type_line(&mut app, "what is the capital of france");
        let first = spoken_words(&mut app, &mut engine, |app| !highlighted(app).is_empty());
        assert_eq!(first, ["The"]);

        app.on_key(KeyEvent::from(KeyCode::Char('x')));
        assert_eq!(highlighted(&app), "");
        assert_eq!(app.input, "x");

        app.on_key(KeyEvent::from(KeyCode::Backspace));
        type_line(&mut app, "is a whale a fish");
        let words = spoken_words(&mut app, &mut engine, |app| {
            app.accepts_typing() && !app.is_speaking() && app.status.contains("ltr")
        });
        assert_eq!(words, ["No,", "a", "whale", "is", "a", "mammal."]);
    }

    /// Skipping boot, asking a question and letting it finish shows the
    /// greedy answer, the monitor and the timing.
    #[test]
    fn asks_and_answers_on_screen() {
        let mut app = app(false);

        app.on_key(KeyEvent::from(KeyCode::Char(' ')));
        assert!(screen(&app).contains("SYSTEM  READY"));
        app.on_key(KeyEvent::from(KeyCode::Char(' ')));
        type_line(&mut app, "what is the capital of peru");
        for _ in 0..200 {
            app.tick();
        }

        let text = screen(&app);
        println!("{text}");
        assert!(text.contains("TINY » Lima."), "{text}");
        assert!(text.contains("LAYER ACTIVITY"));
        assert!(text.contains("5 ltr"));
    }
}
