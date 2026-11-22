//! The NEURAL MONITOR panel: what the network is doing, live.

use std::collections::VecDeque;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Widget;

use super::palette::{self, Attr, Scheme};
use tiny::model::{Config, Output, count};
use tiny::think::Creativity;

/// Number of recent letters in the LAST LETTERS strip.
const HEAT_CELLS: usize = 18;

/// At most this many layers get an activity bar.
const MAX_BARS: usize = 6;

/// Live statistics shown in the monitor.
#[derive(Debug, Clone)]
pub struct Monitor {
    /// Shape of the network, for the NETWORK line.
    config: Config,
    /// Latest mean activity of each layer.
    activity: Vec<f32>,
    /// Slowly decaying peak activity of each layer, to scale the bars.
    peak: Vec<f32>,
    /// Confidence of the latest letter, 0..=1.
    confidence: f32,
    /// Confidence of the latest letters, oldest first, at most [`HEAT_CELLS`].
    heat: VecDeque<f32>,
    /// Network speed of the latest answer in letters per second.
    speed: Option<f64>,
}

impl Monitor {
    /// A quiet monitor for a network of shape `config`.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self {
            config,
            activity: vec![0.0; config.layers],
            peak: vec![0.0; config.layers],
            confidence: 0.0,
            heat: VecDeque::with_capacity(HEAT_CELLS),
            speed: None,
        }
    }

    /// Takes in the layer activity of a network step.
    pub fn observe_step(&mut self, output: &Output) {
        let layers = self.activity.iter_mut().zip(&mut self.peak);
        for ((activity, peak), &now) in layers.zip(&output.layer_activity) {
            *activity = now;
            *peak = peak.max(now) * 0.995;
        }
    }

    /// Takes in the confidence of a chosen letter.
    pub fn observe_letter(&mut self, confidence: f32) {
        self.confidence = confidence;
        if self.heat.len() == HEAT_CELLS {
            self.heat.pop_front();
        }

        self.heat.push_back(confidence);
    }

    /// Starts a new answer: the letter strip empties.
    pub fn start_answer(&mut self) {
        self.heat.clear();
    }

    /// Records how fast the network produced the latest answer.
    pub fn set_speed(&mut self, letters_per_second: f64) {
        self.speed = Some(letters_per_second);
    }

    /// Draws the panel contents into `area`.
    pub fn render(&self, area: Rect, buf: &mut Buffer, scheme: Scheme, creativity: Creativity) {
        let label = |text: &'static str| Line::styled(text, scheme.style(palette::DIM));
        let text = |text: String| Line::styled(text, scheme.style(palette::TEXT));
        let mut lines = vec![label("LAYER ACTIVITY")];

        lines.extend(self.activity_lines(scheme));
        lines.push(Line::default());
        lines.push(label("CONFIDENCE"));
        lines.push(self.confidence_line(scheme));
        lines.push(label("LAST LETTERS"));
        lines.push(self.heat_line(scheme));
        lines.push(label("SPEED"));
        lines.push(text(
            self.speed
                .map_or("-".into(), |s| format!("{s:.0} letters/s")),
        ));
        lines.push(label("NETWORK"));
        lines.push(text(format!(
            "{} layers · {} wide",
            self.config.layers, self.config.width
        )));
        lines.push(Line::default());
        lines.push(label("MODE"));
        lines.push(text(format!("creativity {creativity}")));

        for (row, line) in (area.y..area.bottom()).zip(lines) {
            line.render(
                Rect {
                    y: row,
                    height: 1,
                    ..area
                },
                buf,
            );
        }
    }

    /// One bar per layer, scaled to its recent peak, colours alternating.
    fn activity_lines(&self, scheme: Scheme) -> impl Iterator<Item = Line<'static>> + '_ {
        let layers = self.activity.iter().zip(&self.peak).take(MAX_BARS);
        layers.enumerate().map(move |(l, (&now, &peak))| {
            let share = if peak > 0.0 { now / peak } else { 0.0 };
            let attr = if l % 2 == 0 {
                palette::ANSWER
            } else {
                palette::FRAME
            };
            let mut spans = vec![Span::styled(
                format!("L{:<2}", l + 1),
                scheme.style(palette::FRAME),
            )];
            spans.extend(meter(share, 15, attr, scheme));

            Line::from(spans)
        })
    }

    /// Confidence bar and percentage.
    fn confidence_line(&self, scheme: Scheme) -> Line<'static> {
        let mut spans = meter(self.confidence, 13, heat_attr(self.confidence), scheme);
        let percent = format!(" {:3.0}%", self.confidence * 100.0);
        spans.push(Span::styled(percent, scheme.style(palette::BRIGHT)));

        Line::from(spans)
    }

    /// The last letters' confidence as shaded blocks.
    fn heat_line(&self, scheme: Scheme) -> Line<'static> {
        let shades = self.heat.iter().map(|&c| {
            let glyph = match c {
                c if c > 0.9 => "█",
                c if c > 0.7 => "▓",
                c if c > 0.45 => "▒",
                _ => "░",
            };
            Span::styled(glyph, scheme.style(heat_attr(c)))
        });
        let empty =
            (self.heat.len()..HEAT_CELLS).map(|_| Span::styled("·", scheme.style(palette::DIM)));

        Line::from(shades.chain(empty).collect::<Vec<_>>())
    }
}

/// Green when sure, amber when unsure, red when guessing.
fn heat_attr(confidence: f32) -> Attr {
    match confidence {
        c if c > 0.8 => palette::ANSWER,
        c if c > 0.5 => palette::QUESTION,
        _ => palette::HOT,
    }
}

/// A bar `width` cells long filled to `share` (0..=1) in half-cell steps.
fn meter(share: f32, width: usize, attr: Attr, scheme: Scheme) -> Vec<Span<'static>> {
    let total = 2 * width;
    let filled = share * count(total);
    let halves = (0..=total).rev().find(|&n| filled >= count(n)).unwrap_or(0);

    (0..width)
        .map(|i| match halves {
            n if n >= 2 * i + 2 => Span::styled("█", scheme.style(attr)),
            n if n == 2 * i + 1 => Span::styled("▌", scheme.style(attr)),
            _ => Span::styled("·", scheme.style(palette::DIM)),
        })
        .collect()
}
