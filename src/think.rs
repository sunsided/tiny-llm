//! Answering a question one step at a time, so a UI can draw between steps.
//!
//! ```mermaid
//! stateDiagram-v2
//!     [*] --> Reading
//!     Reading --> Reading: Read (one question letter)
//!     Reading --> Writing: newline fed
//!     Writing --> Writing: Letter
//!     Writing --> [*]: Done (end token or context full)
//! ```

use rand::{Rng, RngExt};

use crate::model::{CTX, Network, safe_exp};
use crate::text::{NEWLINE, letter, token};

/// Sampling creativity: 0 always takes the best letter, 9 is the wildest.
pub type Creativity = u8;

/// Letters whose weight relative to the best one is below this are never picked.
const UNLIKELY: f32 = 0.15;

/// What one [`Thinking::advance`] did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// Read letter `index` (0-based) of a question with `total` letters.
    Read {
        /// Position of the letter just read.
        index: usize,
        /// Length of the question.
        total: usize,
    },
    /// Wrote one answer letter.
    Letter {
        /// The chosen letter.
        letter: char,
        /// Softmax probability of the chosen letter, 0..=1.
        confidence: f32,
    },
    /// The answer is complete; further calls keep returning this.
    Done,
}

/// The progress of one question through the network.
#[derive(Debug, Clone)]
pub struct Thinking {
    /// Question tokens, without the final newline.
    prompt: Vec<usize>,
    /// Next position to feed.
    pos: usize,
    /// Set once the end token was chosen or the context is full.
    done: bool,
}

impl Thinking {
    /// Starts on a question already passed through [`crate::text::clean`].
    ///
    /// # Panics
    ///
    /// Panics if `question` contains a non-printable byte.
    #[must_use]
    pub fn new(question: &str) -> Self {
        Self {
            prompt: question.bytes().map(token).collect(),
            pos: 0,
            done: false,
        }
    }

    /// Runs the network for one letter: reads the next question letter, or
    /// chooses and feeds the next answer letter.
    pub fn advance<R: Rng + ?Sized>(
        &mut self,
        net: &mut Network,
        creativity: Creativity,
        rng: &mut R,
    ) -> Event {
        let total = self.prompt.len();
        if self.done {
            return Event::Done;
        }

        if self.pos < total {
            net.step(self.prompt[self.pos], self.pos);
            self.pos += 1;
            return Event::Read {
                index: self.pos - 1,
                total,
            };
        }

        if self.pos == total {
            net.step(NEWLINE, self.pos);
            self.pos += 1;
        }

        self.write(net, creativity, rng)
    }

    /// Chooses the next answer letter and feeds it back.
    fn write<R: Rng + ?Sized>(
        &mut self,
        net: &mut Network,
        creativity: Creativity,
        rng: &mut R,
    ) -> Event {
        let chosen = (self.pos < CTX)
            .then(|| pick(&net.output().scores, creativity, rng))
            .and_then(|(tok, confidence)| Some((tok, letter(tok)?, confidence)));
        let Some((tok, letter, confidence)) = chosen else {
            self.done = true;
            return Event::Done;
        };

        net.step(tok, self.pos);
        self.pos += 1;

        Event::Letter { letter, confidence }
    }
}

/// Answers a whole question at once (plain mode and tests).
pub fn answer<R: Rng + ?Sized>(
    net: &mut Network,
    question: &str,
    creativity: Creativity,
    rng: &mut R,
) -> String {
    let mut thinking = Thinking::new(question);
    let mut out = String::new();

    loop {
        match thinking.advance(net, creativity, rng) {
            Event::Read { .. } => {}
            Event::Letter { letter, .. } => out.push(letter),
            Event::Done => return out,
        }
    }
}

/// Picks the next letter and its confidence.
///
/// Creativity 0 takes the best letter. Otherwise it samples at temperature
/// `creativity / 5` among the letters that fit nearly as well as the best.
fn pick<R: Rng + ?Sized>(scores: &[f32], creativity: Creativity, rng: &mut R) -> (usize, f32) {
    let best = best_index(scores);
    if creativity == 0 {
        return (best, confidence(scores, best));
    }

    let temperature = f32::from(creativity) / 5.0;
    let weights: Vec<f32> = scores
        .iter()
        .map(|s| safe_exp((s - scores[best]) / temperature))
        .map(|w| if w < UNLIKELY { 0.0 } else { w })
        .collect();
    let mut r = weights.iter().sum::<f32>() * rng.random::<f32>();

    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r < 0.0 && w > 0.0 {
            return (i, confidence(scores, i));
        }
    }

    (best, confidence(scores, best))
}

/// Index of the first highest score.
fn best_index(scores: &[f32]) -> usize {
    (1..scores.len()).fold(0, |best, i| if scores[i] > scores[best] { i } else { best })
}

/// Softmax probability of letter `tok`.
fn confidence(scores: &[f32], tok: usize) -> f32 {
    let sum: f32 = scores.iter().map(|s| safe_exp(s - scores[tok])).sum();
    1.0 / sum
}
