//! The application state and what keys and frames do to it.
//!
//! ```mermaid
//! stateDiagram-v2
//!     [*] --> Boot
//!     Boot --> Idle: key (or demo)
//!     Idle --> Thinking: Enter
//!     Thinking --> Idle: answer done
//!     Idle --> [*]: Esc / QUIT
//! ```

use std::ops::Range;
use std::time::Duration;
use std::time::Instant;

use rand::RngExt;
use rand::rngs::ThreadRng;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::Options;
use super::boot::Boot;
use super::chat::{Chat, Entry, TurnState};
use super::monitor::Monitor;
use super::palette::Scheme;
use crate::sound::{Sound, effects};
use tiny::model::Network;
use tiny::text::{MAX_QUESTION, clean, is_word};
use tiny::think::{Creativity, Event, Thinking};

/// Questions the demo mode asks itself.
const DEMO_QUESTIONS: [&str; 12] = [
    "what is the capital of estonia",
    "is a whale a fish",
    "what is 7 times 8",
    "which countries border estonia",
    "spell computer",
    "what is the symbol for gold",
    "how do you say thank you in spanish",
    "what is a group of lions called",
    "what comes after 199",
    "what time is it",
    "tell me a joke",
    "name the continents",
];

/// Frames to wait after a demo answer before the next question (about 1.5 s).
const DEMO_PAUSE: u32 = 100;

/// Which screen is showing.
#[derive(Debug)]
pub enum Screen {
    /// The boot sequence.
    Boot(Box<Boot>),
    /// The chat interface.
    Chat,
}

/// A question the network is working on.
#[derive(Debug)]
struct Active {
    /// The network's progress.
    thinking: Thinking,
    /// A letter chosen last frame, shown as a flicker, to be revealed this frame.
    pending: Option<char>,
    /// Letters written so far.
    letters: usize,
    /// Time spent inside the network (not animation), for the speed display.
    compute: Duration,
    /// Scanner animation counter.
    spin: usize,
}

/// Demo mode progress.
#[derive(Debug, Clone, Copy)]
enum Demo {
    /// Typing question `index`: `typed` letters shown, next one in `wait` frames.
    Typing {
        /// Index into [`DEMO_QUESTIONS`].
        index: usize,
        /// Letters typed so far.
        typed: usize,
        /// Frames until the next letter.
        wait: u32,
    },
    /// Question `index` was asked; once answered, wait `wait` frames.
    Asked {
        /// Index into [`DEMO_QUESTIONS`].
        index: usize,
        /// Frames left to wait after the answer.
        wait: u32,
    },
}

/// The answer being read aloud, one word at a time.
#[derive(Debug, Clone)]
struct Speech {
    /// Byte ranges of the answer's words, in order.
    words: Vec<Range<usize>>,
    /// Index of the next word to say.
    next: usize,
}

/// Everything the interface shows and remembers.
#[derive(Debug)]
pub struct App {
    /// The network.
    pub net: Network,
    /// Colour scheme.
    pub scheme: Scheme,
    /// Sampling creativity, 0..=9.
    pub creativity: Creativity,
    /// Current screen.
    pub screen: Screen,
    /// The conversation.
    pub chat: Chat,
    /// The neural monitor's statistics.
    pub monitor: Monitor,
    /// The line being typed.
    pub input: String,
    /// Right-hand text of the status bar.
    pub status: String,
    /// Set when the program should end.
    pub quit: bool,
    /// Beeps and the robot voice.
    pub sound: Sound,
    /// The last question asked, for the Up key.
    last_question: String,
    /// The question in progress, if any.
    active: Option<Active>,
    /// Demo mode progress, `None` when a human is typing.
    demo: Option<Demo>,
    /// The answer being spoken, if any.
    speech: Option<Speech>,
    /// Random numbers for sampling and effects.
    rng: ThreadRng,
}

impl App {
    /// A fresh interface at the start of the boot sequence.
    #[must_use]
    pub fn new(net: Network, options: Options, sound: Sound) -> Self {
        let config = net.config();

        Self {
            monitor: Monitor::new(config),
            screen: Screen::Boot(Box::new(Boot::new(config.layers, config.table_count()))),
            net,
            scheme: options.scheme,
            creativity: options.creativity,
            chat: Chat::default(),
            input: String::new(),
            status: String::new(),
            quit: false,
            sound,
            speech: None,
            last_question: String::new(),
            active: None,
            demo: options.demo.then_some(Demo::Typing {
                index: 0,
                typed: 0,
                wait: 30,
            }),
            rng: rand::rng(),
        }
    }

    /// Whether the cursor should show in the input line.
    #[must_use]
    pub fn accepts_typing(&self) -> bool {
        matches!(self.screen, Screen::Chat) && self.active.is_none() && self.demo.is_none()
    }

    /// Whether an answer is being read aloud.
    #[cfg(test)]
    #[must_use]
    pub fn is_speaking(&self) -> bool {
        self.speech.is_some()
    }

    /// Advances animations and the network by one frame.
    pub fn tick(&mut self) {
        match &mut self.screen {
            Screen::Boot(boot) => {
                boot.tick(&self.sound);
                if boot.is_waiting() && self.demo.is_some() {
                    self.enter_chat();
                }
            }
            Screen::Chat if self.active.is_some() => self.think_frame(),
            Screen::Chat if self.speech.is_some() => self.speech_frame(),
            Screen::Chat => self.demo_frame(),
        }
    }

    /// Reacts to a key press.
    pub fn on_key(&mut self, key: KeyEvent) {
        let ctrl_c =
            key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl_c {
            self.quit = true;
            return;
        }

        if self.speech.is_some() {
            self.hush();
        }

        match &mut self.screen {
            Screen::Boot(boot) if boot.is_waiting() => {
                if key.code == KeyCode::Esc {
                    self.quit = true;
                } else {
                    self.enter_chat();
                }
            }
            Screen::Boot(boot) => boot.skip(),
            Screen::Chat if key.code == KeyCode::Esc => self.quit = true,
            Screen::Chat if self.demo.is_some() => self.demo = None,
            Screen::Chat if self.active.is_some() => {}
            Screen::Chat => self.edit(key),
        }
    }

    /// Leaves the boot screen and greets the user.
    fn enter_chat(&mut self) {
        self.screen = Screen::Chat;
        self.sound.play(&effects::start());
        self.sound.play(&effects::ready());
        if !self.sound.is_available() {
            "no audio device".clone_into(&mut self.status);
        }

        self.chat.push(Entry::Note(
            "Hello! I am TINY, a neural network in your computer.",
        ));
        self.chat.push(Entry::Note(
            "Ask about countries, animals, space, words or sums.",
        ));
    }

    /// Line editing and the function keys.
    fn edit(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                let line = std::mem::take(&mut self.input);
                self.submit(&line);
            }
            KeyCode::Backspace => {
                if self.input.pop().is_some() {
                    self.sound.play(&effects::click());
                }
            }
            KeyCode::Up => self.input.clone_from(&self.last_question),
            KeyCode::F(1) => self.chat.push(Entry::Help),
            KeyCode::F(2) => {
                self.sound.enabled = !self.sound.enabled;
                self.sound.play(&effects::click());
            }
            KeyCode::F(3) => {
                self.sound.voice = !self.sound.voice;
                self.sound.play(&effects::click());
            }
            KeyCode::F(4) => {
                self.creativity = (self.creativity + 1) % 10;
                self.sound.play(&effects::click());
            }
            KeyCode::Char(c) if c.is_ascii() && !c.is_ascii_control() => self.type_letter(c),
            _ => {}
        }
    }

    /// Adds a typed letter, or buzzes when the line is full.
    fn type_letter(&mut self, c: char) {
        if self.input.len() < MAX_QUESTION {
            self.input.push(c);
            self.sound.play(&effects::click());
        } else {
            self.sound.play(&effects::error());
        }
    }

    /// Handles an entered line: a command word or a question.
    fn submit(&mut self, line: &str) {
        let q = clean(line);
        if q.is_empty() {
            return;
        }

        if is_word(&q, "quit") || is_word(&q, "exit") {
            self.quit = true;
        } else if is_word(&q, "cls") {
            self.chat.clear();
        } else if is_word(&q, "help") {
            self.chat.push(Entry::Help);
        } else {
            self.last_question.clone_from(&q);
            self.ask(q);
        }
    }

    /// Starts answering `question`.
    fn ask(&mut self, question: String) {
        self.active = Some(Active {
            thinking: Thinking::new(&question),
            pending: None,
            letters: 0,
            compute: Duration::ZERO,
            spin: 0,
        });
        self.chat.push(Entry::Turn(super::chat::Turn {
            question,
            answer: String::new(),
            state: TurnState::Reading {
                percent: 0,
                spin: 0,
            },
            spoken: None,
        }));
        self.monitor.start_answer();
        "thinking...".clone_into(&mut self.status);
    }

    /// One frame of answering: reveal a pending letter, or run one network step.
    fn think_frame(&mut self) {
        let (Some(active), Some(turn)) = (self.active.as_mut(), self.chat.last_turn()) else {
            return;
        };

        if let Some(letter) = active.pending.take() {
            turn.answer.push(letter);
            turn.state = TurnState::Writing { flicker: None };
            return;
        }

        let started = Instant::now();
        let event = active
            .thinking
            .advance(&mut self.net, self.creativity, &mut self.rng);
        active.compute += started.elapsed();
        self.monitor.observe_step(self.net.output());

        match event {
            Event::Read { index, total } => {
                if index.is_multiple_of(3) {
                    self.sound.play(&effects::scan(self.rng.random()));
                }

                active.spin += 1;
                turn.state = TurnState::Reading {
                    percent: 100 * (index + 1) / total,
                    spin: active.spin,
                };
            }
            Event::Letter { letter, confidence } => {
                self.monitor.observe_letter(confidence);
                active.letters += 1;
                if !self.sound.speaks() {
                    self.sound.play(&effects::letter(letter));
                }

                let flicker = (letter != ' ').then(|| char::from(self.rng.random_range(33..123u8)));
                active.pending = Some(letter);
                turn.state = TurnState::Writing { flicker };
            }
            Event::Done => self.finish(),
        }
    }

    /// Wraps up the finished answer.
    fn finish(&mut self) {
        let Some(active) = self.active.take() else {
            return;
        };
        let Some(turn) = self.chat.last_turn() else {
            return;
        };

        if turn.answer.is_empty() {
            turn.answer.push('.');
        }

        turn.state = TurnState::Done;
        let seconds = active.compute.as_secs_f64();
        if seconds > 0.0 {
            self.monitor
                .set_speed(f64::from(u32::try_from(active.letters).unwrap_or(u32::MAX)) / seconds);
        }

        self.status = format!("{} ltr {:.3}s", active.letters, seconds);
        if self.sound.speaks() {
            self.sound.play(&effects::speak());
            self.speech = Some(Speech {
                words: words(&turn.answer),
                next: 0,
            });
        } else {
            self.sound.play(&effects::ready());
        }

        if let Some(Demo::Asked { index, .. }) = self.demo {
            self.demo = Some(Demo::Asked {
                index,
                wait: DEMO_PAUSE,
            });
        }
    }

    /// Once the previous word is spoken, highlights and says the next one.
    fn speech_frame(&mut self) {
        if self.sound.is_speaking() {
            return;
        }

        let (Some(speech), Some(turn)) = (self.speech.as_mut(), self.chat.last_turn()) else {
            return;
        };

        if let Some(word) = speech.words.get(speech.next).cloned() {
            self.sound.say(&turn.answer[word.clone()]);
            turn.spoken = Some(word);
            speech.next += 1;
        } else {
            turn.spoken = None;
            self.speech = None;
        }
    }

    /// Stops the voice and the word highlight (any key does this).
    fn hush(&mut self) {
        self.sound.stop_voice();
        self.speech = None;
        if let Some(turn) = self.chat.last_turn() {
            turn.spoken = None;
        }
    }

    /// One frame of demo mode: type, ask, pause, next question.
    fn demo_frame(&mut self) {
        let Some(demo) = self.demo else {
            return;
        };

        self.demo = match demo {
            Demo::Typing { index, typed, wait } if wait > 0 => Some(Demo::Typing {
                index,
                typed,
                wait: wait - 1,
            }),
            Demo::Typing { index, typed, .. } => Some(self.demo_type(index, typed)),
            Demo::Asked { index, wait } if wait > 0 => Some(Demo::Asked {
                index,
                wait: wait - 1,
            }),
            Demo::Asked { index, .. } => {
                (index + 1 < DEMO_QUESTIONS.len()).then_some(Demo::Typing {
                    index: index + 1,
                    typed: 0,
                    wait: 10,
                })
            }
        };
    }

    /// Types the next demo letter, or asks the question once it is complete.
    fn demo_type(&mut self, index: usize, typed: usize) -> Demo {
        let question = DEMO_QUESTIONS[index];
        if typed < question.len() {
            question[..=typed].clone_into(&mut self.input);
            self.sound.play(&effects::click());
            let wait = self.rng.random_range(3..7);
            return Demo::Typing {
                index,
                typed: typed + 1,
                wait,
            };
        }

        self.input.clear();
        self.ask(question.to_owned());

        Demo::Asked { index, wait: 0 }
    }
}

/// Byte ranges of the space-separated words of `text`.
fn words(text: &str) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices().chain([(text.len(), ' ')]) {
        match (c == ' ', start) {
            (true, Some(s)) => {
                out.push(s..i);
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::words;

    /// Words are found between any number of spaces.
    #[test]
    fn splits_words() {
        assert_eq!(words(" No,  a whale."), vec![1..4, 6..7, 8..14]);
        assert_eq!(words(""), Vec::<std::ops::Range<usize>>::new());
    }
}
