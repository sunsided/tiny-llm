//! Sound: PC-speaker style square-wave beeps from a synthie oscillator and a robot voice
//! from synthie-trombone, mixed on one cpal output stream.
//!
//! ```mermaid
//! flowchart LR
//!     ui[UI thread] -- "Cmd (bounded channel)" --> engine
//!     subgraph audio thread
//!         engine[Engine] --> beeper["Beeper: square-wave tones"]
//!         engine --> talker["Talker: robot voice"]
//!     end
//!     engine -- "words finished (atomic)" --> ui
//! ```
//!
//! Without an audio device everything here quietly does nothing.

pub mod effects;
pub mod engine;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::SyncSender;

use synthie_trombone::speech::Utterance;
use synthie_trombone::{Preset, VoiceSettings};

use engine::Cmd;

/// One beep: a pitch glide over a duration. A pitch of 0 is silence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tone {
    /// Start pitch in Hz.
    pub from: f32,
    /// End pitch in Hz.
    pub to: f32,
    /// Length in milliseconds.
    pub ms: f32,
}

impl Tone {
    /// A steady square beep of `hz` for `ms` milliseconds.
    #[must_use]
    pub fn beep(hz: f32, ms: f32) -> Self {
        Self::sweep(hz, hz, ms)
    }

    /// A square beep gliding from `from` to `to` Hz.
    #[must_use]
    pub fn sweep(from: f32, to: f32, ms: f32) -> Self {
        Self { from, to, ms }
    }
}

/// The sound output and its switches.
pub struct Sound {
    /// Open stream and command channel, `None` without an audio device.
    link: Option<Link>,
    /// F2: all sound on or off.
    pub enabled: bool,
    /// F3: the robot voice on or off.
    pub voice: bool,
    /// Words handed to the voice so far.
    words_sent: usize,
    /// Value of `words_sent` at the last [`Sound::stop_voice`]; words up to it count as done.
    hushed: usize,
    /// Robot voice settings, for compiling words on this thread.
    settings: VoiceSettings,
}

/// The running audio stream and the way to talk to it.
struct Link {
    /// Keeps the stream playing; dropping it stops sound. `None` in offline tests.
    _stream: Option<cpal::Stream>,
    /// Commands for the audio thread.
    commands: SyncSender<Cmd>,
    /// Words the audio thread finished speaking.
    words_done: Arc<AtomicUsize>,
}

impl std::fmt::Debug for Sound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sound")
            .field("available", &self.link.is_some())
            .field("enabled", &self.enabled)
            .field("voice", &self.voice)
            .finish_non_exhaustive()
    }
}

impl Sound {
    /// Opens the default audio output. Falls back to [`Sound::silent`] when that fails.
    #[must_use]
    pub fn open(enabled: bool) -> Self {
        let settings = robot();
        let link = engine::start(&settings)
            .ok()
            .map(|(stream, commands, words_done)| Link {
                _stream: Some(stream),
                commands,
                words_done,
            });

        Self {
            link,
            enabled,
            voice: true,
            words_sent: 0,
            hushed: 0,
            settings,
        }
    }

    /// Sound with no device: the test renders the returned engine itself.
    #[cfg(test)]
    #[must_use]
    pub fn offline() -> (Self, engine::Engine) {
        let settings = robot();
        let (engine, commands, words_done) = engine::Engine::new(16_000.0, &settings);
        let link = Link {
            _stream: None,
            commands,
            words_done,
        };
        let sound = Self {
            link: Some(link),
            enabled: true,
            voice: true,
            words_sent: 0,
            hushed: 0,
            settings,
        };

        (sound, engine)
    }

    /// No audio at all, for tests.
    #[cfg(test)]
    #[must_use]
    pub fn silent() -> Self {
        Self {
            link: None,
            enabled: false,
            voice: true,
            words_sent: 0,
            hushed: 0,
            settings: robot(),
        }
    }

    /// Whether an audio device is open.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.link.is_some()
    }

    /// Whether answers should be spoken.
    #[must_use]
    pub fn speaks(&self) -> bool {
        self.enabled && self.voice && self.is_available()
    }

    /// Queues tones after any still playing. Ignored while sound is off.
    pub fn play(&self, tones: &[Tone]) {
        for &tone in tones {
            self.send(Cmd::Tone(tone));
        }
    }

    /// Starts speaking `word`; see [`Sound::is_speaking`] for when it ends.
    pub fn say(&mut self, word: &str) {
        let utterance = Utterance::from_text(word, &self.settings.prosody());
        if self.speaks() && self.send(Cmd::Say(utterance)) {
            self.words_sent += 1;
        }
    }

    /// Whether a word from [`Sound::say`] is still being spoken.
    #[must_use]
    pub fn is_speaking(&self) -> bool {
        self.words_sent > self.hushed
            && self
                .link
                .as_ref()
                .is_some_and(|link| link.words_done.load(Ordering::Acquire) < self.words_sent)
    }

    /// Cuts the voice off mid-word.
    pub fn stop_voice(&mut self) {
        if let Some(link) = &self.link {
            link.commands.try_send(Cmd::Hush).ok();
        }

        self.hushed = self.words_sent;
    }

    /// Sends a command if sound is on; returns whether it was sent.
    fn send(&self, cmd: Cmd) -> bool {
        match &self.link {
            Some(link) if self.enabled => link.commands.try_send(cmd).is_ok(),
            _ => false,
        }
    }
}

/// The robot voice: monotone and pressed, a little quicker than the preset.
fn robot() -> VoiceSettings {
    Preset::Robot.settings().with_rate(1.15)
}
