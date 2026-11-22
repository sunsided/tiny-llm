//! The audio thread: drains commands, then mixes beeps and voice per sample.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use synthie::audio::osc::Oscillator;
use synthie::params::Waveform;
use synthie_trombone::speech::Utterance;
use synthie_trombone::{Talker, VoiceSettings};

use super::Tone;

/// Loudness of the beeps; a full-scale square wave is harsh.
const BEEP_LEVEL: f32 = 0.12;
/// Loudness of the voice.
const VOICE_LEVEL: f32 = 0.8;
/// Tones that can wait in the queue without the audio thread allocating.
const QUEUE: usize = 1024;

/// UI → audio commands.
#[derive(Debug)]
pub enum Cmd {
    /// Play a tone after the queued ones.
    Tone(Tone),
    /// Speak a compiled word after the current one.
    Say(Utterance),
    /// Stop speaking at once.
    Hush,
}

/// The sample generator owned by the audio callback.
pub struct Engine {
    /// Incoming commands.
    commands: Receiver<Cmd>,
    /// The beep queue.
    beeper: Beeper,
    /// The robot voice.
    talker: Talker,
    /// Words received.
    said: usize,
    /// Words finished or cut off, shared with the UI.
    done: Arc<AtomicUsize>,
}

impl Engine {
    /// An engine at `sample_rate` Hz, with its command sender and finished-word counter.
    pub fn new(
        sample_rate: f32,
        voice: &VoiceSettings,
    ) -> (Self, SyncSender<Cmd>, Arc<AtomicUsize>) {
        let (commands, receiver) = sync_channel(QUEUE);
        let done = Arc::new(AtomicUsize::new(0));
        let engine = Self {
            commands: receiver,
            beeper: Beeper {
                queue: VecDeque::with_capacity(QUEUE),
                current: None,
                osc: Oscillator::default(),
                sample_rate,
            },
            talker: Talker::new(sample_rate, *voice),
            said: 0,
            done: Arc::clone(&done),
        };

        (engine, commands, done)
    }

    /// Fills an interleaved buffer of `channels` channels.
    pub fn render(&mut self, data: &mut [f32], channels: usize) {
        while let Ok(cmd) = self.commands.try_recv() {
            self.apply(cmd);
        }

        for frame in data.chunks_mut(channels) {
            let voice = self.talker.process();
            if self.said > self.done.load(Ordering::Relaxed) && !self.talker.is_speaking() {
                self.done.store(self.said, Ordering::Release);
            }

            let mixed = self.beeper.next() * BEEP_LEVEL + voice * VOICE_LEVEL;
            frame.fill(mixed.clamp(-1.0, 1.0));
        }
    }

    /// Carries out one command.
    fn apply(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Tone(tone) => self.beeper.push(tone),
            Cmd::Say(utterance) => {
                self.talker.say_utterance(utterance);
                self.said += 1;
            }
            Cmd::Hush => {
                self.talker.interrupt();
                self.done.store(self.said, Ordering::Release);
            }
        }
    }
}

/// A PC speaker: one tone at a time from a queue.
struct Beeper {
    /// Waiting tones.
    queue: VecDeque<Tone>,
    /// The tone playing, its length and position in samples.
    current: Option<(Tone, f32, f32)>,
    /// synthie's oscillator, used as a square wave.
    osc: Oscillator,
    /// Output sample rate in Hz.
    sample_rate: f32,
}

impl Beeper {
    /// Queues a tone; drops it if the queue is full rather than allocating.
    fn push(&mut self, tone: Tone) {
        if self.queue.len() < QUEUE {
            self.queue.push_back(tone);
        }
    }

    /// Next sample, 0 when nothing plays.
    fn next(&mut self) -> f32 {
        if self.current.is_none() {
            self.current = self
                .queue
                .pop_front()
                .map(|tone| (tone, tone.ms * self.sample_rate / 1000.0, 0.0));
        }

        let Some((tone, length, pos)) = &mut self.current else {
            return 0.0;
        };

        let hz = tone.from + (tone.to - tone.from) * *pos / *length;
        let sample = if hz > 0.0 {
            self.osc
                .next_sample(hz, self.sample_rate, Waveform::Pulse, 0.5, 0.0)
        } else {
            0.0
        };

        *pos += 1.0;
        if *pos >= *length {
            self.current = None;
        }

        sample
    }
}

/// What [`start`] hands back: the stream to keep alive, the command sender,
/// and the counter of finished words.
type Started = (cpal::Stream, SyncSender<Cmd>, Arc<AtomicUsize>);

/// Opens the default output device and starts the engine.
///
/// # Errors
///
/// Fails when there is no output device or it rejects the stream.
pub fn start(voice: &VoiceSettings) -> Result<Started, Box<dyn std::error::Error>> {
    let device = cpal::default_host()
        .default_output_device()
        .ok_or("no audio output device")?;
    let config = device.default_output_config()?;
    let sample_rate = rate(config.sample_rate());
    let channels = usize::from(config.channels());

    let (mut engine, commands, done) = Engine::new(sample_rate, voice);

    let stream_config = cpal::StreamConfig {
        channels: config.channels(),
        sample_rate: config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };
    let stream = device.build_output_stream(
        stream_config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| engine.render(data, channels),
        |_| {},
        None,
    )?;
    stream.play()?;

    Ok((stream, commands, done))
}

/// A sample rate as `f32`.
#[expect(
    clippy::cast_precision_loss,
    reason = "sample rates are far below 2^24 and so exact"
)]
fn rate(hz: u32) -> f32 {
    hz as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use synthie_trombone::Preset;

    /// Sample rate of the offline tests (16 samples per millisecond).
    const RATE: f32 = 16_000.0;

    /// Renders `ms` milliseconds of mono audio and returns its peak level.
    fn peak(engine: &mut Engine, ms: usize) -> f32 {
        let mut buf = vec![0.0; ms * 16];
        engine.render(&mut buf, 1);
        buf.iter().fold(0.0f32, |m, s| m.max(s.abs()))
    }

    /// Nothing queued, nothing heard.
    #[test]
    fn silent_when_idle() {
        let (mut engine, _commands, _) = Engine::new(RATE, &Preset::Robot.settings());

        assert!(peak(&mut engine, 1000) < 1e-3);
    }

    /// A queued beep is audible and then the beeper falls silent.
    #[test]
    fn beeps() {
        let (mut engine, commands, _) = Engine::new(RATE, &Preset::Robot.settings());
        commands.send(Cmd::Tone(Tone::beep(440.0, 50.0))).unwrap();

        assert!((peak(&mut engine, 50) - BEEP_LEVEL).abs() < 1e-6);
        assert!(peak(&mut engine, 50) < f32::EPSILON);
    }

    /// Spoken words make sound and are counted once finished; answer-like
    /// words (numbers, spelling, punctuation) do not hang the counter.
    #[test]
    fn speaks_and_counts_words() {
        let voice = Preset::Robot.settings();
        let (mut engine, commands, done) = Engine::new(RATE, &voice);
        let words = ["Lima.", "56.", "200.", "E-L-E-P-H-A-N-T.", "XIV."];
        for word in words {
            let utterance = Utterance::from_text(word, &voice.prosody());
            assert!(!utterance.is_empty(), "{word} has no sound");
            commands
                .send(Cmd::Say(Utterance::from_text(word, &voice.prosody())))
                .unwrap();
        }

        assert!(peak(&mut engine, 1000) > 0.01);
        peak(&mut engine, 10_000);
        assert_eq!(done.load(Ordering::Acquire), words.len());
    }

    /// Hush counts every queued word as done at once.
    #[test]
    fn hush_finishes_everything() {
        let voice = Preset::Robot.settings();
        let (mut engine, commands, done) = Engine::new(RATE, &voice);
        commands
            .send(Cmd::Say(Utterance::from_text(
                "hello there",
                &voice.prosody(),
            )))
            .unwrap();
        peak(&mut engine, 100);
        commands.send(Cmd::Hush).unwrap();
        peak(&mut engine, 10);

        assert_eq!(done.load(Ordering::Acquire), 1);
    }
}
