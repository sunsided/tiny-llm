//! The sound effects of `TINY.C`, as tone lists.

use super::Tone;

/// Rising arpeggio when the weights finished loading.
#[must_use]
pub fn boot() -> Vec<Tone> {
    let notes = [392.0, 523.0, 659.0, 784.0, 1047.0, 1319.0];
    let mut tones: Vec<Tone> = notes.iter().map(|&hz| Tone::beep(hz, 55.0)).collect();
    tones.push(Tone::beep(1568.0, 140.0));

    tones
}

/// Two-note chime: ready for a question.
#[must_use]
pub fn ready() -> [Tone; 2] {
    [Tone::beep(1319.0, 40.0), Tone::beep(1760.0, 70.0)]
}

/// Key click.
#[must_use]
pub fn click() -> [Tone; 1] {
    [Tone::beep(1800.0, 2.0)]
}

/// Low buzz: the line is full.
#[must_use]
pub fn error() -> [Tone; 1] {
    [Tone::beep(140.0, 120.0)]
}

/// Whoosh up into the chat screen.
#[must_use]
pub fn start() -> [Tone; 1] {
    [Tone::sweep(300.0, 1400.0, 120.0)]
}

/// Falling sweep on quitting.
#[must_use]
pub fn goodbye() -> [Tone; 1] {
    [Tone::sweep(1600.0, 200.0, 360.0)]
}

/// Short blip before the voice starts.
#[must_use]
pub fn speak() -> [Tone; 1] {
    [Tone::beep(1200.0, 8.0)]
}

/// Scanner tick while reading, at one of eight pitches.
#[must_use]
pub fn scan(step: u8) -> [Tone; 1] {
    [Tone::beep(2400.0 + f32::from(step % 8) * 150.0, 3.0)]
}

/// Letter tick when the voice is off; pitch follows the letter.
#[must_use]
pub fn letter(c: char) -> [Tone; 1] {
    let step = u8::try_from(u32::from(c) % 16).unwrap_or(0);
    [Tone::beep(900.0 + f32::from(step) * 70.0, 6.0)]
}

/// Rain static: a 2 ms blip at a random high pitch (`hz`, 3000..6000).
#[must_use]
pub fn rain(hz: f32) -> [Tone; 1] {
    [Tone::beep(hz, 2.0)]
}

/// Logo cell landing; pitch rises with the number of cells settled.
#[must_use]
pub fn settle(settled: u16) -> [Tone; 1] {
    [Tone::beep(200.0 + f32::from(settled) * 12.0, 3.0)]
}

/// A loading log line coming online.
#[must_use]
pub fn online(line: u16) -> [Tone; 1] {
    [Tone::beep(880.0 + 110.0 * f32::from(line % 6), 12.0)]
}
