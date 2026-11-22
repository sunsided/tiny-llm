//! TINY 2.1: a 367,000-weight transformer that answers questions letter by
//! letter, ported from the Turbo C original to Rust.
//!
//! - [`model`] loads `TINY.BIN` and runs one letter per step.
//! - [`think`] turns steps into question reading and answer writing.
//! - [`text`] maps letters to tokens and tidies typed questions.

#![forbid(unsafe_code)]

pub mod model;
pub mod text;
pub mod think;

/// The released network, built into the program so it runs anywhere.
pub const BUILTIN_MODEL: &[u8] = include_bytes!("../assets/TINY.BIN");
