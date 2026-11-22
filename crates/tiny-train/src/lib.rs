//! Training kit for TINY: loading checkpoints, exporting `TINY.BIN` files, and
//! (step by step) the question generator, trainer and evaluation of the
//! original Python kit.

#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod eval;
pub mod export;
