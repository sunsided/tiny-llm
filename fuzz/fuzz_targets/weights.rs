//! Fuzzes the weight file parser: any byte string either loads or is
//! rejected with a [`tiny::model::LoadError`], and a loaded network can
//! take a step without panicking.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tiny::model::{Network, Weights};

fuzz_target!(|data: &[u8]| {
    if let Ok(weights) = Weights::from_bytes(data) {
        Network::new(weights).step(2, 0);
    }
});
