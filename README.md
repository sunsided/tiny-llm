# TINY 2.1, in Rust

[![CI](https://github.com/sunsided/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/tiny-llm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/tiny-llm/graph/badge.svg)](https://codecov.io/gh/sunsided/tiny-llm)
[![MSRV](https://img.shields.io/badge/MSRV-1.95-blue)](https://github.com/sunsided/tiny-llm#msrv)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/sunsided/tiny-llm#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

A port of [TINY 2.1](https://github.com/WollyMonjac/Tiny-2.1) by WollyMonjac to Rust: a 367,000-weight transformer that answers questions letter by letter, with the neon full-screen interface, the neural monitor and the robot voice of the original, running in a terminal.

The original is a delightful piece of work: a question-answering transformer written in Turbo C 2.0 that runs on a 486DX under plain DOS, needs about 540 KB of conventional memory and fits on a floppy disk, PC-speaker voice included. This port is a joyful one, and it completely misses the point of the original (which was, after all, compiling it with DOS tooling for an actual DOS machine). It needs a 64-bit operating system, a Rust toolchain and about ninety crates instead, and it answers in milliseconds where the 486 took a few seconds (which, admittedly, takes some of the drama out of the neural monitor).

```text
 ◄► T I N Y ·  NEURAL INTERFACE 2.1  ·  366,528 WEIGHTS   ■ ONLINE
╔╡ CONVERSATION ╞════════════════════════════════════════╗╔╡ NEURAL MONITOR ╞══╗
║ Hello! I am TINY, a neural network in your computer.   ║║ LAYER ACTIVITY     ║
║ Ask about countries, animals, space, words or sums.    ║║ L1 ███████████████ ║
║                                                        ║║ L2 █████████████▌· ║
║  YOU » what is the capital of peru                     ║║ L3 ██████▌········ ║
║ TINY » Lima.                                           ║║ L4 ███████████████ ║
║                                                        ║║                    ║
║                                                        ║║ CONFIDENCE         ║
║                                                        ║║ ████████████▌  98% ║
╚════════════════════════════════════════════════════════╝╚════════════════════╝
╔╡ ASK TINY ╞═════════════════════════════════════════════╤════════════════════╗
║ ►                                                       │ ↑ last  F1 help    ║
╚═════════════════════════════════════════════════════════╧════════════════════╝
 F1 Help  F2 Sound ON   F3 Voice ON   F4 Creat. 0  Esc Quit  5 ltr 0.011s
```

## What carried over

The network is the original one: `crates/tiny/assets/TINY.BIN`, 4 layers, 96 wide, 8-bit weights with one scale per row, built into the binary. The inference code follows `TINY.C` closely enough to round the same way, down to the 8-bit key/value cache and the `±0.5`-then-truncate quantisation. At creativity 0 it therefore gives the same answers as the C program, letter for letter: `crates/tiny/tests/it/parity.rs` checks 44 questions against answers from a gcc build of `TINY.C` in plain mode.

The interface is the DOS screen redrawn with [ratatui](https://ratatui.rs): the digital rain and the logo assembling at boot, the scanner while TINY reads, letters flickering into place, the layer activity and confidence meters, and the neon VGA palette as 24-bit colour. The original's robot voice made each letter a buzz, hiss or click on the PC speaker, since real speech didn't fit into 640 KB next to the network. Here the beeps come from [synthie](https://crates.io/crates/synthie) and the voice from [synthie-trombone](https://crates.io/crates/synthie-trombone), a Pink Trombone vocal tract model, so TINY actually speaks its answers, one highlighted word at a time.

TINY knows about 3,500 facts (countries, animals, space, the body, simple sums, spelling, a few jokes) and confidently makes things up outside of them; ask it who the king of France is.

## Training it yourself

The repository now doubles as a playground for ML training in Rust. `crates/tiny-train` ports the original training kit: `gen2.py`, which turns the fact files into question/answer pairs in many wordings, and `train2.py`, the PyTorch trainer. The generator brings its own copy of Python's `random.Random` (the Mersenne Twister plus the handful of helper methods `gen2.py` calls), so the same seed gives the same questions as the original, typos included; tests compare 20,000 samples line by line. Exporting the original `tiny2.pt` gives the released `TINY.BIN` byte for byte, and the released network scores 61.3% on the original's 780 held-back test questions, against the 61% its README reports.

The trainer runs on [burn](https://burn.dev). It started out on candle, until it turned out that candle's `softmax_last_dim` has no backward pass, so the attention never learned a thing (there is a test for every parameter's gradient now). On a laptop with an i7-13700H and an RTX 3050, a step of 64 examples takes about 0.33 s on the CPU and 0.21 s with CUDA, where the original PyTorch kit managed roughly 0.09 s on two cores. Fine-tuning is fine; training from scratch is a job for a night or two.

```sh
cargo run --release -p tiny-train -- gen          # sample questions of every kind
cargo run --release -p tiny-train -- train 60 more.safetensors \
    --resume crates/tiny-train/data/tiny2.pt --lr 5e-4 --decay 0.3 --eval-every 20
cargo run --release -p tiny-train -- eval more.safetensors
cargo run --release -p tiny-train -- export more.safetensors TINY.BIN
cargo run --release -- --model TINY.BIN
```

That is the original's fine-tuning recipe. In the one 60-minute run I made with it, the score went down to 57.8% rather than up, so take the learning rate as a starting point. For the GPU, build with `--features cuda` (or `wgpu`) and train with `--device cuda` (or `wgpu`); the first minute or two go into compiling kernels. New facts go into `crates/tiny-train/data/facts/` in the original's format: `question | question => answer | answer` per line in the `.txt` files, `relation<TAB>subject<TAB>answer` in the `.tsv` files.

## Running it

On Linux, sound needs the ALSA headers (`libasound2-dev` on Debian and Ubuntu):

```sh
cargo run --release              # creativity 5, the default
cargo run --release -- 0         # always the surest answer
cargo run --release -- --demo    # TINY asks itself sample questions
```

| Key | Action |
|---|---|
| Enter | ask |
| ↑ | last question |
| F1 | help |
| F2 / F3 | sound / robot voice on or off |
| F4 | creativity 0 to 9 |
| any key | stops the voice |
| Esc | quit |

Typing `HELP`, `CLS` or `QUIT` works too. Options:

| Option | Effect |
|---|---|
| `0` … `9` | creativity: 0 takes the best letter, higher values sample more freely |
| `--demo` | demo mode; a key takes over |
| `-q`, `--quiet` | no sound |
| `--mono`, `--classic` | grey or standard terminal colours instead of neon |
| `-p`, `--plain` | plain text, also used when input or output is redirected |
| `--model FILE` | another `TINY.BIN` (version 2 format) |

Plain mode reads one question per line, so `tiny 0 < questions.txt` works like `TINY < QUESTIONS.TXT` did.

## MSRV

Rust 1.95, because synthie uses `core::hint::cold_path`.

## License

MIT, keeping the copyright of the original (see `LICENSE-MIT`). The network in `crates/tiny/assets/TINY.BIN` was trained partly on data from [mledoze/countries](https://github.com/mledoze/countries) (ODbL 1.0) and [Periodic-Table-JSON](https://github.com/Bowserinator/Periodic-Table-JSON) (CC BY-SA 3.0); see `crates/tiny/assets/DATA_LICENSES.md`. The training data in `crates/tiny-train/data/facts/` keeps the same licences: `geo2.tsv` and `geo3.tsv` are ODbL 1.0, `elements.tsv` is CC BY-SA 3.0. The sound crates synthie and synthie-trombone are EUPL-1.2, so a built binary contains EUPL code.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work by you shall be licensed as above, without any additional terms or conditions.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/tiny-llm/blob/main/CONTRIBUTING.md).
