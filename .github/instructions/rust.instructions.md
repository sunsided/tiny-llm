---
applyTo: "**/*.rs"
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment, private and `pub(crate)` items included. Docs state intent, invariants, units and
  error/panic conditions (`# Errors`, `# Panics`). They do not restate the name.
- No `mod.rs`: modules with children are `foo.rs` plus `foo/`. Integration tests live under
  `tests/it/`.
- No separator comments (`// -----`, `// =====`). If a file needs sections, split it into
  submodules.
- Leave a blank line after the closing `}` of a block statement when another statement
  follows, and between logical phases of a function.
- One level of abstraction per function. The UI's frame handling (`src/tui/app.rs`) and the
  boot sequence (`src/tui/boot.rs`) are short drivers over documented step functions; keep
  them that way.
- `clippy::pedantic` clean. Suppression only with a scoped
  `#[expect(lint, reason = "...")]`. Never `#[allow]`, and never crate-level suppression.
- `#![forbid(unsafe_code)]` in `src/lib.rs` and `src/main.rs`. Reject any `unsafe`.
- `src/model/` must keep rounding exactly like the original `TINY.C`: `f32` accumulation in
  index order, `f64` only where C calls `sqrt` or `exp`, and the `±0.5`-then-truncate
  quantisation of the key/value cache. `tests/it/parity.rs` compares greedy answers letter for
  letter; a change that breaks it is a bug, not a fixture update.
- The audio callback (`src/sound/engine.rs`) must not block. It drains a bounded channel with
  `try_recv` and must not allocate in the steady state; compile utterances on the UI thread.
- The UI renders DOS text attributes through `src/tui/palette.rs`. New colours become named
  attribute constants there, not ad-hoc `Color` values.
- Mermaid diagrams in doc comments must match the code they describe. Update them when the
  structure changes.
