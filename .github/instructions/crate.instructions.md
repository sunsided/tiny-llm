---
applyTo: "**/Cargo.toml,deny.toml"
---

# Crate metadata and dependencies

- TINY is not released or published: `publish = false` stays, and there is no release
  automation. Do not add a changelog or release workflow without discussing it first.
- `Cargo.toml` keeps `description`, `license`, `repository`, `readme`, `keywords` (at most 5),
  `categories` (valid crates.io slugs) and `rust-version`.
- New dependencies must pass `cargo deny check`: allowed licenses, no advisories, no git or
  unknown-registry sources. A new copyleft licence needs a note in the README's License
  section, like the EUPL-1.2 of synthie.
- `rust-version` is a signal, not a contract (1.95, set by synthie's use of
  `core::hint::cold_path`). Raising it is fine when a dependency or a feature benefits; keep
  the README's MSRV section and badge in step.
