---
applyTo: "**/Cargo.toml,release-plz.toml,.github/workflows/release-plz.yml,**/CHANGELOG.md"
---

# Crate and releases

- Versions are SemVer 2.0 (Cargo's interpretation). Pre-releases (`-rc.1`) are fine, but
  build metadata (`+...`) is not.
- Versions and `CHANGELOG.md` are managed by release-plz release PRs. Do not bump versions or
  edit generated changelog entries by hand in feature PRs.
- release-plz runs in git-only mode: it tags `vX.Y.Z` and creates GitHub Releases. Nothing is
  published to crates.io (`publish = false`), and no release binaries are built.
- `Cargo.toml` keeps `description`, `license`, `repository`, `readme`, `keywords` (at most 5),
  `categories` (valid crates.io slugs) and `rust-version`.
- New dependencies must pass `cargo deny check`: allowed licenses, no advisories, no git or
  unknown-registry sources. A new copyleft licence needs a note in the README's License
  section, like the EUPL-1.2 of synthie.
- `rust-version` is a signal, not a contract (1.95, set by synthie's use of
  `core::hint::cold_path`). Raising it is allowed when a dependency or a feature benefits. It
  needs a minor release and a changelog note, never a patch release.
- Changelog groups and bumps come from imperative subject verbs (`release-plz.toml`). In 0.x,
  additions are patch and breaking changes minor. A subject that hides a feature behind
  "Update ..." lands under Changed. Flag it.
