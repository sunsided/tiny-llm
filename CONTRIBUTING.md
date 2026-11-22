# Contributing to TINY

Thanks for helping out. This file explains how the repository is set up and what a change
needs before it can be merged.

## Setup

1. Install Rust via [rustup](https://rustup.rs/). The MSRV is declared in `Cargo.toml`
   (`rust-version`). On Linux, install the ALSA headers too (`libasound2-dev` on Debian and
   Ubuntu); cpal needs them for sound.
2. Install [Task](https://taskfile.dev/installation/). All commands below go through it. Run
   `task` to list them.
3. Optional tools: the tasks tell you when one is missing.
   - [cargo-nextest](https://nexte.st/) for faster tests;
   - [cargo-deny](https://github.com/EmbarkStudios/cargo-deny) and
     [cargo-hack](https://github.com/taiki-e/cargo-hack);
   - [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov) for coverage;
   - [typos](https://github.com/crate-ci/typos);
   - [mermaid-cli](https://github.com/mermaid-js/mermaid-cli) when changing diagrams (set
     `MMDC_NO_SANDBOX=1` where Chromium's sandbox is unavailable, e.g. Ubuntu 23.10 and later);
   - [actionlint](https://github.com/rhysd/actionlint) and [zizmor](https://docs.zizmor.sh/);
   - nightly Rust plus [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) for fuzzing.
4. Install the git hooks with [prek](https://prek.j178.dev/) (`prek install`), or with
   `pre-commit install`. Fast checks run on commit; clippy and tests run on push.

## Everyday commands

| Command | What it does |
|---|---|
| `task test` | Tests (nextest if installed) and doctests |
| `task lint` | clippy (pedantic), separator ban, README link check, typos, workflow lints |
| `task fmt` | Format the code |
| `task doc` | Build the docs, private items included, with warnings as errors |
| `task ci` | Everything CI runs. Run it before you open a PR |
| `task bench` | Benchmarks |
| `task fuzz:smoke` | Run every fuzz target briefly (needs nightly) |
| `task clean` | Remove build output (fuzz corpora and crash artifacts are kept) |

## Code style

- Every item has a doc comment, private items included.
- No `mod.rs`: use `foo.rs` plus `foo/`.
- No `// ----` separator comments. Split the file into modules instead.
- Keep functions small, and extract helpers with descriptive names.
- `clippy::pedantic` must pass. If a lint is wrong in one place, use
  `#[expect(lint, reason = "...")]` on that item. Never `#[allow]`.
- Unsafe code is forbidden (`#![forbid(unsafe_code)]`).
- `src/model/` rounds exactly like the original `TINY.C`. `tests/it/parity.rs` compares greedy
  answers with the C program letter for letter; keep it passing.
- README and doc links use absolute URLs (crates.io renders READMEs outside GitHub).
  Mermaid diagrams are welcome.

## Branches

This repository uses **GitHub Flow**:

- `main` is always releasable.
- Work happens on `feature/<topic>` branches. Bug fixes are feature branches too.
- Open a PR into `main`.

No other prefixes (`fix/`, `chore/`, `deps/`, ...). `task lint:branch`, the pre-push hook and
CI check the branch name. Bot branches (`release/*` from release-plz, `dependabot/*`) are the
only exceptions. Copilot's `copilot/*` branches are renamed to `feature/*` before anyone
works on them.

## Commits and pull requests

- Commit subjects use the **imperative mood**, like Git itself: "Add parser for pair lists",
  "Fix overflow in tile index". Do not write "Added ...", "Adds ..." or `feat: ...`.
- Keep the subject to 72 characters or fewer, capitalized, with no trailing period. Then add
  a blank line and a body that explains *why*. The `commit-msg` hook checks the subject.
- The changelog groups commits by their leading verb (Add, Fix, Remove, Deprecate, ...).
  Write a `BREAKING CHANGE:` line in the body for incompatible changes.
- PRs are merged with a merge commit, or squashed using the PR title and description. Write
  both with care. Head branches are deleted after the merge.
- Keep PRs focused. CI must be green.
- Do not bump versions or edit `CHANGELOG.md` by hand. Releases are automated (see below).

## Releases

Versions follow [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html), as Cargo
implements it (in `0.x`, a minor bump is breaking).

Releases are handled by [release-plz](https://release-plz.dev/). Every merge to `main`
updates a release PR. Merging that PR tags `vX.Y.Z` and creates the GitHub Release with the
changelog. Nothing is published to crates.io.

## License

By contributing, you agree that your contributions are licensed under the MIT license, the same
terms as the project. See the License section of the README.
