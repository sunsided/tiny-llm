---
applyTo: ".github/workflows/**,Taskfile.dist.yaml,**/Taskfile.dist.yaml,.pre-commit-config.yaml"
---

# CI, Taskfile and hooks stay in sync

- The `ci` task in `Taskfile.dist.yaml` mirrors the jobs in `.github/workflows/ci.yml`
  (`branch`, `fmt`, `clippy`, `style`, `doc`, `test` on Linux, macOS and Windows, `coverage`,
  `deny`, `msrv`, `fuzz`). A change that adds, removes or renames a CI job, or changes its
  flags, must make the same change to `task ci` and the task it calls, and the other way round.
- `coverage` is left out of `task ci` (it only matters for the upload). The `security.yml`
  jobs (CodeQL, clippy and zizmor SARIF, SBOM submission) are CI-only and report rather than
  gate.
- Every Linux job that compiles the crate installs `libasound2-dev` first: cpal links ALSA.
  Flag a new compiling job without that step.
- The `msrv` job runs `cargo hack check --rust-version` without `--ignore-private`: the crate
  has `publish = false`, so that flag would skip it.
- Local hooks in `.pre-commit-config.yaml` call Taskfile tasks. They must not duplicate
  commands.
- Every `uses:` is pinned to a full 40-character commit SHA, with a `# vX.Y.Z` comment. Reject
  tag or branch refs.
- The workflow sets `permissions: {}` at the top, and each job sets the least privilege it
  needs. Every `actions/checkout` sets `persist-credentials: false`.
- No `pull_request_target`. No `${{ github.event.* }}` or other untrusted expressions
  directly in `run:`. Pass them through `env:`.
- `scripts/check-branch.sh` is the single definition of allowed branch names. It is used by
  the Taskfile, the hook and the CI `branch` job.
- New jobs that should gate merges must also be added to branch protection. Mention that in
  the PR.
