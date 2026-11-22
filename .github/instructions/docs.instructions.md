---
applyTo: "**/*.md,docs/**"
---

# Documentation

- README files use absolute URLs for every link to a repo file and every image
  (`https://github.com/sunsided/tiny-llm/...`). Only in-page anchors may be relative.
  `task lint:readme` checks this.
- The README credits the original TINY 2.1 (https://github.com/WollyMonjac/Tiny-2.1) and its
  data licences. Keep both when editing.
- Mermaid diagrams (```` ```mermaid ````) must be valid (`task lint:mermaid` renders them)
  and must match the code. Flag diagrams that the change makes stale. Keep each diagram under
  about 15 nodes.
- Code blocks are tagged with their language (`sh`, `text`, `toml`, ...).
- Keep badges, the License section and the MSRV statement consistent with `Cargo.toml`.
