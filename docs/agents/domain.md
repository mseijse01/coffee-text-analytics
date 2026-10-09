# Domain Docs

Layout: **single-context**. One `CONTEXT.md` and one `docs/adr/` directory at the repo root.

## Consumer rules

Skills that need domain knowledge should:

1. Read `CONTEXT.md` at the repo root before making domain-level decisions. It defines the project's terms and key concepts.
2. Read relevant ADRs in `docs/adr/` (if the directory exists) before proposing changes that touch a past architectural decision.
3. Use the vocabulary from `CONTEXT.md` in code, issues, and docs rather than inventing synonyms.
4. If a change contradicts an ADR, say so explicitly and propose a new ADR rather than silently overriding it.

## ADR conventions

- Location: `docs/adr/`
- Filename: `NNNN-short-title.md` (zero-padded sequence number)
- Contents: context, decision, consequences
- ADRs are append-only; to reverse a decision, add a new ADR that supersedes the old one.
