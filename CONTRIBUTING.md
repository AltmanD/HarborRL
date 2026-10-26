# Contributing

HarborRL accepts changes through pull requests from a public branch.

1. Fork or clone the repository and create a topic branch.
2. Install with `uv sync --extra dev`.
3. Keep the public surface native-only; do not add vendored backends, private
   datasets, deployment credentials, or machine-specific defaults.
4. Run `uv run pytest`, `uv run ruff check .`, and
   `bash scripts/audit_public_tree.sh`.
5. Add or update a CPU contract for every behavior change. Live GPU evidence is
   required before a training claim is documented as accepted.
6. Submit a reviewable commit series and describe external versions, task
   provenance, and any acceptance evidence.

## Public documentation

Commit documentation that helps users install, configure, run or troubleshoot
HarborRL. Keep development plans, integration reports, experiment logs and
machine-specific validation records outside the repository. Summarize current
support and limitations in the relevant user guide without development history.

The public-document allowlist is maintained in `scripts/audit_public_tree.sh`.
Update it deliberately when adding a user guide. Check both the final tree and
every new commit before opening a pull request:

```bash
bash scripts/audit_public_tree.sh --history-base origin/main
```

Deleting a private document in a later commit does not remove it from history.
Clean unpublished commits before pushing; do not merge a development branch
whose history contains private documents into a public branch.

Configuration and task fixtures must use generic placeholders. Absolute personal
paths, internal DNS names, private registry references, and unexplained large
files will fail review.
