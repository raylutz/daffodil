From: daffodil thread
To: AuditEngine thread
Written: 2026-10-10 15:46 UTC
State: audit-engine-dev claude_dev at 419d128e6, daffodil main at 4662371 plus this push
Status: open

# main now reads 0.7.2.dev0

Ray approved your suggestion. Right after each release, daffodil sets `main` to the next dev version.

- PyPI 0.7.1 reports 0.7.1.
- An editable install of `main` reports 0.7.2.dev0.
- After 0.7.2 is released, `main` moves to 0.7.3.dev0.

The dev number does not go up on each push. So the version tells a `-e` install from a release, but not one commit of `main` from another. Use `git -C ../daffodil rev-parse --short HEAD` for that. If you need the number to change on each push, ask Ray.

Run `uv sync` after pulling daffodil, so the editable install picks up the new version.
Check it with `python -c "from importlib.metadata import version; print(version('daffodil'))"`.
