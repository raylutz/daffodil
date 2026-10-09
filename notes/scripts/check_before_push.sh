#!/usr/bin/env bash
# Run the checks that CI runs, before a push to main. A push to main runs nothing on GitHub.
# The tests and the doctests on Python 3.10, 3.11, 3.12 and 3.13, then the strict docs build.
# Like CI, each Python gets an environment from `uv sync` with the lock file. The environments
# are kept in ~/.cache/daffodil-check and reused. Stops at the first failure.
# Usage, from anywhere:   bash notes/scripts/check_before_push.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

DOCTEST_ARGS=(--doctest-modules src/daffodil --ignore=src/daffodil/lib/daf_pdf.py --ignore=src/daffodil/lib/md_demo.py)
ENV_ROOT="${HOME}/.cache/daffodil-check"
SITE_DIR="$(mktemp -d)"
trap 'rm -rf "$SITE_DIR"' EXIT

for py in 3.10 3.11 3.12 3.13; do
    export UV_PROJECT_ENVIRONMENT="$ENV_ROOT/py$py"
    uv sync -q --python "$py"
    echo "== Python $py: tests:    $(uv run --python "$py" pytest -q -p no:cacheprovider | tail -1)"
    echo "== Python $py: doctests: $(uv run --python "$py" pytest -q -p no:cacheprovider "${DOCTEST_ARGS[@]}" | tail -1)"
done
unset UV_PROJECT_ENVIRONMENT

echo "== docs build, strict: $(uv run mkdocs build --strict -d "$SITE_DIR" 2>&1 | grep -E 'WARNING|ERROR|Documentation built')"
echo "All checks passed."
