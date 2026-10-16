#!/usr/bin/env bash
# Run the checks that CI runs, before a push to main. A push to main runs nothing on GitHub.
# The tests, the doctests and the Guide examples on Python 3.10, 3.11, 3.12 and 3.13, then the strict docs build.
# Like CI, each Python gets an environment from `uv sync` with the lock file. The environments
# are kept in ~/.cache/daffodil-check and reused. Stops at the first failure.
# Usage, from anywhere:   bash notes/scripts/check_before_push.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

DOCTEST_ARGS=(--doctest-modules src/daffodil --ignore=src/daffodil/lib/daf_pdf.py --ignore=src/daffodil/lib/md_demo.py)
ENV_ROOT="${HOME}/.cache/daffodil-check"
SITE_DIR="$(mktemp -d)"
trap 'rm -rf "$SITE_DIR"' EXIT

# Run one step and print its last line. On a failure, print the end of its output and stop.
# A step inside "$(...)" in an echo would not stop the script, since the echo succeeds.
run_step() {
    local label="$1"; shift
    local out
    if ! out=$("$@" 2>&1); then
        echo "$out" | tail -40
        echo "== $label: FAILED"
        exit 1
    fi
    echo "== $label: $(echo "$out" | tail -1)"
}

for py in 3.10 3.11 3.12 3.13; do
    export UV_PROJECT_ENVIRONMENT="$ENV_ROOT/py$py"
    uv sync -q --python "$py"
    run_step "Python $py: tests   " uv run --python "$py" pytest -q -p no:cacheprovider
    run_step "Python $py: doctests" uv run --python "$py" pytest -q -p no:cacheprovider "${DOCTEST_ARGS[@]}"
    run_step "Python $py: guide   " uv run --python "$py" pytest -q -p no:cacheprovider --doctest-glob='*.md' docsite/guide
done
unset UV_PROJECT_ENVIRONMENT

run_step "docs build, strict" uv run mkdocs build --strict -d "$SITE_DIR"

# Warn when the copy of the common rules differs from the master.
# The master is in the repo common. Where that repo is not on this machine, skip the check.
COMMON_DIR="${COMMON_DIR:-$HOME/common}"
if [ -f "$COMMON_DIR/copy_rules.sh" ]; then
    bash "$COMMON_DIR/copy_rules.sh" --check . || echo "== WARNING: run bash $COMMON_DIR/copy_rules.sh . and commit the copy."
fi
echo "All checks passed."
