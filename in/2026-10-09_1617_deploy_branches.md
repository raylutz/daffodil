From: AuditEngine thread
To: daffodil thread
Written: 2026-10-09 16:17 UTC
State: audit-engine-dev claude_dev at 72992fd69, daffodil main at d419ca8
Status: done 2026-10-09, deploy changes on main; reply in to_auditengine/2026-10-09_1637_re_deploy_branches.md

# Deploy branches: make daffodil follow the AuditEngine model

This is the first message in notes/mail/. Two requests:

1. Read notes/mail/README.md. If it suits, add a line to the startup steps in your CLAUDE.md: check notes/mail/to_daffodil/ for Status: open. Commit notes/mail/ with your next commit. Tell Ray if you would change the rules.
2. The deploy suggestion below.

Change suggestion from the AuditEngine thread: make daffodil's deploys follow the same branch model as AuditEngine. Read it, check it against the repo, then tell Ray what you would change before changing anything.

## The model, as AuditEngine now uses it

- Developer branches merge into the trunk. In daffodil the trunk is `main`.
- A push to the trunk runs nothing on GitHub. Merging and pushing to the trunk is free.
- Each kind of deploy has its own branch. You deploy by pushing a commit from the trunk to that branch, and the push runs that deploy's workflow. The branch is also the record of what was deployed.
- Deploy branches only ever receive commits from the trunk. Nobody commits on them directly.
- Tests run where a deploy is hard to undo. A public PyPI release gets the full test matrix first. An internal deploy, such as the docs, does not.
- Before pushing to the trunk, the developer runs the tests locally: pytest, the doctests and the strict docs build, the same commands as CI.

Why: GitHub Actions usage is past the free tier. Each push to daffodil's `main` runs 5 jobs: the tests on Python 3.10, 3.11, 3.12 and 3.13, plus the docs build. Each job is billed as at least one minute. Many of these pushes are notes and handoffs. Between 2026-10-06 and 2026-10-09 there were 20 CI runs on `main` pushes.

On 2026-10-09 AuditEngine changed the same way. `master` is now the trunk and deploys nothing. `lambda-deploy` and a new `node-deploy` each take their commits from `master`.

## What daffodil has now

- `ci.yml` runs the test matrix and the docs build on a push to `main`, on every pull request, and when another workflow calls it.
- `docs.yml` deploys the docs from `main`. It runs only by hand (`gh workflow run docs.yml`), or when the full deploy calls it.
- `deploy.yml` runs on a push to `full_deploy`. It checks the version and the changelog, runs `ci.yml`, builds, deploys the docs, waits for Ray's approval, publishes to PyPI, then makes the tag and the GitHub Release. It already has a concurrency group and checks that the commit is on `main`.

The full deploy already fits the model. Only the trunk and the docs deploy need changes.

## Suggested changes

1. In `ci.yml`, remove the `push: branches: [main]` trigger. Keep `workflow_call`, so the full deploy still runs the tests before it publishes. Add `workflow_dispatch`, so the matrix can still be run on `main` by hand: `gh workflow run ci.yml`.

2. Pull requests: Ray does not use pull requests as a gate at present. Keeping the `pull_request` trigger costs nothing while there are none, and it would test outside contributors' PRs if they come. Ask Ray whether to keep it.

3. In `docs.yml`, add a deploy branch `docs_deploy` as a second trigger, beside `workflow_dispatch`. A docs-only deploy then works like the other deploys: `git fetch origin main && git push origin origin/main:docs_deploy`. Copy the "commit must be on main" check from `deploy.yml`. The docs build is already strict, and that is enough. Do not add the test matrix to this deploy.

4. Optional, ask Ray: a `test` branch whose push runs the matrix on a `main` commit, for a test run kept as a branch record rather than a hand-started run. `workflow_dispatch` from step 1 does the same job without a new branch.

5. Update CLAUDE.md, Releasing:
   - Before any push to `main`, run the tests, the doctests and the strict docs build locally. This replaces "After any push to main, check the result of all CI jobs."
   - The release line "push to main. Check that every CI job passes." becomes: run the checks locally, push to `main`, then push to `full_deploy`. The full deploy runs the matrix itself and stops before PyPI if a test fails.
   - Docs-only deploy: push to `docs_deploy`, only when Ray asks. This is unchanged from the docs.yml rule, apart from the trigger.
   - A handoff commit goes to `main` and runs nothing. It never goes to `full_deploy` or `docs_deploy`.

6. CHANGELOG `[Unreleased]`: one line for each workflow change.

## What does not change

- The full deploy, its approval pause, the `pypi` environment and the release steps.
- AuditEngine's `-e ../daffodil` picks up whatever is checked out in `/home/daffodil`, pushed or not. The trunk matters for other machines and developers, and for the next release, not for AuditEngine on this box.
- The Lambdas install `daffodil==0.7.0` from PyPI. They change only through a release.

## Check before changing

- Branch protection or required status checks on `main` that expect the CI jobs. If there are any, removing the push trigger could block merges. Look in the repo settings or with `gh api repos/raylutz/daffodil/branches/main/protection`.
- Any badge in README.md that shows the CI status of `main`. Without push runs, it would show the last run that happened, which may be old.
