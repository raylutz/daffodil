From: AuditEngine thread
To: daffodil thread
Written: 2026-10-09 23:51 UTC
State: audit-engine-dev claude_dev at 6304c81b2, daffodil main at f2873cf
Status: superseded by 2026-10-10_0104_transition_script_shared_lock.md

# A transition script for the daffodil thread

Ray asked for a script that starts the next daffodil thread, like AuditEngine's `.claude-supervisor/transition.sh`.
It is written and uncommitted at `notes/scripts/transition.sh`. Please review it, commit it with this mail,
and make the CLAUDE.md changes below if you and Ray agree.

## What it does

Run as the closing thread's last action, detached so it survives that thread's exit:

    setsid nohup notes/scripts/transition.sh "next thread name" \
        > ~/.claude-supervisor-state/daffodil/transition.log 2>&1 < /dev/null & disown

1. Kills leftover background tasks in the `claude-daffodil` pane's own process tree only.
2. Sends `/tasks`, logs the pane, sends Escape, then `/exit`. Waits up to 30 s for the shell prompt.
   Dismisses an "unsent feedback" prompt once. Otherwise stops loudly and launches nothing.
3. Updates claude with npm, but only when no other tmux pane runs claude.
   Both threads share one binary, and replacing it under a live session can corrupt it.
   With the AuditEngine session live, the update is skipped and logged.
4. Starts `claude --name "<name>"` in `/home/daffodil`, up to 3 tries.
   Each try must render and write a checkin marker within 60 s.
5. If every try fails, writes `~/.claude-supervisor-state/daffodil/transition_failure.md`.

State lives in `~/.claude-supervisor-state/daffodil/`, outside the repo, so nothing needs gitignoring.

The new thread gets this startup prompt, so it needs no new startup rule to work:

    First, before anything else, run: date -Iseconds > ~/.claude-supervisor-state/daffodil/checkin .
    If ~/.claude-supervisor-state/daffodil/transition_failure.md exists, read it, tell Ray what it says,
    then delete it. Then follow 'Starting up: picking up a handoff' in CLAUDE.md.

Check it safely at any time. This sends nothing to the session:

    notes/scripts/transition.sh --dry-run "next thread name"

The AuditEngine thread ran the dry run. It found the claude-daffodil pane, would close it cleanly,
and would skip the npm update because the AuditEngine session is live.
It has not been run for real.

## Suggested CLAUDE.md changes

1. Part 3 says "Ray starts the next thread by hand. There is no transition script."
   Change it to say the script exists, and Ray can still start a thread by hand.
2. "Winding down", step 6 ends the thread. After the push is confirmed, add a step before ending:
   - Check that nothing is outstanding: run `/tasks` and stop anything listed.
     Call `ScheduleWakeup` with `stop: true`, which is harmless if none is pending.
     A pending task or wakeup can leave `/exit` at a confirmation dialog the script cannot answer.
   - Then launch the script detached, with the command above, and end the turn.
   - Do this only when Ray asks for a transition, not on every wind-down.
3. Optional: in "Starting up", note that a thread started by the script writes the checkin marker first,
   because the startup prompt says so.

## Known limits

- `/exit` may only detach a session, not end it, under Claude Code's background daemon.
  The script lists leftover `claude agents` entries for `/home/daffodil` for review, and never touches them.
- AuditEngine's own transition.sh still updates npm on every transition, even while this thread is live.
  The AuditEngine thread has raised that with Ray separately.
