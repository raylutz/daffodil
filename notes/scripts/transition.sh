#!/bin/bash
# transition.sh -- start the next daffodil thread in its tmux session.
#
# Adapted from /home/audit-engine-dev/.claude-supervisor/transition.sh.
# That script's header holds the history behind each safeguard here.
#
# Process orchestration only. The handoff itself follows CLAUDE.md:
# the closing thread writes and pushes the handoff, then launches this script.
# The new thread picks it up per "Starting up: picking up a handoff".
#
# Usage, from the closing thread as its last action:
#   setsid nohup notes/scripts/transition.sh "next thread name" \
#       > ~/.claude-supervisor-state/daffodil/transition.log 2>&1 < /dev/null & disown
#
# Check what it would do, without sending anything to the session:
#   notes/scripts/transition.sh --dry-run "next thread name"
#
# Steps:
#   1. Kill leftover background tasks in the session's own process tree.
#   2. Send /tasks, log the pane, send Escape, then /exit. Wait for the shell prompt.
#      Dismiss an "unsent feedback" prompt once. Stop loudly after 30 s.
#   3. Take the lock shared with AuditEngine's transition.sh, waiting up to 10 minutes.
#   4. Update claude with npm.
#   5. Launch claude --name, up to 3 tries.
#      Each try must render and write the checkin marker within 60 s.
#      Then release the shared lock.
#   6. If every try fails, write transition_failure.md to the state folder.
#
# Why a shared lock
#   Both threads share one claude binary.
#   Replacing it under a running session is safe. npm writes a new file, and the old process keeps its copy.
#   Starting claude while npm is still writing it is not safe.
#   The lock keeps the two scripts from installing or launching at the same time.
#
# State lives outside the repo, so nothing in it needs gitignoring:
#   ~/.claude-supervisor-state/daffodil/  transition.lock, checkin, npm_update.log, transition_failure.md
#   ~/.claude-supervisor-state/claude_update.lock  shared with AuditEngine's transition.sh

set -u

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

SESSION="claude-daffodil"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="$HOME/.claude-supervisor-state/daffodil"
LOCK_FILE="$STATE_DIR/transition.lock"
CHECKIN_FILE="$STATE_DIR/checkin"
FAILURE_FILE="$STATE_DIR/transition_failure.md"
SHARED_UPDATE_LOCK="$HOME/.claude-supervisor-state/claude_update.lock"

mkdir -p "$STATE_DIR"

log() { echo "[transition $(date +%H:%M:%S)] $*"; }

DISPLAY_NAME="${1:?Usage: transition.sh [--dry-run] \"next thread name\"}"

STARTUP_PROMPT="First, before anything else, run: date -Iseconds > $CHECKIN_FILE . \
If $FAILURE_FILE exists, read it, tell Ray what it says, then delete it. \
Then follow 'Starting up: picking up a handoff' in CLAUDE.md."

# All PIDs below $1 in the process tree.
list_descendant_pids() {
  local children
  children=$(pgrep -P "$1" 2>/dev/null || true)
  for c in $children; do
    echo "$c"
    list_descendant_pids "$c"
  done
}

# claude agents entries for $1, a project folder. Logged for review, never touched.
list_other_agent_sessions() {
  claude agents --json 2>/dev/null | python3 -c '
import json, sys
cwd = sys.argv[1]
try:
    rows = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for row in rows:
    if row.get("cwd") != cwd:
        continue
    # Idle spare workers carry their own id as name. Skip them.
    if row.get("kind") == "background" and row.get("name") == row.get("id"):
        continue
    print("  {}  kind={}  status={}  state={}  name={}".format(
        row.get("sessionId"), row.get("kind"), row.get("status"),
        row.get("state", ""), row.get("name", "")))
' "$1"
}

shell_name=$(basename "${SHELL:-bash}")

# ---- dry run: report what would happen, send nothing ----
if [ "$DRY_RUN" -eq 1 ]; then
  log "DRY RUN for '$DISPLAY_NAME'"
  log "project: $PROJECT_DIR   session: $SESSION   state: $STATE_DIR"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    cmd=$(tmux list-panes -t "$SESSION" -F '#{pane_current_command}' 2>/dev/null | head -1)
    path=$(tmux list-panes -t "$SESSION" -F '#{pane_current_path}' 2>/dev/null | head -1)
    log "session found, pane runs '$cmd' in $path"
    if [ "$cmd" != "$shell_name" ]; then log "would send /tasks, Escape, /exit and wait for '$shell_name'"; fi
  else
    log "no '$SESSION' session: would create one in $PROJECT_DIR"
  fi
  if flock -n "$SHARED_UPDATE_LOCK" true 2>/dev/null; then
    log "shared lock is free: would take it, run the npm update, launch, then release it"
  else
    log "shared lock is held by another transition: would wait up to 10 minutes for it"
  fi
  if [ -f "$LOCK_FILE" ]; then log "WARNING: lock file exists: $(cat "$LOCK_FILE")"; fi
  log "would launch: claude --name \"$DISPLAY_NAME\" with the startup prompt:"
  log "  $STARTUP_PROMPT"
  exit 0
fi

if [ -f "$LOCK_FILE" ]; then
  log "ERROR: transition already in progress ($LOCK_FILE exists)."
  log "If nothing is actually running, remove it and retry: rm '$LOCK_FILE'"
  exit 1
fi
echo "pid=$$ started=$(date -Iseconds)" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

sleep 5   # let the closing turn finish rendering before /exit reaches the pane

# ---- close the running session, if there is one ----
if tmux has-session -t "$SESSION" 2>/dev/null; then
  cmd=$(tmux list-panes -t "$SESSION" -F '#{pane_current_command}' 2>/dev/null | head -1)

  if [ -n "$cmd" ] && [ "$cmd" != "$shell_name" ]; then
    log "running process detected ('$cmd'), checking for background tasks first"

    # Background tasks run through Claude Code's shell-snapshot wrapper.
    # Only this pane's own tree is searched. Other sessions are left alone.
    pane_pid=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' 2>/dev/null | head -1)
    if [ -n "$pane_pid" ]; then
      for pid in $(list_descendant_pids "$pane_pid"); do
        cmdline=$(ps -o cmd= -p "$pid" 2>/dev/null || true)
        if echo "$cmdline" | grep -q 'shell-snapshots/snapshot-bash'; then
          log "killing leftover background task (pid $pid): $cmdline"
          pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
          if [ -n "$pgid" ]; then kill -TERM -- "-$pgid" 2>/dev/null || true; fi
        fi
      done
    fi

    tmux send-keys -t "$SESSION" '/tasks' Enter
    sleep 2
    log "pane after /tasks:"
    tmux capture-pane -p -t "$SESSION" -S -20 2>/dev/null | while IFS= read -r line; do log "  $line"; done

    # The /tasks view is a modal. /exit sent while it is open is swallowed.
    tmux send-keys -t "$SESSION" Escape
    sleep 1

    log "sending /exit"
    tmux send-keys -t "$SESSION" '/exit' Enter
    waited=0
    dismissed_feedback_prompt=0
    while true; do
      cmd=$(tmux list-panes -t "$SESSION" -F '#{pane_current_command}' 2>/dev/null | head -1)
      if [ "$cmd" = "$shell_name" ] || [ -z "$cmd" ]; then
        log "close confirmed"
        break
      fi
      # An unsent feedback draft makes /exit ask first. Escape means discard and exit.
      if [ "$dismissed_feedback_prompt" -eq 0 ]; then
        if tmux capture-pane -p -t "$SESSION" -S -10 2>/dev/null | grep -qi "discard and exit\|unsent feedback"; then
          log "unsent-feedback prompt detected, dismissing with Escape"
          tmux send-keys -t "$SESSION" Escape
          dismissed_feedback_prompt=1
          sleep 1
          continue
        fi
      fi
      sleep 1
      waited=$((waited + 1))
      if [ "$waited" -ge 30 ]; then
        log "ERROR: pane still shows '$cmd' after ${waited}s. Stopping without launching anything."
        log "Pane content:"
        tmux capture-pane -p -t "$SESSION" -S -15 2>/dev/null | while IFS= read -r line; do log "  $line"; done
        log "Check it by hand: tmux attach -t $SESSION"
        log "Once it is idle, rerun: notes/scripts/transition.sh \"$DISPLAY_NAME\""
        exit 1
      fi
    done
  else
    log "session exists but is idle, nothing to close"
  fi
else
  log "no '$SESSION' session found, creating one"
  tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"
fi

# ---- take the shared lock, held until the launch checks in or gives up ----
exec 9>"$SHARED_UPDATE_LOCK"
log "waiting for the shared update/launch lock ($SHARED_UPDATE_LOCK)"
if ! flock -w 600 9; then
  log "ERROR: another transition held $SHARED_UPDATE_LOCK for 10 minutes. Stopping without launching anything."
  log "Check the other thread's transition log, then rerun: notes/scripts/transition.sh \"$DISPLAY_NAME\""
  exit 1
fi
log "shared lock acquired"

# ---- update claude ----
log "updating claude (npm install -g @anthropic-ai/claude-code@latest)"
rm -f "$HOME/.claude/.update.lock"
before_version=$(claude --version 2>/dev/null)
UPDATE_LOG="$STATE_DIR/npm_update.log"
if npm install -g @anthropic-ai/claude-code@latest > "$UPDATE_LOG" 2>&1; then
  after_version=$(claude --version 2>/dev/null)
  if [ "$before_version" = "$after_version" ]; then
    log "claude already at latest ($after_version)"
  else
    log "claude updated: $before_version -> $after_version"
  fi
else
  log "ERROR: npm update failed. Stopping without launching anything. See $UPDATE_LOG:"
  tail -n 30 "$UPDATE_LOG" 2>/dev/null | while IFS= read -r line; do log "  $line"; done
  exit 1
fi

other_sessions=$(list_other_agent_sessions "$PROJECT_DIR")
if [ -n "$other_sessions" ]; then
  log "NOTE: other 'claude agents' entries exist for $PROJECT_DIR. Review by hand, not touched:"
  echo "$other_sessions" | while IFS= read -r line; do log "$line"; done
fi

# ---- launch the new thread, and confirm it checked in ----
CHECKIN_TIMEOUT_S=60
MAX_ATTEMPTS=3
FAILURE_LOG=""
append_failure() { FAILURE_LOG="${FAILURE_LOG}${1}"$'\n'; }

attempt=1
launch_ok=0
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  log "launch attempt $attempt/$MAX_ATTEMPTS: starting claude as '$DISPLAY_NAME'"
  rm -f "$CHECKIN_FILE"
  tmux send-keys -t "$SESSION" "cd \"$PROJECT_DIR\" && unset ANTHROPIC_API_KEY; claude --name \"$DISPLAY_NAME\" \"$STARTUP_PROMPT\"" Enter

  # The pane must leave the shell within 15 s.
  launch_waited=0
  launch_confirmed=0
  while [ "$launch_waited" -lt 15 ]; do
    new_cmd=$(tmux list-panes -t "$SESSION" -F '#{pane_current_command}' 2>/dev/null | head -1)
    if [ -n "$new_cmd" ] && [ "$new_cmd" != "$shell_name" ]; then
      log "new process running ('$new_cmd')"
      launch_confirmed=1
      break
    fi
    sleep 1
    launch_waited=$((launch_waited + 1))
  done

  reason=""
  if [ "$launch_confirmed" -ne 1 ]; then
    reason="claude did not start within ${launch_waited}s (pane showed '$new_cmd')"
  else
    # The new thread writes the checkin marker as its first action.
    checkin_waited=0
    while [ "$checkin_waited" -lt "$CHECKIN_TIMEOUT_S" ]; do
      if [ -f "$CHECKIN_FILE" ]; then
        log "checkin confirmed ($(cat "$CHECKIN_FILE" 2>/dev/null))"
        launch_ok=1
        break
      fi
      sleep 1
      checkin_waited=$((checkin_waited + 1))
    done
    if [ "$launch_ok" -eq 1 ]; then break; fi
    reason="claude started but wrote no checkin within ${CHECKIN_TIMEOUT_S}s"
  fi

  pane_snap=$(tmux capture-pane -p -t "$SESSION" -S -20 2>/dev/null)
  log "launch attempt $attempt/$MAX_ATTEMPTS FAILED: $reason"
  echo "$pane_snap" | while IFS= read -r line; do log "  $line"; done
  append_failure "## Attempt $attempt/$MAX_ATTEMPTS -- $(date -Iseconds)"
  append_failure "**Failure**: $reason"
  append_failure ""
  append_failure '```'
  append_failure "$pane_snap"
  append_failure '```'
  append_failure ""

  if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
    log "tearing down the stuck pane and retrying"
    tmux kill-session -t "$SESSION" 2>/dev/null
    tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"
  fi
  attempt=$((attempt + 1))
done

flock -u 9
log "shared lock released"

if [ "$launch_ok" -ne 1 ]; then
  log "ERROR: gave up after $MAX_ATTEMPTS launch attempts. No healthy session."
  log "Check by hand: tmux attach -t $SESSION"
  {
    echo "# Transition failure: $DISPLAY_NAME"
    echo
    echo "notes/scripts/transition.sh gave up after $MAX_ATTEMPTS launch attempts at $(date -Iseconds)."
    echo "A person had to start the session that is reading this."
    echo
    echo "$FAILURE_LOG"
  } > "$FAILURE_FILE"
  log "failure report written to $FAILURE_FILE"
  exit 1
fi

log "transition complete: new session running and checked in"
