#!/usr/bin/env bash
# Fail if staged files look like secrets. Used by .git/hooks/pre-commit.
set -u
staged=$(git diff --cached --name-only --diff-filter=ACMR)
[ -z "$staged" ] && exit 0
fail=0
while IFS= read -r f; do
  case "$(basename "$f")" in
    kaggle.json|*.env|.env|.env.*) echo "BLOCKED filename: $f"; fail=1 ;;
  esac
done <<< "$staged"
pattern='(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"key"[[:space:]]*:[[:space:]]*"[0-9a-f]{32}")'
while IFS= read -r f; do
  [ -f "$f" ] || continue
  [ "$f" = "scripts/check_secrets.sh" ] && continue
  if git show ":$f" | grep -aEq "$pattern"; then
    echo "BLOCKED token-like content: $f"; fail=1
  fi
done <<< "$staged"
[ "$fail" -eq 0 ] || { echo "Secret check failed."; exit 1; }
