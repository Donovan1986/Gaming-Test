#!/usr/bin/env bash
# Phase 5: locate the client and report its build. Stops if the server does not know that build.
source "$(dirname "$0")/common.sh"
WOW="${1:-}"; [[ -n "$WOW" ]] || WOW="$(dirname "$(find /Applications "$HOME" -maxdepth 4 -name .build.info 2>/dev/null | head -1)")"
[[ -f "$WOW/.build.info" ]] || die "no .build.info found; pass the World of Warcraft folder as arg 1"
log "client dir: $WOW"; grep -i classic_beta "$WOW/.build.info" | tee "$QA/notes/client-build.txt" || true
BUILDNO="$(grep -io 'wow_classic_beta.*' "$WOW/.build.info" | grep -o '1\.60\.[0-9]*\.[0-9]*' | head -1)"
SUPPORTED="$(ls "$SRC/sql/custom/auth" | grep -o 'beta_[0-9]*' | cut -d_ -f2 | tr '\n' ' ')"
echo "SERVER BUILD: $(git -C "$SRC" rev-parse --short HEAD) (supports: $SUPPORTED)"
echo "CLIENT BUILD: ${BUILDNO:-UNKNOWN}"
echo "DATABASE BUILD: TDB 1210.26091 + sql/custom"
[[ -n "$BUILDNO" && " $SUPPORTED " == *" ${BUILDNO##*.} "* ]] && echo "COMPATIBILITY STATUS: OK" || { echo "COMPATIBILITY STATUS: MISMATCH"; exit 2; }
echo "$WOW" > "$QA/notes/client-path.txt"
