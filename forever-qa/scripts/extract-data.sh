#!/usr/bin/env bash
# Extract dbc/gt/maps/vmaps/mmaps from the client into $DATA (skips anything already present).
source "$(dirname "$0")/common.sh"
WOW="$(cat "$QA/notes/client-path.txt")" || die "run check-client.sh first"
W="$QA/data/.work"; mkdir -p "$W"; cd "$W"
[[ -d "$DATA/dbc" && -d "$DATA/maps" ]] || { "$BIN/mapextractor" -i "$WOW" -o "$W" 2>&1 | tee "$LOGS/mapextractor.log"; mv dbc gt maps cameras "$DATA/" 2>/dev/null || true; }
[[ -d "$DATA/vmaps" ]] || { "$BIN/vmap4extractor" -d "$WOW" 2>&1 | tee "$LOGS/vmap.log"; mkdir -p vmaps; "$BIN/vmap4assembler" Buildings vmaps >>"$LOGS/vmap.log" 2>&1; mv vmaps "$DATA/"; }
[[ -d "$DATA/mmaps" ]] || { ln -sfn "$DATA/maps" maps; ln -sfn "$DATA/vmaps" vmaps; "$BIN/mmaps_generator" 2>&1 | tee "$LOGS/mmaps.log"; mv mmaps "$DATA/"; }
ls "$DATA"
