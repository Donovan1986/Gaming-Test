#!/usr/bin/env bash
# Phase 1-3: check deps, clone the pinned forever commit, apply QA patch, build natively (arm64).
source "$(dirname "$0")/common.sh"
[[ "$(uname -s)" == Darwin ]] || die "Run this on the Mac"
{ uname -m; sw_vers; df -h /; } | tee "$LOGS/env.txt"
command -v brew >/dev/null || die "Homebrew is missing: https://brew.sh"
need=(); for f in cmake boost openssl@3 mysql readline; do brew list --versions "$f" >/dev/null 2>&1 || need+=("$f"); done
((${#need[@]})) && { log "brew install ${need[*]}"; brew install "${need[@]}"; }
if [[ ! -d "$SRC/.git" ]]; then git clone --single-branch -b forever https://github.com/advocaite/TrinityCore.git "$SRC"; fi
git -C "$SRC" fetch origin forever && git -C "$SRC" checkout -q "$COMMIT_PIN"
git -C "$SRC" apply --reverse --check "$KIT/patches/qa-level60.patch" 2>/dev/null || git -C "$SRC" apply "$KIT/patches/qa-level60.patch"
git -C "$SRC" log -1 --oneline | tee "$QA/notes/commit.txt"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="$SERVER" \
  -DTOOLS=1 -DSCRIPTS=static -DWITH_WARNINGS=0 \
  -DOPENSSL_ROOT_DIR="$(brew --prefix openssl@3)" -DMYSQL_ADD_INCLUDE_PATH="$(brew --prefix mysql)/include" \
  2>&1 | tee "$LOGS/cmake.log"
cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu)" 2>&1 | tee "$LOGS/build.log" | grep -E "error|Linking CXX executable" || true
cmake --install "$BUILD" >>"$LOGS/build.log" 2>&1 || die "install failed (see logs/build.log)"
for b in worldserver bnetserver mapextractor vmap4extractor vmap4assembler mmaps_generator; do
  [[ -x "$BIN/$b" ]] && file "$BIN/$b" | grep -q arm64 && log "OK $b" || die "missing/non-arm64 binary: $b"
done
