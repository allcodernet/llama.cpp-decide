#!/usr/bin/env bash
# Keep a copy of this repository and of engine/ on the test machine, without a git remote: the laptop stays the git
# authority, the test machine builds and runs.
#
#   ./remote-sync.sh push [--take-over] [--force] [project|engine|all]   (default all)
#       0. Guard: REMOTE_DIR must not be a symbolic link, and must be absent or empty (outside any git work tree or
#          git directory; it then gets `git init`), or a git repository with its own .git directory that this script
#          pushes to (its HEAD is a commit known here, or it holds our manifest, or it has no commit and nothing
#          staged). REMOTE_DIR/engine is pushed only into such a REMOTE_DIR, and is either absent (it then starts as a
#          clone of SEED, origin removed, detached at UPSTREAM_COMMIT) or passes the same test. Source: each remote
#          repository records the checkout that pushes to it (this host's name and machine id and this repository's
#          root, in its git directory); a push from another checkout is refused unless --take-over (which records
#          the new source; step 2 still runs). A target without a record adopts the pushing checkout.
#       1. Committed state: a git bundle of the commits the test machine lacks (the engine's first one carries
#          UPSTREAM_COMMIT..decide), copied with scp into the remote repository's git directory and fetched there.
#       2. Safety check there, before anything is changed: a file whose content exists only there is listed and the
#          push stops. That is a tracked file changed there (e.g. rewritten by a run), or an untracked file at a path
#          the incoming state (the commit plus this side's working-tree changes) writes, unless its content equals
#          the incoming commit's, this working tree's or what the previous push wrote there (a file missing there has
#          nothing to lose). --force prints the same list and then replaces those files.
#       3. The laptop's current branch is checked out there at the same commit id (commits made there are discarded:
#          the branch moves to this side's commit, they stay only in the reflog there), tracked files are reset to it (a
#          reverted edit does not linger), tracked files deleted here are deleted there, and the tracked-modified and
#          untracked-unignored files here are copied with tar over ssh (their list and hashes are kept in the remote
#          repository's git directory for the next check). Any other untracked or ignored file there (results,
#          models, builds, other worktrees) is never touched.
#   ./remote-sync.sh pull PATH...   copy files or directories (relative to the repository; not ., .git or engine) from
#                                    REMOTE_DIR to the same paths here
#   ./remote-sync.sh run CMD...     run CMD (a shell command line) in REMOTE_DIR on the test machine
#
# Settings (required, exported): REMOTE (ssh destination, e.g. user@gpu-host), REMOTE_DIR (the repository there; absolute,
# trailing slashes dropped). REMOTE_SEED: what a missing REMOTE_DIR/engine is cloned from there (default upstream
# llama.cpp on GitHub; a local clone that holds UPSTREAM_COMMIT is faster). Every command first prints its target and
# source on stderr.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
[ -n "${REMOTE:-}" ] || { echo "remote-sync: set and export REMOTE (ssh destination, e.g. user@gpu-host)" >&2; exit 1; }
[ -n "${REMOTE_DIR:-}" ] || { echo "remote-sync: set and export REMOTE_DIR (absolute path of the repository there)" >&2; exit 1; }
while [ "${REMOTE_DIR%/}" != "$REMOTE_DIR" ]; do REMOTE_DIR=${REMOTE_DIR%/}; done
SEED=${REMOTE_SEED:-https://github.com/ggml-org/llama.cpp}  # cloned there for a missing engine/; read only, must contain UPSTREAM_COMMIT
UPSTREAM_COMMIT=60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f
MANIFEST=remote-sync-overlay  # in the remote repository's git directory: "<hash or ->\t<path>\0" per overlaid path
SOURCE_FILE=remote-sync-source  # in the remote repository's git directory: SOURCE of the checkout that pushes there
SOURCE="$(uname -n) $(cat /etc/machine-id 2>/dev/null || echo -) $ROOT"
FORCE=0
TAKEOVER=0

rsh() { ssh -o BatchMode=yes "$REMOTE" "$1"; }
q() { printf '%q' "$1"; }
die() { echo "remote-sync: $*" >&2; exit 1; }
case $REMOTE_DIR in /?*) ;; *) die "REMOTE_DIR must be an absolute path other than /: '$REMOTE_DIR'" ;; esac

# Runs there as: bash -c "$STATE" state DIR. Prints "symlink" (DIR itself is a symbolic link); or "gitlink" (the top
# of a work tree whose .git is a file or a symbolic link: a worktree, a submodule); or "repo", then HEAD's id (or
# "unborn" with an empty index, "unborn-staged" with staged files), the absolute git directory, "manifest" or "-" and
# the recorded source (or "-"); or "absent" (missing or empty) and the git work tree or git directory its nearest existing ancestor is in (or
# "-"); or "other" (a non-empty directory that is not the top of a git work tree).
read -r -d '' STATE <<'EOF' || true
d=$1
if [ -L "$d" ]; then
    echo symlink
elif [ -d "$d" ] && top=$(git -C "$d" rev-parse --show-toplevel 2>/dev/null) && [ "$top" = "$(cd "$d" && pwd -P)" ]; then
    if [ -L "$d/.git" ] || [ ! -d "$d/.git" ]; then echo gitlink; exit; fi
    echo repo
    if ! git -C "$d" rev-parse -q --verify HEAD; then
        if [ -z "$(git -C "$d" ls-files | head -c 1)" ]; then echo unborn; else echo unborn-staged; fi
    fi
    gd=$(git -C "$d" rev-parse --path-format=absolute --git-dir)
    echo "$gd"
    if [ -f "$gd/remote-sync-overlay" ]; then echo manifest; else echo -; fi
    if [ -s "$gd/remote-sync-source" ]; then head -n 1 "$gd/remote-sync-source"; else echo -; fi
elif [ ! -e "$d" ] || { [ -d "$d" ] && [ -z "$(ls -A "$d")" ]; }; then
    echo absent
    a=$d
    while [ ! -e "$a" ]; do a=$(dirname "$a"); done
    if [ "$(git -C "$a" rev-parse --is-inside-git-dir 2>/dev/null)" = true ]; then
        git -C "$a" rev-parse --path-format=absolute --git-dir
    else
        git -C "$a" rev-parse --show-toplevel 2>/dev/null || echo -
    fi
else
    echo other
fi
EOF

# Runs there as: bash -c "$CHECK" check DIR SHA, with this side's manifest on stdin. Prints every path whose content
# there would be lost: it exists there and its blob hash differs from the remote HEAD's, the incoming commit's, this
# side's working tree's (manifest) and what the previous push wrote (stored manifest).
read -r -d '' CHECK <<'EOF' || true
cd "$1" || exit 2
sha=$2
declare -A lap prev
while IFS= read -r -d '' rec; do [ -n "$rec" ] && lap[${rec#*$'\t'}]=${rec%%$'\t'*}; done
mf=$(git rev-parse --git-path remote-sync-overlay)
if [ -f "$mf" ]; then
    sep=''
    if tr -d '\0' <"$mf" | cmp -s - "$mf"; then sep=$'\n'; fi  # a manifest written by an earlier version of this script: one line each
    while IFS= read -r -d "$sep" rec; do [ -n "$rec" ] && prev[${rec#*$'\t'}]=${rec%%$'\t'*}; done <"$mf"
fi
blob() { git rev-parse -q --verify "$1:$2" 2>/dev/null || echo -; }
{
    if git rev-parse -q --verify HEAD >/dev/null; then git diff -z --no-renames --name-only HEAD; fi
    comm -z -23 <(git ls-tree -z -r --name-only "$sha" | sort -z) <(git ls-files -z | sort -z)
    for p in "${!lap[@]}"; do printf '%s\0' "$p"; done
} | sort -zu | while IFS= read -r -d '' p; do
    if [ -e "$p" ] || [ -L "$p" ]; then r=$(git hash-object -- "$p"); else continue; fi  # absent: nothing to lose
    [ "$r" = "$(blob HEAD "$p")" ] && continue
    [ "$r" = "$(blob "$sha" "$p")" ] && continue
    [ -n "${lap[$p]+x}" ] && [ "$r" = "${lap[$p]}" ] && continue
    [ -n "${prev[$p]+x}" ] && [ "$r" = "${prev[$p]}" ] && continue
    printf '%s\n' "$p"
done
EOF

# remote_state DIR: runs STATE there and sets RS_KIND, RS_HEAD, RS_GITDIR, RS_MANIFEST, RS_SOURCE, RS_INSIDE.
remote_state() {
    local out st
    out=$(rsh "bash -c $(q "$STATE") state $(q "$1")")
    mapfile -t st <<<"$out"
    RS_KIND=${st[0]} RS_HEAD= RS_GITDIR= RS_MANIFEST= RS_SOURCE= RS_INSIDE=
    case $RS_KIND in
        repo) RS_HEAD=${st[1]} RS_GITDIR=${st[2]} RS_MANIFEST=${st[3]} RS_SOURCE=${st[4]} ;;
        absent) RS_INSIDE=${st[1]} ;;
    esac
}

# require_ours LOCAL_REPO DIR: step 0 for an existing DIR (after remote_state DIR).
require_ours() {
    case $RS_KIND in
        symlink) die "$2 is a symbolic link: wrong REMOTE_DIR?" ;;
        gitlink) die "$2 has no .git directory of its own (a worktree, a submodule or a linked .git): wrong REMOTE_DIR?" ;;
        other) die "$2 is not empty and not a git repository: wrong REMOTE_DIR?" ;;
        absent) return ;;
    esac
    [ "$RS_MANIFEST" = manifest ] || [ "$RS_HEAD" = unborn ] || git -C "$1" cat-file -e "$RS_HEAD^{commit}" 2>/dev/null \
        || die "$2 is a git repository this script did not push to (HEAD ${RS_HEAD:0:13} unknown here, no $MANIFEST): wrong REMOTE_DIR?"
    if [ "$RS_SOURCE" != - ] && [ "$RS_SOURCE" != "$SOURCE" ]; then
        [ "$TAKEOVER" = 1 ] || die "$2 is pushed from another checkout ($RS_SOURCE; this is $SOURCE): push --take-over" \
            "to take it over (files there that would be lost still stop the push)"
        echo "remote-sync: $2: taking over from $RS_SOURCE (--take-over)" >&2
    fi
}

# sync_repo LOCAL_REPO REMOTE_REPO BASE: steps 1-3 of the header; REMOTE_REPO exists and passed require_ours.
sync_repo() {
    local repo=$1 rdir=$2 base=$3 branch sha have range tmp conflicts hashes rbundle
    branch=$(git -C "$repo" symbolic-ref --short HEAD) || die "$repo: detached HEAD"
    sha=$(git -C "$repo" rev-parse HEAD)
    remote_state "$rdir"
    have=$RS_HEAD
    [ "$have" = unborn ] && have=
    rbundle=$RS_GITDIR/remote-sync.bundle
    tmp=$(mktemp -d)
    if [ "$have" != "$sha" ] && ! rsh "git -C $(q "$rdir") cat-file -e $sha^{commit} 2>/dev/null"; then
        range=$branch
        if [ -n "$have" ] && git -C "$repo" merge-base --is-ancestor "$have" "$sha" 2>/dev/null; then
            range=$have..$branch
        elif [ -n "$base" ]; then
            range=$base..$branch
        fi
        git -C "$repo" bundle create -q "$tmp/sync.bundle" "$range"
        scp -q -o BatchMode=yes "$tmp/sync.bundle" "$REMOTE:$rbundle"  # SFTP mode: the path is taken verbatim
        rsh "git -C $(q "$rdir") fetch -q $(q "$rbundle") refs/heads/$(q "$branch"); rc=\$?; rm -f $(q "$rbundle"); exit \$rc"
        echo "$rdir: fetched $range"
    fi

    # this side's working-tree changes: deleted (-) and changed or untracked (blob hash) paths, NUL-separated
    git -C "$repo" diff -z --no-renames --name-only --diff-filter=D HEAD >"$tmp/deleted"
    { git -C "$repo" diff -z --no-renames --name-only --diff-filter=d HEAD
      git -C "$repo" ls-files -z --others --exclude-standard; } >"$tmp/changed"
    hashes=$(xargs -0 -r git -C "$repo" hash-object -- <"$tmp/changed")
    local -a H=() P=()
    [ -n "$hashes" ] && mapfile -t H <<<"$hashes"
    mapfile -d '' -t P <"$tmp/changed"
    [ ${#H[@]} -eq ${#P[@]} ] || die "$repo: ${#H[@]} hashes for ${#P[@]} changed files"
    { while IFS= read -r -d '' p; do printf -- '-\t%s\0' "$p"; done <"$tmp/deleted"
      for i in "${!P[@]}"; do printf '%s\t%s\0' "${H[$i]}" "${P[$i]}"; done; } >"$tmp/manifest"

    conflicts=$(rsh "bash -c $(q "$CHECK") check $(q "$rdir") $sha" <"$tmp/manifest")
    if [ -n "$conflicts" ] && [ "$FORCE" = 0 ]; then
        rm -rf "$tmp"
        die "$rdir: these files there differ from every version here and would be lost (pull them, or push --force):
$conflicts"
    elif [ -n "$conflicts" ]; then
        echo "remote-sync: $rdir: --force replaces these files there, which differ from every version here:
$conflicts" >&2
    fi

    rsh "git -C $(q "$rdir") checkout -q -f -B $(q "$branch") $sha && git -C $(q "$rdir") reset -q --hard"
    echo "$rdir: $branch at ${sha:0:9}, tracked files reset"
    if [ -s "$tmp/deleted" ]; then
        rsh "cd $(q "$rdir") && xargs -0 rm -f --" <"$tmp/deleted"
        echo "$rdir: removed $(tr -cd '\0' <"$tmp/deleted" | wc -c) file(s) deleted here"
    fi
    if [ ${#P[@]} -gt 0 ]; then
        tar -C "$repo" -cf - --null --verbatim-files-from -T "$tmp/changed" | rsh "tar -xf - -C $(q "$rdir")"
        echo "$rdir: copied ${#P[@]} changed or untracked file(s)"
    fi
    rsh "cat >$(q "$RS_GITDIR/$MANIFEST")" <"$tmp/manifest"
    if [ "$RS_SOURCE" != "$SOURCE" ]; then
        rsh "printf '%s\n' $(q "$SOURCE") >$(q "$RS_GITDIR/$SOURCE_FILE")"
        [ "$RS_SOURCE" = - ] && echo "$rdir: source recorded (adopted): $SOURCE"
    fi
    rm -rf "$tmp"
}

push_project() {
    remote_state "$REMOTE_DIR"
    require_ours "$ROOT" "$REMOTE_DIR"
    if [ "$RS_KIND" = absent ]; then
        [ "$RS_INSIDE" = - ] || die "$REMOTE_DIR would be created inside the git repository $RS_INSIDE: wrong REMOTE_DIR?"
        rsh "mkdir -p $(q "$REMOTE_DIR") && git -C $(q "$REMOTE_DIR") init -q"
    fi
    sync_repo "$ROOT" "$REMOTE_DIR" ""
}

push_engine() {
    local rengine=$REMOTE_DIR/engine
    remote_state "$REMOTE_DIR"
    require_ours "$ROOT" "$REMOTE_DIR"
    [ "$RS_KIND" = repo ] || die "$REMOTE_DIR is not a repository yet: push project first"
    remote_state "$rengine"
    require_ours "$ROOT/engine" "$rengine"
    if [ "$RS_KIND" = absent ]; then
        rsh "git clone -q --no-hardlinks --no-checkout $(q "$SEED") $(q "$rengine") \
            && git -C $(q "$rengine") remote remove origin \
            && git -C $(q "$rengine") checkout -q --detach $UPSTREAM_COMMIT || { rm -rf $(q "$rengine"); exit 1; }"
    fi
    sync_repo "$ROOT/engine" "$rengine" "$UPSTREAM_COMMIT"
}

cmd=${1:-}
[ $# -gt 0 ] && shift
case $cmd in push | pull | run) echo "remote-sync: $cmd $REMOTE:$REMOTE_DIR (source $SOURCE)" >&2 ;; esac
case $cmd in
    push)
        while :; do
            case ${1:-} in
                --force) FORCE=1; shift ;;
                --take-over) TAKEOVER=1; shift ;;
                *) break ;;
            esac
        done
        case ${1:-all} in
            project) push_project ;;
            engine) push_engine ;;
            all) push_project; push_engine ;;
            *) die "push [--take-over] [--force] [project|engine|all]" ;;
        esac ;;
    pull)
        [ $# -gt 0 ] || die "pull PATH..."
        paths=()
        for p in "$@"; do
            while [ "${p%/}" != "$p" ]; do p=${p%/}; done
            case /$p/ in //* | */../* | */./* | */.git/*) die "pull: '$p' is not a relative path inside the repository" ;; esac
            [ "$p" != engine ] || die "pull: not engine/ (pull files inside it)"
            paths+=("$(q "$p")")
        done
        rsh "cd $(q "$REMOTE_DIR") && tar -cf - -- ${paths[*]}" | tar -xf - -C "$ROOT"
        echo "pulled: $*" ;;
    run)
        [ $# -gt 0 ] || die "run CMD..."
        rsh "cd $(q "$REMOTE_DIR") || exit 1; $*" ;;  # `;`: a trailing `&` in CMD backgrounds CMD alone
    *) die "usage: $0 push [--take-over] [--force] [project|engine|all] | pull PATH... | run CMD..." ;;
esac
