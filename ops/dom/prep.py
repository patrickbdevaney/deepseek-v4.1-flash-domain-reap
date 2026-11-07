#!/usr/bin/env python3
"""D0 source preparation for the domain corpora (CPU + network only; no GPU, no model).

Produces everything ops/dom/corpus_spec_dom.py reads that is not a HF dataset:
  clone       bare shallow clones (--depth 1 --filter=blob:limit=512k) of every `git:` source into
              $V41_DOM_SRC (default: $V41_WORK/dom_src); existing clones are kept as they are
  threads     GitHub GraphQL: per THREAD_REPOS repo, the most recently created closed issues and the answered
              discussions -> $V41_WORK/dom/threads/<domain>.jsonl as user/assistant chat rows
              (issue author = user, everyone else = assistant; bots dropped; repos interleaved)
  localrepos  read-only git snapshot of the repos in corpus_spec_dom.OWNER_HOLDOUT["repos"] (empty in this
              release) -> $V41_WORK/dom/repos/<name>.jsonl, one source file per row
  all         every step above, in that order
A summary of what each step wrote goes to $V41_WORK/dom/prep_manifest.json.

RUN (detached):
    setsid nohup .venv/bin/python ops/dom/prep.py all > ops/logs/dom_prep.log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ops.dom import corpus_spec_dom as spec  # noqa: E402
from ops.lib import env  # noqa: E402

STAGE = "dom_prep"
OUT = env.WORK / "dom"
MANIFEST = OUT / "prep_manifest.json"
MSG_MAX = 8000


def _manifest(step: str, rec: dict) -> None:
    m = json.load(open(MANIFEST)) if MANIFEST.exists() else {}
    prev = m.get(step, {}) if step == "threads" else {}     # a --domains run keeps the other domains' records
    m[step] = {**{k: v for k, v in prev.items() if k != "ts"}, **rec, "ts": env.now()}
    env.atomic_json(MANIFEST, m)


def _write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    n = 0
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


# ----------------------------------------------------------------------------- clone
def clone(a) -> None:
    spec.DOM_SRC.mkdir(parents=True, exist_ok=True)
    rec = {}
    for s in spec.git_sources():
        repo_dir, ref = spec.git_paths(s)
        if not repo_dir.startswith(str(spec.DOM_SRC)):
            r = subprocess.run(["git", "-C", repo_dir, "rev-parse", ref], capture_output=True, text=True)
            rec[s["repo"]] = {"path": repo_dir, "ref": ref, "sha": r.stdout.strip(), "reused_local": True}
            continue
        if not (Path(repo_dir) / "HEAD").exists():
            t0 = time.time()
            flt = ["--filter=blob:none"] if s.get("lazy") else \
                [f"--filter=blob:limit={s['blob_limit']}"] if s.get("blob_limit", "512k") else []
            r = subprocess.run(["git", "clone", "--bare", "--depth", "1", *flt, "--quiet",
                                s.get("url") or f"https://github.com/{s['repo']}.git", repo_dir], capture_output=True, text=True)
            if r.returncode != 0:
                rec[s["repo"]] = {"error": r.stderr.strip()[-300:]}
                env.log(f"clone {s['repo']} FAILED: {r.stderr.strip()[-200:]}", STAGE, "WARN")
                subprocess.run(["rm", "-rf", repo_dir])          # inside dom_src only: a half clone would look done
                continue
            env.log(f"clone {s['repo']} in {time.time() - t0:.0f}s", STAGE)
        if s.get("lazy"):
            prefetch(s, repo_dir, ref)
        sha = subprocess.run(["git", "-C", repo_dir, "rev-parse", ref], capture_output=True, text=True).stdout.strip()
        du = subprocess.run(["du", "-sm", repo_dir], capture_output=True, text=True).stdout.split()[0]
        rec[s["repo"]] = {"path": repo_dir, "ref": ref, "sha": sha, "mb": int(du)}
    _manifest("clone", rec)
    bad = [k for k, v in rec.items() if "error" in v]
    env.log(f"clone done: {len(rec) - len(bad)}/{len(rec)} ok{'; FAILED ' + ', '.join(bad) if bad else ''}", STAGE)


def prefetch(s: dict, repo_dir: str, ref: str) -> None:
    """Fetch exactly the blobs select_files() will read (first max_files in hash order) for a blob:none clone,
    in batches of 200 oids per fetch, so s02 never triggers one-object-at-a-time lazy fetches."""
    want = [oid for _, _, oid in spec.select_files(s)]
    miss = spec.missing_blobs(repo_dir, ref)
    todo = [o for o in want if o in miss]
    t0 = time.time()
    for i in range(0, len(todo), 200):
        r = subprocess.run(["git", "-C", repo_dir, "-c", "fetch.negotiationAlgorithm=noop", "fetch", "-q", "--no-tags",
                            "--no-write-fetch-head", "--filter=blob:none", "origin", *todo[i:i + 200]],
                           capture_output=True, text=True)
        if r.returncode != 0:
            env.log(f"prefetch {s['repo']} batch {i}: {r.stderr.strip()[:200]}", STAGE, "WARN")
    still = spec.missing_blobs(repo_dir, ref)          # once: one rev-list per oid took ~10 min per lazy repo
    left = len([o for o in want if o in still])
    env.log(f"prefetch {s['repo']}: {len(want)} selected, {len(todo)} fetched, {left} still missing, "
            f"{time.time() - t0:.0f}s", STAGE)


# ----------------------------------------------------------------------------- threads
Q_ISSUES = """query($owner:String!,$name:String!,$n:Int!){repository(owner:$owner,name:$name){
 hasDiscussionsEnabled
 issues(first:$n,states:CLOSED,orderBy:{field:CREATED_AT,direction:DESC}){nodes{number title body url
   author{login} comments(first:30){nodes{body author{login}}}}}}}"""
Q_DISC = """query($owner:String!,$name:String!,$n:Int!){repository(owner:$owner,name:$name){
 discussions(first:$n,answered:true,orderBy:{field:UPDATED_AT,direction:DESC}){nodes{number title body url
   author{login} answer{body author{login}} comments(first:12){nodes{body author{login}}}}}}}"""


def _gql(query: str, repo: str, n: int) -> dict:
    owner, name = repo.split("/")
    for attempt in range(4):
        r = subprocess.run(["gh", "api", "graphql", "-f", f"query={query}", "-F", f"owner={owner}", "-F", f"name={name}",
                            "-F", f"n={n}"], capture_output=True, text=True)
        if r.returncode == 0:
            return json.loads(r.stdout)["data"]["repository"]
        env.log(f"gql {repo} attempt {attempt}: {r.stderr.strip()[:200]}", STAGE, "WARN")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"graphql failed for {repo}")


def _login(node) -> str:
    return ((node or {}).get("author") or {}).get("login") or "ghost"


def _is_bot(login: str) -> bool:
    return login.endswith("[bot]") or login in ("github-actions", "dependabot", "ghost", "stale", "pytorchmergebot",
                                                 "facebook-github-bot", "codecov", "netlify")


def _thread(repo: str, title: str, body: str, opener: str, replies: list[tuple[str, str]]) -> dict | None:
    msgs = [{"role": "user", "content": f"[{repo}] {title}\n\n{(body or '').strip()}"[:MSG_MAX]}]
    seen = set()
    for who, text in replies:
        text = (text or "").strip()
        # drop bots, one-liners ("+1", "thanks") and repeated boilerplate (sign-off / relicense comments)
        if len(text) < 40 or _is_bot(who) or text in seen:
            continue
        seen.add(text)
        role = "user" if who == opener else "assistant"
        if msgs[-1]["role"] == role:
            msgs[-1]["content"] = (msgs[-1]["content"] + "\n\n" + text)[:MSG_MAX]
        else:
            msgs.append({"role": role, "content": text[:MSG_MAX]})
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    return {"messages": msgs}


def threads(a) -> None:
    rec = {}
    for dom, repos in spec.THREAD_REPOS.items():
        if a.domains and dom not in a.domains:
            continue
        per_repo = {}
        for repo in repos:
            rows = []
            try:
                d = _gql(Q_ISSUES, repo, a.issues)
                for it in d["issues"]["nodes"]:
                    t = _thread(repo, it["title"], it["body"], _login(it),
                                [(_login(c), c["body"]) for c in it["comments"]["nodes"]])
                    if t:
                        rows.append({**t, "url": it["url"], "kind": "issue"})
                if d.get("hasDiscussionsEnabled"):
                    dd = _gql(Q_DISC, repo, a.discussions)
                    for it in dd["discussions"]["nodes"]:
                        ans = it.get("answer") or {}
                        rep = [(_login(ans), ans.get("body"))] + [(_login(c), c["body"]) for c in it["comments"]["nodes"]
                                                                  if c.get("body") != ans.get("body")]
                        t = _thread(repo, it["title"], it["body"], _login(it), rep)
                        if t:
                            rows.append({**t, "url": it["url"], "kind": "discussion"})
            except Exception as e:
                env.log(f"threads {repo}: {type(e).__name__}: {str(e)[:200]}", STAGE, "WARN")
            # issues and discussions alternate inside a repo too, so a quota cut keeps both kinds
            iss = [r for r in rows if r["kind"] == "issue"]; dis = [r for r in rows if r["kind"] == "discussion"]
            per_repo[repo] = [x for pair in zip(iss, dis) for x in pair] + iss[len(dis):] + dis[len(iss):]
            env.log(f"threads {dom}/{repo}: {len(iss)} issues, {len(dis)} discussions", STAGE)
        # round-robin across repos: the s02 quota cut then covers every repo, not just the first
        order, k = [], 0
        while any(k < len(v) for v in per_repo.values()):
            order += [v[k] for v in per_repo.values() if k < len(v)]
            k += 1
        n = _write_jsonl(OUT / "threads" / f"{dom}.jsonl", order)
        rec[dom] = {"rows": n, "per_repo": {r: len(v) for r, v in per_repo.items()}}
    _manifest("threads", rec)


# ----------------------------------------------------------------------------- helpers
def _hash_order(rows: list[dict], key: str) -> list[dict]:
    return sorted(rows, key=lambda r: hashlib.sha1(r[key].encode()).hexdigest())


# ----------------------------------------------------------------------------- local owner repos
# Owner repos (optional; private calibration only, never published). Rows come from git, never the working tree,
# and NOTHING dated on/after spec.OWNER_HOLDOUT["cutoff"] is used: those commits on any ref are reserved for a
# hold-out eval. File rows = the tree of the last commit before the cutoff on the repo's
# main ref; commit rows = message + diff of every commit (any ref) whose author AND committer dates precede it.
LOCAL_EXTS = (".rs", ".md", ".toml", ".wgsl", ".ron", ".sh", ".py", ".js", ".ts", ".html", ".css", ".patch")
LOCAL_SKIP_DIRS = {"target", ".git", "node_modules", "assets", "artifacts", ".cargo", "dist", "build", "vendor"}
COMMIT_MAX_BYTES = 32_000


def _git(root, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, errors="replace").stdout


def _skip_path(path: str) -> bool:
    return any(part in LOCAL_SKIP_DIRS or part.startswith(".") for part in path.split("/")[:-1])


def localrepos(a) -> None:
    cut = spec.OWNER_HOLDOUT["cutoff"]
    rec = {}
    for name, cfg in spec.OWNER_HOLDOUT["repos"].items():
        root = Path(cfg["root"]).expanduser()
        if not (root / ".git").exists():
            rec[name] = {"error": f"{root} has no git history"}
            continue
        head = _git(root, "rev-list", "-1", f"--before={cut}T00:00:00", cfg["ref"]).strip()
        if not head:
            rec[name] = {"error": f"no commit before {cut} on {cfg['ref']}"}
            continue
        ev = tuple(cfg.get("eval_exclude", ()))     # eval-only paths never feed calibration
        # file rows at the pre-cutoff tree
        rows = []
        for line in _git(root, "ls-tree", "-r", "-l", head).splitlines():
            meta, path = line.split("\t", 1)
            _, typ, oid, size = meta.split()
            if typ != "blob" or not path.endswith(LOCAL_EXTS) or _skip_path(path) or path.startswith(ev):
                continue
            if not (spec.FILE_MIN_BYTES <= int(size) <= spec.FILE_MAX_BYTES):
                continue
            data = subprocess.run(["git", "-C", str(root), "cat-file", "blob", oid], capture_output=True).stdout
            if b"\0" in data[:4096]:
                continue
            rows.append({"text": f"// ===== {name}/{path} =====\n" + data.decode("utf-8", "replace"), "id": path})
        rows = _hash_order(rows, "id")
        n_files = _write_jsonl(OUT / "repos" / f"{name}.jsonl", rows)
        # commit-diff rows from every ref, pre-cutoff by both dates, merges skipped
        commits = []
        for line in _git(root, "log", "--all", "--no-merges", f"--before={cut}T00:00:00", "--format=%H %aI %cI").splitlines():
            sha, ad, cd = line.split()
            if ad[:10] >= cut or cd[:10] >= cut:
                continue
            body = _git(root, "show", "--no-color", "-M", "--format=commit %H%nDate: %aI%n%n%B", sha, "--", ".",
                        *[f":(exclude){d}" for d in ("assets", "artifacts", "target", "node_modules", "vendor", "dist")],
                        *[f":(exclude){d}" for d in ev])
            if "diff --git" not in body:
                continue
            if len(body) > COMMIT_MAX_BYTES:     # keep the message and the head of the diff
                body = body[:COMMIT_MAX_BYTES] + "\n[... diff truncated ...]\n"
            commits.append({"text": f"// ===== {name} commit =====\n{body}", "id": sha, "date": ad[:10]})
        commits = _hash_order(commits, "id")
        n_commits = _write_jsonl(OUT / "repos" / f"{name}_commits.jsonl", commits)
        rec[name] = {"root": str(root), "ref": cfg["ref"], "tree_commit": head, "cutoff": cut, "file_rows": n_files,
                     "commit_rows": n_commits, "eval_exclude": list(ev), "latest_commit_used": max((c["date"] for c in commits), default=None)}
        env.log(f"localrepos {name}: {rec[name]}", STAGE)
    _manifest("localrepos", rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["clone", "threads", "localrepos", "all"])
    ap.add_argument("--domains", nargs="*", help="threads: limit to these domains")
    ap.add_argument("--issues", type=int, default=60, help="threads: most recently created closed issues per repo")
    ap.add_argument("--discussions", type=int, default=30, help="threads: answered discussions per repo")
    a = ap.parse_args()
    steps = ["clone", "threads", "localrepos"] if a.step == "all" else [a.step]
    for s in steps:
        env.log(f"start {s}", STAGE)
        {"clone": clone, "threads": threads, "localrepos": localrepos}[s](a)
    env.log("prep done", STAGE)


if __name__ == "__main__":
    main()
