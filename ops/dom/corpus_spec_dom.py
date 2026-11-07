"""Domain calibration corpora for the domain-REAP track (docs/domain/PLAN.md, phase D0).

Selected by `V41_DOM=1`: s02 then builds from THIS module into $V41_WORK/dom/corpus and s03 reads
that corpus and writes everything under $V41_WORK/dom/s03. Without the variable the original
pipeline (ops/corpus_spec.py, $V41_WORK/corpus, $V41_WORK/s03) is untouched.

Buckets in this release: gamedev, mlsys, robotics, quantum (the published subset of a larger frozen
domain set; the other buckets' source lists are not released). Each is one 8,192-token-row pool of
~400k routed tokens (50 rows, 5 of them the frozen held-out split). No vision and no long-context
pool: the domain pass adds statistics to the original 9-bucket bank (same teacher, same expert ids),
it does not replace it.

Mix inside a domain: (a) agentic/chat-format material (agent transcripts where they exist, GitHub
issue/discussion threads as user/assistant turns, Stack Exchange Q&A, tool-calling SFT sets) and
(b) code/docs text (per-FILE rows from upstream git refs, HF code/paper datasets). Sources run in
list order and a short source's shortfall flows to the sources after it by weight, so the chat
sources come first and a missing `local:` file just hands its quota on.

Provenance: generator "claude" is allowed and tagged on this track, and `--generator-policy no-claude`
rebuilds the corpus without it. Licence policy used for calibration: permissive, attribution,
share-alike and copyleft (GPL/LGPL/AGPL/MPL) are allowed and recorded; NC, no-derivatives and
Commons-Clause are excluded. s02 records the live licence per source (HF card, or the LICENSE /
COPYING text of the git ref via `git_licence`).

Source kinds (same format as ops/corpus_spec.py, plus an optional per-source `stream` callable that
s02 uses in place of its default loader when present):
  hf_id                      a HF dataset (`data_files` / `loader` via the `stream` callable)
  git:<bare repo>@<ref>      per-file rows of an upstream ref, read with `git cat-file` (never a
                             working tree), filtered by `include` path prefixes and `exts`
  local:<path>               jsonl under $V41_WORK (self-traces, threads, optional private rows);
                             produced by ops/dom/prep.py and ops/dom/selftraces.py
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
from pathlib import Path

from ops import corpus_spec as base

SEQ_LEN = base.SEQ_LEN
ROWS_PER_DOMAIN = 50
# Public release: the 4 published buckets, in their relative order from the original frozen domain set. The original
# run used a larger frozen bucket list (the other buckets' source lists are withheld), so bucket ids, sub-area ids and
# pass grouping produced by THIS module differ from the banks behind the published masks (see REVIEW.md / README.md).
BUCKETS = ["gamedev", "mlsys", "robotics", "quantum"]
TOKEN_TARGET = {b: 1.0 / len(BUCKETS) for b in BUCKETS}
TOTAL_TOKENS = ROWS_PER_DOMAIN * len(BUCKETS) * SEQ_LEN                   # 409,600 per domain
# D1 runs as passes over bucket subsets (ops/dom/passes.py): ~55 KB/token of NVMe scratch makes 4 buckets
# (1.64M tokens, ~90 GB) the largest pass that fits. Every pass banks all buckets, so banks merge by sum/max.
# Contiguous groups in frozen order, as even as possible with at most PASS_MAX buckets (4 buckets -> one pass).
PASS_MAX = 4
_NP = -(-len(BUCKETS) // PASS_MAX)
_SZ = [len(BUCKETS) // _NP + (i < len(BUCKETS) % _NP) for i in range(_NP)]
PASSES = {f"p{i + 1}": BUCKETS[sum(_SZ[:i]):sum(_SZ[:i + 1])] for i in range(_NP)}
HELDOUT_ROWS_FRACTION = 0.10
LONG_MIN, LONG_MAX = base.LONG_MIN, base.LONG_MAX                        # unused (no longctx pool); s02 hashes them
VISION_SOURCES: list = []
LONG_SOURCES: list = []
REPO_EXTS = base.REPO_EXTS
SCIENCE_URL, KNOWLEDGE_URL, FINANCE_URL = base.SCIENCE_URL, base.KNOWLEDGE_URL, base.FINANCE_URL

# Bare shallow clones (`--filter=blob:limit=512k`) of every git source live here (a large, slow disk is fine).
DOM_SRC = Path(os.environ.get("V41_DOM_SRC", Path(os.environ.get("V41_WORK", Path.home() / "v41-work")) / "dom_src"))
FILE_MIN_BYTES, FILE_MAX_BYTES = 400, 200_000
SPLIT_BYTES = 48_000

S = base.S
_first, _text, _qa = base._first, base._text, base._qa


# ----------------------------------------------------------------------------- row functions
def _chat(row):
    """OpenAI-format messages, keeping tool_calls / tool results / reasoning_content / tools so the
    checkpoint encoding renders DSML tool calls (base._messages keeps only role+content strings).
    Accepts {"messages": [...]} or {"conversations": [...]} (ShareGPT roles mapped)."""
    msgs = row.get("messages") or row.get("conversations")
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            return None
    tools = row.get("tools")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except Exception:
            tools = None
    if not isinstance(msgs, list):
        return None
    role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant",
                "system": "system", "tool": "tool", "function": "tool"}
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = role_map.get(m.get("role", m.get("from")))
        if role is None:
            continue
        c = m.get("content", m.get("value"))
        if isinstance(c, list):
            c = "\n".join(str(x.get("text", "")) for x in c if isinstance(x, dict) and x.get("type", "text") == "text")
        c = c if isinstance(c, str) else ""
        msg = {"role": role, "content": c}
        for k in ("tool_calls", "reasoning_content", "tools", "tool_call_id", "name"):
            if m.get(k):
                msg[k] = m[k]
        if not c.strip() and not msg.get("tool_calls") and not msg.get("reasoning_content") and role != "tool":
            continue
        out.append(msg)
    if len(out) < 2 or not any(m["role"] == "assistant" for m in out):
        return None
    if tools and not any("tools" in m for m in out):    # row-level tool schema -> the leading system turn
        tl = [t if "function" in t else {"type": "function", "function": t} for t in tools if isinstance(t, dict)]
        if out[0]["role"] == "system":                  # encoding.py renders tools only from a system message
            out[0]["tools"] = tl
        else:
            out.insert(0, {"role": "system", "content": "", "tools": tl})
    return out


_TAG = re.compile(r"<[^>]+>")
# Standard computational research only: hazardous wet-lab / pathogen / weapons material is dropped wherever it appears.
_HAZARD = re.compile(r"(gain[- ]of[- ]function|select agent|bioweapon|biological weapon|chemical weapon|nerve agent|"
                     r"toxin (production|synthesis)|enhanced (transmissib|virulen)|pandemic potential|"
                     r"weaponi[sz]|explosive synthesis|synthesi[sz]e (an? )?explosive)", re.I)


_Q = ("instruction", "prompt", "question", "query", "input", "problem", "user_prompt", "description")
_A = ("output", "response", "answer", "completion", "solution", "code", "target", "chosen")
_T = ("text", "content", "code", "func", "document")


def _auto(row):
    """Generic row -> document for HF sets whose columns vary: chat columns first (via _chat), then a
    question/answer pair, then a raw text/code column."""
    if row.get("messages") or row.get("conversations"):
        return _chat(row)
    q, a = _first(row, *_Q), _first(row, *_A)
    if q and a and q != a:
        inp = row.get("input") if isinstance(row.get("input"), str) and row.get("input") != q else ""
        return [{"role": "user", "content": q + ("\n\n" + inp if inp else "")}, {"role": "assistant", "content": a}]
    return _first(row, *_T)


def _strip_html(s: str) -> str:
    s = re.sub(r"<pre[^>]*><code>", "\n```\n", s)
    s = s.replace("</code></pre>", "\n```\n")
    return html.unescape(_TAG.sub("", s)).strip()


def _se(keep: str | None = None, min_score: int = 1):  # noqa: C901
    """Stack Exchange (HuggingFaceH4/stack-exchange-preferences): question -> best answer (accepted,
    else highest score) as a chat turn. `keep` is a regex the question or answer must match."""
    rx = re.compile(keep, re.I) if keep else None

    def fn(row):
        q, ans = row.get("question"), row.get("answers") or []
        if not q or not ans:
            return None
        best = max(ans, key=lambda a: (bool(a.get("selected")), a.get("pm_score") or 0))
        if (best.get("pm_score") or 0) < min_score and not best.get("selected"):
            return None
        q, a = _strip_html(q), _strip_html(best.get("text") or "")
        if len(a) < 80 or (rx and not rx.search(q + "\n" + a)) or _HAZARD.search(q + "\n" + a):
            return None
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return fn


_BAD_LIC = re.compile(r"(non-?commercial|\bnc\b|noderiv|no-?deriv|\bnd\b)", re.I)


def _arxiv(pattern: str, min_hits: int = 4):
    """common-pile/arxiv_papers (openly licensed arXiv): keep a paper whose text matches `pattern` at
    least `min_hits` times; the per-paper licence is re-checked (no NC/ND)."""
    rx = re.compile(pattern, re.I)

    def fn(row):
        t, md = row.get("text"), row.get("metadata") or {}
        lic = md.get("license", "") if isinstance(md, dict) else str(md)
        if not t or _BAD_LIC.search(lic or ""):
            return None
        head = t[:60_000]
        if _HAZARD.search(head):
            return None
        return t if len(rx.findall(head)) >= min_hits else None
    return fn


def _kernelbook(row):
    """PyTorch module + its inductor Triton kernel (permissive subset); the row's own licence is kept."""
    py, tr = row.get("python_code"), row.get("triton_code")
    if not py or not tr or _BAD_LIC.search(str(row.get("licenses", ""))):
        return None
    return f"# PyTorch reference\n{py}\n\n# Triton kernel (torch inductor)\n{tr}"


def _tooluse(pattern: str | None = None):
    """Tool-calling agentic rows only (structured `tool_calls`, rendered as DSML by the checkpoint encoder through
    _chat, with the row's tool schema on the system turn). `pattern` keeps rows whose tool schema or first user turn
    matches the bucket (case-insensitive); None keeps every tool-calling row."""
    rx = re.compile(pattern, re.I) if pattern else None

    def fn(row):
        msgs = _chat(row)
        if not msgs or not any(m.get("tool_calls") for m in msgs):
            return None
        if rx is not None:
            u = next((m["content"] for m in msgs if m["role"] == "user"), "")
            tl = row.get("tools")
            hay = (tl if isinstance(tl, str) else json.dumps(tl, default=str))[:20000] + "\n" + u[:4000]
            if not rx.search(hay):
                return None
        return msgs
    return fn


SW_MAX_CHARS = 60_000           # s02 caps one document at 2 rows (16k tokens) anyway


# ----------------------------------------------------------------------------- custom streams
def _hf_files(src):
    """A HF dataset restricted to `data_files` (per-site / per-shard), or a builder `loader` over hub files."""
    from datasets import load_dataset
    kw = {"split": src.get("hf_split", "train"), "streaming": True, "data_files": src["data_files"]}
    return iter(load_dataset(src.get("loader") or src["hf_id"], **kw))


def _hf_jsonl(src):
    """Raw line reader over one hub jsonl file (HfFileSystem). For large agentic sets whose rows do not share one
    arrow schema (datasets' streaming cast fails past the first chunk on Nemotron-SFT-Agentic-v2)."""
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f"datasets/{src['hf_id']}/{src['jsonl']}", "r", block_size=8 << 20) as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except Exception:
                continue


def git_paths(src: dict) -> tuple[str, str]:
    repo, ref = src["hf_id"][4:].split("@")
    return os.path.expanduser(repo), ref


def select_files(src: dict) -> list[tuple[str, str, str]]:
    """(hash, path, oid) of every candidate file of the ref, in the fixed hash order. For a `lazy`
    source (blob:none clone of a huge tree) the list is capped at `max_files`; prep.py prefetches
    exactly these blobs in batches."""
    repo, ref = git_paths(src)
    ls = subprocess.run(["git", "-C", repo, "ls-tree", "-r", ref], capture_output=True, text=True, check=True).stdout
    inc, exts, exc = src.get("include") or ("",), src.get("exts") or REPO_EXTS, src.get("exclude") or ()
    files = []
    for line in ls.splitlines():
        meta, path = line.split("\t", 1)
        mode, typ, oid = meta.split()
        if typ != "blob" or not path.endswith(tuple(exts)):
            continue
        if not any(path.startswith(p) for p in inc) or any(x in path for x in exc):
            continue
        files.append((hashlib.sha1(path.encode()).hexdigest(), path, oid))
    files.sort()
    return files[: src["max_files"]] if src.get("lazy") else files


def missing_blobs(repo: str, ref: str) -> set[str]:
    r = subprocess.run(["git", "-C", repo, "rev-list", "--objects", "--missing=print", ref],
                       capture_output=True, text=True)
    return {line[1:].strip() for line in r.stdout.splitlines() if line.startswith("?")}


def _git_files(src):
    """Per-file documents of an upstream ref: paths under `include`, extensions in `exts`, visited in
    a fixed hash order (a sample across the tree, not the alphabetically-first directory); blobs the
    partial clone does not hold (> blob limit, or not prefetched) are skipped, never fetched."""
    repo, ref = git_paths(src)
    missing = missing_blobs(repo, ref)
    files = [f for f in select_files(src) if f[2] not in missing]
    p = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for _, path, oid in files:
            p.stdin.write(oid.encode() + b"\n"); p.stdin.flush()
            hdr = p.stdout.readline().split()
            if len(hdr) < 3 or hdr[1] != b"blob":
                continue
            data = p.stdout.read(int(hdr[2])); p.stdout.read(1)
            if len(data) > FILE_MAX_BYTES and src.get("split_large") and b"\0" not in data[:4096]:
                # single-file specs (WHATWG `source`, CSS Overview.bs): ~48 KB line-aligned pieces, hash-ordered
                txt, pieces, pos = data.decode("utf-8", "replace"), [], 0
                while pos < len(txt):
                    end = txt.find("\n", pos + SPLIT_BYTES)
                    end = len(txt) if end < 0 else end + 1
                    pieces.append((hashlib.sha1(f"{path}#{len(pieces)}".encode()).hexdigest(), len(pieces), txt[pos:end]))
                    pos = end
                for _, k, piece in sorted(pieces):
                    yield {"text": f"// ===== {src['repo']}/{path} (part {k + 1}/{len(pieces)}) =====\n" + piece,
                           "_doc": f"{src['repo']}:{path}#{k}"}
                continue
            if not (FILE_MIN_BYTES <= len(data) <= FILE_MAX_BYTES) or b"\0" in data[:4096]:
                continue
            yield {"text": f"// ===== {src['repo']}/{path} =====\n" + data.decode("utf-8", "replace"),
                   "_doc": f"{src['repo']}:{path}"}
    finally:
        p.stdin.close(); p.kill()


_LIC_RULES = [
    (r"commons clause", "commons-clause"), (r"business source license", "busl"), (r"non-?commercial", "nc"), (r"noderivatives|no derivatives", "nd"),
    (r"gnu free documentation license", "gfdl"), (r"gnu affero general public license", "agpl-3.0"), (r"gnu lesser general public license", "lgpl"),
    (r"gnu general public license", "gpl"), (r"mozilla public license", "mpl-2.0"),
    (r"apache license", "apache-2.0"), (r"mit license|permission is hereby granted, free of charge", "mit"),
    (r"redistribution and use in source and binary forms", "bsd"), (r"cc0|creative commons zero|public domain", "cc0/pd"),
    (r"detection rule license", "drl-1.1"), (r"semgrep rules license", "semgrep-rules-1.0"),
    (r"solderpad", "solderpad"), (r"\bisc license|permission to use, copy, modify, and/or distribute", "isc"),
    (r"att&ck", "mitre-attack-tou"), (r"attribution-sharealike|share-?alike", "cc-by-sa"),
    (r"creative commons attribution 4\.0|cc-by-4\.0", "cc-by-4.0"),
]


# GPL/LGPL boilerplate that mentions non-commercial distribution of object code (GPLv3 6d "occasionally and
# noncommercially", GPLv2 3c "allowed only for noncommercial distribution"). It is copyleft, not an NC term; strip it
# before the NC rule so GPL-family licences classify as copyleft (allowed for private calibration, flagged).
_GPL_NC_BOILERPLATE = re.compile(
    r"occasionally\s+and\s+non-?commercially|allowed\s+only\s+for\s+non-?commercial\s+distribution"
    r"|only\s+for\s+non-?commercial\s+distribution\s+and\s+only\s+if\s+you\s+received")


def git_licence(repo: str, ref: str, ignore=()) -> str | None:
    """Live licence of a git ref: classify every top-level LICENSE*/COPYING*/README licence file (except `ignore`:
    files that describe something other than the code we read, e.g. Biomni's data-lake licence_info.md)."""
    ls = subprocess.run(["git", "-C", repo, "ls-tree", "--name-only", ref], capture_output=True, text=True).stdout.split()
    names = [n for n in ls if re.match(r"(?i)^(licen[sc]e|copying|unlicense)", n) and n not in ignore]
    tags = []
    for n in names:
        t = subprocess.run(["git", "-C", repo, "show", f"{ref}:{n}"], capture_output=True).stdout[:20_000]
        low = re.sub(r"\s+", " ", t.decode("utf-8", "replace").lower())
        if "general public license" in low or "free documentation license" in low:
            low = _GPL_NC_BOILERPLATE.sub(" ", low)
        low = re.sub(r"(either\s+)?commercially\s+or\s+non-?commercially", " ", low)   # a grant of both, never a restriction
        for rx, tag in _LIC_RULES:
            if re.search(rx, low) and tag not in tags:
                tags.append(tag)
    sha = subprocess.run(["git", "-C", repo, "rev-parse", ref], capture_output=True, text=True).stdout.strip()[:12]
    if not tags:
        return f"unknown ({','.join(names) or 'no licence file'} @ {sha})" if sha else None
    return f"{'+'.join(tags)} ({','.join(names)} @ {sha})"


def HF(hf_id, weight, *, licence, generator, config=None, split="train", status="new", fn=None, data_files=None,
       thinking=False):
    s = S(hf_id, config, split, weight, fn or _auto, licence=licence, generator=generator, status=status, thinking=thinking)
    if data_files:
        s.update(data_files=data_files, hf_split=split, stream=_hf_files)
    return s


# Stack Exchange, one rule for every bucket (coordinator 2026-11-01): only the pre-2024 H4 dump, CALIBRATION ONLY (never
# fine-tune data), every row flagged share-alike. No newer SE dump anywhere (flymin/stackexchange_20260331 is EXCLUDED).
SE_USE = "calibration-only (never fine-tune); share-alike"


def SE(site, weight, keep=None, licence="cc-by-sa-4.0 (Stack Exchange, H4 pre-2024 dump; SA)", tag=""):
    s = S("HuggingFaceH4/stack-exchange-preferences", site + (f":{tag}" if tag else ""), "train", weight, _se(keep), licence=licence,
          generator="human (Stack Exchange)", status="H4 pre-2024 dump: " + SE_USE)
    s["calibration_only"] = True          # s02 tags its docs {"sa", "calibration_only"} and never takes prompts from it
    s.update(data_files=f"data/{site}/*.parquet", stream=_hf_files)
    return s


def ARXIV(weight, pattern, min_hits=4, tag=""):
    s = S("common-pile/arxiv_papers", "2022-2025 shards" + (f":{tag}" if tag else ""), "train", weight, _arxiv(pattern, min_hits),
          licence="per-paper open licence (CC-BY / CC-BY-SA / CC0 / PD; any non-open per-paper licence dropped)", generator="human (arXiv)",
          status=f"new; keyword-filtered >= {min_hits} hits")
    s.update(data_files=[f"000{i}_arxiv-papers.jsonl.gz" for i in range(16, 22)], stream=_hf_files)
    return s


def G(repo, weight, include=("",), *, licence, exts=None, exclude=(), local=None, ref="HEAD", url=None,
      lazy=False, max_files=800, generator=None, blob_limit="512k", split_large=False):
    """A git source. `local` reuses an existing upstream clone (read at its remote ref only). `lazy`:
    a huge tree (Chromium, WebKit, Firefox, KiCad, FreeCAD) is cloned blob:none and only the first
    `max_files` selected blobs are prefetched. `url` overrides the GitHub clone URL."""
    path = local or str(DOM_SRC / (repo.replace("/", "__") + ".git"))
    s = S(f"git:{path}@{ref}", None, None, weight, None, licence=licence,
          generator=generator or f"human (upstream {repo})", status="git per-file" + (" (lazy subset)" if lazy else ""))
    s.update(repo=repo, include=tuple(include), exts=tuple(exts or CODE_EXTS), exclude=tuple(exclude), stream=_git_files,
             url=url or f"https://github.com/{repo}.git", lazy=lazy, max_files=max_files, blob_limit=blob_limit,
             split_large=split_large)
    return s


def L(path, weight, fn, *, licence, generator, status, thinking=False):
    return S(f"local:{path}", None, None, weight, fn, licence=licence, generator=generator, status=status, thinking=thinking)


def TOOL(bucket, weight, pattern):
    """Generic tool-calling rows (NVIDIA Nemotron-SFT-Agentic-v2, tool_calling split) filtered to the bucket."""
    s = S("nvidia/Nemotron-SFT-Agentic-v2", f"tool_calling:{bucket}", "train", weight, _tooluse(pattern),
          licence="cc-by-4.0 (card lists cc-by-4.0 / apache-2.0 / mit per subset)",
          generator="NVIDIA SDG agent trajectories (per-row metadata.sdg_model)",
          status="tool-calling DSML rows (coordinator 2026-11-01: >= 10% per bucket); bucket keyword filter on tools/user",
          thinking=True)
    s.update(jsonl="data/tool_calling.jsonl", stream=_hf_jsonl)
    return s


def TOOL_ANY(weight):
    """Unfiltered tool-calling fallback (NVIDIA Nemotron-Agentic-v1, a different set): takes any shortfall of TOOL."""
    s = S("nvidia/Nemotron-Agentic-v1", "tool_calling", "train", weight, _tooluse(None), licence="cc-by-4.0",
          generator="NVIDIA SDG agent trajectories (card)", status="tool-calling DSML rows; fallback, no keyword filter",
          thinking=True)
    s.update(jsonl="data/tool_calling.jsonl", stream=_hf_jsonl)
    return s


# Bucket keyword filters for the generic tool-calling rows (matched against the tool schema and first user turn).
TOOL_RX = {
    "gamedev": r"\b(game|gaming|player|sprite|quest|leaderboard|esports|tournament)\w*",
    "mlsys": r"\b(gpu|cuda|kernel|inference|training job|benchmark|cluster|machine learning|llm)\w*",
    "robotics": r"\b(robot|drone|navigation|sensor|motion|autonomous|vehicle|fleet)\w*",
    "quantum": r"\b(quantum|qubit|circuit|simulat|physics)\w*",
}
TOOL_SHARE = 0.12          # tool-use sub-area floor (rows); the hard check in s02 is >= 10% of the bucket's tokens


CODE_EXTS = (".rs", ".toml", ".md", ".wgsl", ".ron", ".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp",
             ".rst", ".txt", ".yml", ".yaml", ".yar", ".yara", ".ql", ".qll", ".qhelp",
             ".v", ".sv", ".svh", ".vh", ".vhd", ".scala", ".tcl", ".go", ".ts", ".mdx", ".adoc")
RS = (".rs", ".md", ".toml", ".wgsl")
CU = (".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".py", ".md", ".rst")
THREADS_STATUS = "ops/dom/prep.py threads (GitHub GraphQL: recent closed issues + answered discussions)"
GH_LIC = "github user content (ToS; private calibration only)"


# Owner repos (optional, `local:` rows built by `prep.py localrepos`). Every commit dated on/after `cutoff` on ANY ref is
# reserved for evaluation and never enters calibration or training rows; file rows are read from the last pre-cutoff
# commit of `ref`. s02 records this dict in the corpus manifest. The original run used private repositories here;
# they are not part of this release, so `repos` is empty. Entry format:
#   "<name>": {"root": "<path to a git checkout>", "ref": "refs/heads/main", "bucket": "<bucket>",
#              "eval_exclude": ["<path prefix never read>", ...]}
OWNER_HOLDOUT = {"cutoff": "2026-10-15", "eval": "owner hold-out eval", "repos": {}}
OWNER_STATUS = "ops/dom/prep.py localrepos; git, pre-cutoff only (OWNER_HOLDOUT); "


def _transcripts(domain, w):
    """Private agent-session rows (`local:dom/transcripts/<domain>.jsonl`, OpenAI-format messages with tool calls).
    The original run used private owner transcripts here; they are not released and no bucket below lists them."""
    return L(f"dom/transcripts/{domain}.jsonl", w, _chat, licence="owner (private)",
             generator="claude (agent transcripts)", status="private; not part of this release", thinking=True)


def _threads(domain, w):
    """GitHub issue/discussion threads of THREAD_REPOS[domain] as chat rows (`prep.py threads`). Not listed in any
    bucket of this release; add `_threads(<bucket>, <weight>)` to a bucket to use them."""
    return L(f"dom/threads/{domain}.jsonl", w, _chat, licence=GH_LIC, generator="human (GitHub threads)",
             status=THREADS_STATUS)


# GitHub repos whose most-discussed closed issues (and answered discussions) become chat rows (prep.py threads).
THREAD_REPOS = {
    "gamedev": ["bevyengine/bevy", "avianphysics/avian", "Leafwing-Studios/leafwing-input-manager",
                "jakobhellermann/bevy-inspector-egui", "StarArawn/bevy_ecs_tilemap", "dimforge/rapier"],
    "mlsys": ["vllm-project/vllm", "ggml-org/llama.cpp", "flashinfer-ai/flashinfer", "NVIDIA/cutlass",
              "triton-lang/triton", "sgl-project/sglang", "HazyResearch/ThunderKittens"],
    "robotics": ["ros-navigation/navigation2", "moveit/moveit2", "ros-controls/ros2_control", "google-deepmind/mujoco",
                 "mujocolab/mjlab", "isaac-sim/IsaacLab", "huggingface/lerobot", "Physical-Intelligence/openpi", "ompl/ompl",
                 "stack-of-tasks/pinocchio"],
    "quantum": ["Qiskit/qiskit", "quantumlib/Cirq", "PennyLaneAI/pennylane", "quantumlib/Stim", "oscarhiggott/PyMatching",
                "Qiskit/qiskit-aer", "openqasm/openqasm"],
}


def _tasks_table_stream(src):
    """LeRobot v3 `meta/tasks.parquet` (task strings are the pandas index column) -> text tables of `per` unique
    natural-language task instructions each (the only text a text-only spine can use from a VLA episode set)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f"datasets/{src['hf_id']}/{src['pq']}", "rb", block_size=8 << 20) as fh:
        tab = pq.read_table(fh)
    col = "task" if "task" in tab.column_names else next(c for c in tab.column_names if c != "task_index")
    seen, tasks = set(), []
    for t in tab.column(col).to_pylist():
        t = (t or "").strip()
        if len(t) >= 8 and t.lower() not in seen:
            seen.add(t.lower())
            tasks.append(t)
    per = src.get("per", 200)
    for i in range(0, len(tasks), per):
        body = "\n".join(f"- {t}" for t in tasks[i:i + per])
        yield {"text": f"# {src['hf_id']}: natural-language task instructions {i + 1}-{min(i + per, len(tasks))} "
                       f"of {len(tasks)} ({src.get('embodiment', 'robot')})\n{body}", "_doc": f"{src['hf_id']}#{i}"}


def _behavior_stream(src):
    """BEHAVIOR-1K 2026 challenge demos: per-episode skill/primitive annotations (JSON) -> a readable plan
    (task, ordered skills with objects and skill types, primitives). `per_task` episodes from each task folder."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    root = f"datasets/{src['hf_id']}/annotations"
    for task_dir in sorted(fs.ls(root, detail=False)):
        if not task_dir.rsplit("/", 1)[-1].startswith("task-"):
            continue
        for ep in sorted(fs.ls(task_dir, detail=False))[: src.get("per_task", 2)]:
            try:
                d = json.loads(fs.read_text(ep))
            except Exception:
                continue
            lines = [f"# BEHAVIOR-1K task `{d.get('task_name', '?')}` ({ep.rsplit('/', 2)[-2]}/{ep.rsplit('/', 1)[-1]}), "
                     f"{(d.get('meta_data') or {}).get('task_duration', '?')} frames", "## skills"]
            for s in d.get("skill_annotation") or []:
                lines.append(f"{s.get('skill_idx')}. {', '.join(s.get('skill_description') or [])} "
                             f"objects={s.get('object_id')} manipulating={s.get('manipulating_object_id')} "
                             f"type={s.get('skill_type')} frames={s.get('frame_duration')}"
                             + (f" memory={s['memory_prefix']}" if s.get("memory_prefix") else "")
                             + (f" spatial={s['spatial_prefix']}" if s.get("spatial_prefix") else ""))
            lines.append("## primitives")
            for p in d.get("primitive_annotation") or []:
                lines.append(f"{p.get('primitive_idx')}. {', '.join(p.get('primitive_description') or [])} "
                             f"objects={p.get('object_id')} skills={p.get('skill_idxes')}")
            yield {"text": "\n".join(lines), "_doc": ep}


_SYSREM = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def _cc_blocks_text(c, limit=6000):
    """Claude Code message content (str or blocks) -> text; images -> [image]; long tool output clipped."""
    if isinstance(c, str):
        out = c
    elif isinstance(c, list):
        parts = []
        for b in c:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(str(b.get("text") or ""))
            elif b.get("type") == "image":
                parts.append("[image]")
        out = "\n".join(parts)
    else:
        out = ""
    out = _SYSREM.sub("", out).strip()
    return out if len(out) <= limit else out[:limit] + f"\n[... {len(out) - limit} chars clipped]"


def _cc_messages(lines):
    """Claude Code JSONL events (file order) -> OpenAI messages: assistant events sharing message.id merge (text ->
    content, visible thinking -> reasoning_content, tool_use -> tool_calls), tool_result -> role tool, meta events and
    empty/encrypted thinking dropped."""
    msgs, cur_id = [], None
    for ev in lines:
        t, m = ev.get("type"), ev.get("message") or {}
        if t not in ("user", "assistant") or ev.get("isMeta"):
            continue
        c = m.get("content")
        if t == "assistant":
            if m.get("id") != cur_id or not msgs or msgs[-1]["role"] != "assistant":
                msgs.append({"role": "assistant", "content": ""})
                cur_id = m.get("id")
            a = msgs[-1]
            for b in c if isinstance(c, list) else [{"type": "text", "text": c or ""}]:
                if b.get("type") == "text" and (b.get("text") or "").strip():
                    a["content"] = (a["content"] + "\n" + b["text"]).strip()
                elif b.get("type") == "thinking" and isinstance(b.get("thinking"), str) and b["thinking"].strip():
                    a["reasoning_content"] = (a.get("reasoning_content", "") + "\n" + b["thinking"]).strip()
                elif b.get("type") == "tool_use":
                    a.setdefault("tool_calls", []).append(
                        {"id": b.get("id"), "type": "function",
                         "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input") or {})}})
            continue
        cur_id = None
        if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                                 "content": _cc_blocks_text(b.get("content")) or "(empty)"})
            continue
        txt = _cc_blocks_text(c, limit=20000)
        if txt and not txt.startswith(("<command-", "<local-command", "[Request interrupted")):
            msgs.append({"role": "user", "content": txt})
    return [m for m in msgs if m["role"] != "assistant" or m["content"] or m.get("tool_calls") or m.get("reasoning_content")]


def _cc_chunks(msgs, max_chars=36_000):
    """Split a long session into chunks that never separate a tool call from its results; every chunk after the first
    re-states the session's opening user turn as context."""
    first = next((m["content"] for m in msgs if m["role"] == "user"), "")
    chunk, size, out = [], 0, []
    for i, m in enumerate(msgs):
        n = len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or ""))
        boundary = m["role"] == "user" or (m["role"] == "assistant" and i and msgs[i - 1]["role"] in ("tool", "user"))
        if chunk and size + n > max_chars and boundary:
            out.append(chunk)
            chunk, size = ([] if m["role"] == "user" else
                           [{"role": "user", "content": f"(continuing the session; original task)\n{first[:4000]}"}]), 0
        chunk.append(m)
        size += n
    if chunk:
        out.append(chunk)
    return out


def _rax_stream(src):
    """RAX robot-session-logs: Claude Code transcripts (main + subagents) with real MCP robot tool calls, read from the
    uncompressed per-session tars (streamed sequentially, smallest first; camera-video tars never opened)."""
    import tarfile
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    tx = re.compile(r"claude-home/projects/[^/]+/(?:[^/]+/subagents/(?:[^/]+/)*)?[^/]+\.jsonl$")
    for sess in src["sessions"]:
        name = sess.rsplit("/", 1)[-1]
        with fs.open(f"datasets/{src['hf_id']}/{sess}/{name}.tar", "rb", block_size=16 << 20) as fh:
            tf = tarfile.open(fileobj=fh, mode="r|")
            for mem in tf:
                if not (mem.isfile() and tx.search(mem.name)):
                    continue
                raw = tf.extractfile(mem).read().decode("utf-8", "replace")
                lines = []
                for line in raw.splitlines():
                    try:
                        lines.append(json.loads(line))
                    except Exception:
                        continue
                for k, ch in enumerate(_cc_chunks(_cc_messages(lines))):
                    yield {"messages": ch, "_doc": f"{name}:{mem.name.rsplit('/', 1)[-1]}#c{k}"}


SOURCES: dict[str, list[dict]] = {
    "gamedev": [
        SE("gamedev.stackexchange.com", 0.06, keep=r"\b(rust|bevy|ecs|entity component|wgpu|shader|collision|physics)\b"),
        G("bevyengine/bevy", 0.16, ("examples/", "crates/", "docs/", "release-content/"), licence="mit OR apache-2.0", exts=RS),
        G("avianphysics/avian", 0.04, licence="mit OR apache-2.0", exts=RS),
        G("Leafwing-Studios/leafwing-input-manager", 0.03, licence="mit OR apache-2.0", exts=RS),
        G("vladbat00/bevy_egui", 0.02, licence="mit", exts=RS),
        G("NiklasEi/bevy_asset_loader", 0.02, licence="mit OR apache-2.0", exts=RS),
        G("StarArawn/bevy_ecs_tilemap", 0.02, licence="mit", exts=RS),
        G("jakobhellermann/bevy-inspector-egui", 0.02, licence="mit OR apache-2.0", exts=RS),
        G("TheBevyFlock/bevy_new_2d", 0.02, licence="mit OR apache-2.0", exts=RS),
        G("rust-lang/book", 0.06, ("src/", "listings/"), licence="mit OR apache-2.0", exts=RS),
        G("rust-lang/reference", 0.03, ("src/",), licence="mit OR apache-2.0", exts=(".md",)),
        G("rust-lang/rust-by-example", 0.03, ("src/",), licence="mit OR apache-2.0", exts=(".md",)),
        G("rust-lang/nomicon", 0.02, ("src/",), licence="mit OR apache-2.0", exts=(".md",)),
        G("rust-lang/rust", 0.04, ("library/core/src/", "library/std/src/", "library/alloc/src/"),
          licence="mit OR apache-2.0", exts=(".rs",), exclude=("/tests/", "/benches/")),
    ],
    # reconciled 2026-11-01 against docs/domain/sources/mlsys.md (rows 1-12)
    "mlsys": [
        HF('nvidia/Nemotron-SFT-CUDA-v1', 0.06, licence='cc-by-4.0', generator='NVIDIA pipeline (card: messages + tools)', config=None, split='train', status='new (sources/mlsys.md 2)', fn=_auto),
        HF('hkust-nlp/drkernel-coldstart-8k', 0.04, licence='mit', generator='LLM multi-turn with KernelGYM feedback', config=None, split='train', status='new (sources/mlsys.md 4)', fn=_auto),
        G("NVIDIA/cutlass", 0.07, ("include/cute/", "examples/cute/", "media/docs/cpp/", "examples/python/CuTeDSL/"),
          licence="bsd-3-clause", local="~/cutlass", ref="origin/main", exts=CU),   # python/CuTeDSL + docs/pythonDSL are EULA: excluded
        G("ggml-org/llama.cpp", 0.05, ("ggml/src/ggml-cuda/", "docs/"), licence="mit", local="~/llama.cpp",
          ref="origin/master", exts=CU),
        G("flashinfer-ai/flashinfer", 0.04, ("include/", "csrc/", "flashinfer/", "docs/"), licence="apache-2.0", exts=CU),
        G("vllm-project/vllm", 0.04, ("csrc/", "vllm/model_executor/layers/", "vllm/v1/attention/", "docs/design/"),
          licence="apache-2.0", exts=CU),
        G("sgl-project/sglang", 0.03, ("sgl-kernel/", "python/sglang/srt/layers/", "docs/"), licence="apache-2.0", exts=CU),
        G("Dao-AILab/flash-attention", 0.03, ("hopper/", "csrc/flash_attn/src/", "flash_attn/cute/"), licence="bsd-3-clause",
          exts=CU),
        G("NVIDIA/TensorRT-LLM", 0.02, ("cpp/tensorrt_llm/kernels/", "docs/source/"), licence="apache-2.0", exts=CU + (".md",),
          lazy=True, max_files=400),
        G("triton-lang/triton", 0.04, ("python/tutorials/", "python/triton/language/", "docs/", "python/triton_kernels/"),
          licence="mit", exts=CU),
        G("tile-ai/tilelang", 0.03, ("examples/", "docs/", "tilelang/language/"), licence="mit", exts=CU),
        G("HazyResearch/ThunderKittens", 0.03, licence="mit", exts=CU),
        G("NVIDIA/cutile-python", 0.02, licence="apache-2.0", exts=CU),
        G("deepseek-ai/DeepGEMM", 0.03, licence="mit", exts=CU),
        G("deepseek-ai/FlashMLA", 0.02, licence="mit", exts=CU),
        G("deepseek-ai/DeepEP", 0.02, licence="mit", exts=CU),
        G("pytorch/pytorch", 0.03, ("docs/source/notes/", "torch/_inductor/", "torch/distributed/", "aten/src/ATen/native/cuda/"),
          licence="bsd-3-clause (pytorch)", exts=CU + (".md",), exclude=("/test",)),
        G("pytorch/torchtitan", 0.02, licence="bsd-3-clause", exts=(".py", ".md")),
        G("NVIDIA/Megatron-LM", 0.02, ("megatron/core/",), licence="bsd-3-clause-style (NVIDIA)", exts=(".py", ".md")),
        G("verl-project/verl", 0.02, ("verl/", "docs/"), licence="apache-2.0", exts=(".py", ".md", ".rst")),
        G("pytorch/ao", 0.02, ("torchao/", "docs/"), licence="bsd-3-clause", exts=CU, exclude=("/test",)),
        G("fla-org/flash-linear-attention", 0.02, ("fla/",), licence="mit", exts=(".py", ".md")),
        G("vllm-project/llm-compressor", 0.01, ("src/", "docs/", "examples/"), licence="apache-2.0", exts=(".py", ".md")),
        G("NVIDIA/Model-Optimizer", 0.01, ("modelopt/", "docs/"), licence="apache-2.0", exts=(".py", ".md", ".rst")),
        G("SafeAILab/EAGLE", 0.01, licence="apache-2.0", exts=(".py", ".md")),
        G("jax-ml/scaling-book", 0.02, licence="mit", exts=(".md",)),
        G("modal-labs/gpu-glossary", 0.01, licence="mit", exts=(".md",)),
        G("gpu-mode/lectures", 0.01, licence="apache-2.0", exts=CU),
        S("GPUMODE/KernelBook", None, "train", 0.02, _kernelbook, licence="other (dataset_permissive: per-row permissive)",
          generator="human + torch inductor", status="reused (original code bucket)"),
        S("SakanaAI/AI-CUDA-Engineer-Archive", None, "level_2", 0.01, _text("CUDA_Code"), licence="cc-by-4.0",
          generator="Sakana AI CUDA Engineer (agent)", status="reused (original code bucket used level_1)"),
        ARXIV(0.06, r"\b(CUDA|GPU kernel|tensor core|FlashAttention|quantiz\w+|KV cache|mixture[- ]of[- ]experts|"
                    r"speculative decoding|inference throughput|Triton|pipeline parallel\w*|tensor parallel\w*)\b", 6),
    ],
    # reconciled 2026-11-01 against docs/domain/sources/robotics.md (A1-A18). Weights are catalogue shares (% / 100);
    # the A16 claude-session 6% (not yet generated) is redistributed as the catalogue says (SE +2, arXiv +2, planning +2).
    # S5 manipulation + S8 VLA (tied smallest at 8%) merge into `manipulation-vla` to stay within 10 sub-areas.
    # ADDED: mujoco_warp, Newton, gz-sim, robot_descriptions.py, wuji-description, allex_model, mujoco-skills, urdf-skill;
    # docs.nav2.org, control.ros.org, ros2_control_demos, ros2-mcp, rosclaw, ros2-agent-workflow; cuRobo, mink, pink,
    # pybullet-planning, robotics-toolbox-python; crocoddyl, Drake, CasADi, underactuated (code only); manipulation (code
    # only), cap-x, CaPtain, CaPtain-Harness, HexaShell, RoboHarness, RAX harness, Isaac-GR00T, AgenticRoboRL, DROID task
    # instructions, BEHAVIOR-1K 2026 skill annotations; mjlab_playground, foothold-prediction-model, Playground locomotion;
    # mjlab, Eureka (with its GPT-4 reward outputs); robocode, kindergarden, BTGenBot-2, btgenbot, BehaviorTree.CPP,
    # unified-planning, pddlstream; RAX robot-session-logs (Claude Code + MCP robot tool calls) -> tool-use; arXiv split
    # into VLA / locomotion / sim-to-real slices; SE split by site (robotics -> ROS 2, engineering -> control, dsp ->
    # estimation) so no Q&A row lands in two sub-areas.
    # REJECTED: emgena ROS 2 MCP "teasers" (probed: Gumroad advert rows, template outputs, per-row commercial EULA vs
    # apache card); OXE mirror jxu124/OpenX-Embodiment (RLDS TFRecord tars, instructions need TF decoding); RoboTwin2.0
    # (zip archives); Open-H-Embodiment (100k per-episode files, task text only in per-subset READMEs);
    # tencent/Hy-Embodied-0.5-VLA-Data (probed: 22 identical 993-row Chinese task lists, negligible text);
    # text2reward and rax skills-lib (no licence); CodeBotler/Robo-Instruct (ML-training ban); NC per the licence flags:
    # AgiBot World 2026 (+challenge), Galaxea, GR00T-Teleop-Sim, Locomanipulation-GRAIL, loco-mujoco-datasets,
    # URDF-Anything+, LIBERO-Pro2.0-Train. Dropped from the provisional block (not in robotics.md): octo, ros2/rclpy,
    # PX4 (covered by another bucket of the original domain set).
    "robotics": [
        # --- tool-use (bucket-specific structured tool calls; generic Nemotron rows are auto-inserted)
        S("robot-agent-exploration/robot-session-logs", "transcripts", "train", 0.04, _tooluse(None), licence="cc-by-4.0",
          generator="claude (third-party Claude Code, claude-opus-5-5, RAX sessions; real + simulated robots)",
          status="new (robotics.md A11); text only (images -> [image]), smallest session tars first", thinking=True)
        | {"stream": _rax_stream, "sessions": [
            "rby1/20261008T022520Z_pnp_box_249b23", "rby1/20261008T212446Z_pnp_box_f7b297",
            "rby1/20261009T151644Z_pnp_orange_basket_54c973", "rby1/20261009T023700Z_practice_pnp_box_61befa",
            "rby1/20261009T000539Z_pnp_box_f75f61", "franka/20261004T131834Z_air_fry_pan_dbb6a4",
            "rby1/20261009T185605Z_push_chair_4ab505", "franka/20261006T132109Z_swipe_table_9f3e18",
            "rby1/20261010T011251Z_compositional_manipulation_aa1adf", "franka/20261007T024357Z_peg_insertion_cc65b7"]},
        # --- locomotion and whole-body control (S6 8%; listed before sim-scene/rl-rewards so its repos match first)
        G("mujocolab/mjlab_playground", 0.015, ("src/",), licence="apache-2.0", exts=(".py", ".md")),
        G("leggedrobotics/foothold-prediction-model", 0.01, ("fpm/", "scripts/"), licence="mit", exts=(".py", ".md")),
        G("google-deepmind/mujoco_playground", 0.015, ("mujoco_playground/_src/locomotion/", "learning/"),
          licence="apache-2.0", exts=(".py", ".md")),
        ARXIV(0.03, r"\b(locomotion|legged|quadruped|humanoid|whole-body control|gait|motion retarget\w*|biped\w*)\b", 8,
              tag="locomotion-wbc"),
        # --- simulators and scene description (S1 12%)
        G("google-deepmind/mujoco", 0.02, ("doc/", "python/mujoco/", "model/"), licence="apache-2.0",
          exts=(".rst", ".md", ".py", ".xml")),
        G("google-deepmind/mujoco_warp", 0.012, ("mujoco_warp/", "notebooks/"), licence="apache-2.0", exts=(".py", ".md")),
        G("google-deepmind/mujoco_menagerie", 0.01, licence="apache-2.0 (per-model licences vary; see each LICENSE)",
          exts=(".md", ".xml")),
        G("newton-physics/newton", 0.015, ("newton/examples/", "docs/"), licence="apache-2.0", exts=(".py", ".md", ".rst")),
        G("Genesis-Embodied-AI/genesis-world", 0.012, ("genesis/", "examples/"), licence="apache-2.0", exts=(".py", ".md")),
        G("gazebosim/gz-sim", 0.012, ("tutorials/", "examples/worlds/", "src/systems/"), licence="apache-2.0",
          exts=(".md", ".sdf", ".cc", ".hh"), lazy=True, max_files=300),
        G("robot-descriptions/robot_descriptions.py", 0.006, ("robot_descriptions/", "examples/"), licence="apache-2.0",
          exts=(".py", ".md")),
        G("wuji-technology/wuji-description", 0.006, ("hand/", "hand2/", "glove/"), licence="mit",
          exts=(".urdf", ".xml", ".md", ".xacro"), lazy=True, max_files=150),
        G("wirobotics-rih/allex_model", 0.007, ("mjcf/", "urdf/", "examples/"), licence="bsd-3-clause",
          exts=(".xml", ".urdf", ".py", ".md"), lazy=True, max_files=150),
        G("coolbeevip/mujoco-skills", 0.012, ("skills/", "examples/"), licence="mit", exts=(".md", ".py", ".xml"),
          generator="human/LLM-assisted skill files (UNVERIFIED split)"),
        G("earthtojake/urdf-skill", 0.008, ("references/", "scripts/", "agents/"), licence="mit", exts=(".md", ".py"),
          generator="human/LLM-assisted skill files (UNVERIFIED split)"),
        # --- ROS 2 and middleware (S2 13%)
        SE("robotics.stackexchange.com", 0.04, tag="ros2-middleware"),
        G("ros2/ros2_documentation", 0.027, ("source/",), licence="cc-by-4.0", exts=(".rst", ".md")),
        G("ros-navigation/docs.nav2.org", 0.015, ("docs/",), licence="apache-2.0", exts=(".md",), lazy=True, max_files=300),
        G("ros-controls/control.ros.org", 0.015, ("doc/",), licence="apache-2.0", exts=(".rst", ".md"), lazy=True, max_files=300),
        G("ros-controls/ros2_control_demos", 0.012, licence="apache-2.0",
          exts=(".cpp", ".hpp", ".py", ".xacro", ".yaml", ".md", ".rst")),
        G("ros-navigation/navigation2", 0.013, ("nav2_", "doc/"), licence="apache-2.0 + lgpl-2.1+ portions",
          exts=(".cpp", ".hpp", ".py", ".md", ".xml"), lazy=True, max_files=400),
        G("gazebosim/docs", 0.008, licence="cc-by-4.0", exts=(".md",)),
        G("mergeos-bounties/ros2-mcp", 0.006, ("src/", "skills/", "docs/", "examples/"), licence="mit", exts=(".py", ".md")),
        G("ros-claw/rosclaw", 0.008, ("packages/", "docs/", "tutorials/", "src/"), licence="mit", exts=(".py", ".md"),
          lazy=True, max_files=300),
        G("TmxjTmxj/ros2-agent-workflow", 0.006, ("agent_ros/", "mcp_server/", "skills/", "docs/"), licence="mit",
          exts=(".py", ".md", ".yaml")),
        # --- motion planning and kinematics (S3 10%)
        G("ompl/ompl", 0.015, ("src/ompl/geometric/planners/", "doc/markdown/", "demos/"), licence="bsd-3-clause",
          exts=(".cpp", ".h", ".md", ".py")),
        G("moveit/moveit2", 0.015, ("moveit_core/", "moveit_ros/planning/", "doc/"), licence="bsd-3-clause",
          exts=(".cpp", ".hpp", ".py", ".md"), lazy=True, max_files=400),
        G("moveit/moveit2_tutorials", 0.01, ("doc/",), licence="bsd-3-clause", exts=(".rst", ".py", ".cpp")),
        G("NVlabs/curobo", 0.015, ("curobo/", "docs/"), licence="apache-2.0", exts=(".py", ".md", ".rst")),
        G("kevinzakka/mink", 0.01, ("src/", "examples/", "docs/"), licence="apache-2.0", exts=(".py", ".rst", ".md")),
        G("pink-kinematics/pink", 0.008, ("pink/", "examples/", "doc/"), licence="apache-2.0", exts=(".py", ".md", ".rst")),
        G("caelan/pybullet-planning", 0.007, ("pybullet_tools/", "examples/"), licence="mit", exts=(".py",)),
        G("petercorke/robotics-toolbox-python", 0.02, ("src/roboticstoolbox/", "docs/source/"), licence="mit",
          exts=(".py", ".rst", ".md"), lazy=True, max_files=300),
        # --- control, MPC, trajectory optimisation (S4 12%)
        SE("engineering.stackexchange.com", 0.02, keep=r"\b(pid|controller|control loop|state[- ]space|transfer function|"
                                                      r"mpc|model predictive|lqr|servo|motor control|feedback|bode)\b",
           tag="control-mpc"),
        G("acados/acados", 0.015, ("examples/acados_python/", "docs/", "interfaces/acados_template/acados_template/"),
          licence="bsd-2-clause", exts=(".py", ".md", ".rst")),
        G("casadi/casadi", 0.01, ("docs/examples/python/", "docs/"), licence="lgpl-3.0", exts=(".py", ".md", ".rst")),
        G("loco-3d/crocoddyl", 0.015, ("examples/", "bindings/python/crocoddyl/", "doc/"), licence="bsd-3-clause",
          exts=(".py", ".md")),
        G("stack-of-tasks/pinocchio", 0.015, ("include/pinocchio/algorithm/", "examples/", "doc/"), licence="bsd-2-clause",
          exts=(".hpp", ".hxx", ".py", ".md")),
        G("RobotLocomotion/drake", 0.015, ("examples/", "tutorials/", "multibody/plant/", "systems/controllers/"),
          licence="bsd-3-clause (Drake LICENSE.TXT)", exts=(".py", ".cc", ".h", ".md"), lazy=True, max_files=400),
        G("python-control/python-control", 0.01, ("control/", "doc/", "examples/"), licence="bsd-3-clause",
          exts=(".py", ".rst"), exclude=("/tests/",)),
        G("RussTedrake/underactuated", 0.02, ("underactuated/",), licence="bsd-3-clause (code only; book prose excluded)",
          exts=(".py",)),
        # --- manipulation + VLA / embodied foundation models (S5 8% + S8 8%, merged)
        G("RussTedrake/manipulation", 0.02, ("manipulation/",), licence="bsd-3-clause (code only; book prose excluded)",
          exts=(".py",)),
        G("HexaFuture/HexaShell", 0.006, ("hexashell/", "library/", "recipes/", "tasks/"), licence="apache-2.0",
          exts=(".py", ".md", ".yaml")),
        G("BinceQu/RoboHarness", 0.007, ("harness/", "prompt/", "tasks/", "docs/"), licence="mit", exts=(".py", ".md", ".yaml")),
        G("robot-agent-exploration/rax", 0.007, ("docs/", "skills/"), licence="apache-2.0", exts=(".md", ".py")),
        S("lerobot/droid_1.0.1", "tasks", "train", 0.015, _text("text"), licence="apache-2.0",
          generator="human (DROID teleoperation task annotations)", status="new (robotics.md A14); meta/tasks.parquet only")
        | {"stream": _tasks_table_stream, "pq": "meta/tasks.parquet", "per": 200, "embodiment": "Franka Panda, DROID"},
        S("behavior-1k/2026-challenge-demos", "annotations", "train", 0.015, _text("text"), licence="mit",
          generator="human (BEHAVIOR-1K skill/primitive annotations)", status="new (robotics.md A14); annotations/ JSON only")
        | {"stream": _behavior_stream, "per_task": 2},
        G("Physical-Intelligence/openpi", 0.008, ("src/openpi/", "examples/", "docs/"), licence="apache-2.0", exts=(".py", ".md")),
        G("NVIDIA/Isaac-GR00T", 0.009, ("gr00t/", "getting_started/", "examples/"), licence="apache-2.0", exts=(".py", ".md")),
        G("huggingface/lerobot", 0.01, ("src/lerobot/policies/", "src/lerobot/robots/", "docs/", "examples/"),
          licence="apache-2.0", exts=(".py", ".md", ".mdx")),
        G("xinxinliu-ai/AgenticRoboRL", 0.003, ("harnesses/", "docs/", "rl/", "sft/"), licence="apache-2.0",
          exts=(".py", ".md")),
        ARXIV(0.02, r"\b(vision-language-action|VLA|robot foundation model|imitation learning|diffusion policy|"
                    r"teleoperat\w+|language-conditioned|grasp\w*|dexterous manipulation|contact-rich)\b", 8,
              tag="manipulation-vla"),
        # --- reward design and RL training code (S7 10%; catalogue keeps S7 at 8% when A16 is off)
        G("isaac-sim/IsaacLab", 0.03, ("source/isaaclab_tasks/", "docs/", "scripts/tutorials/"), licence="bsd-3-clause",
          exts=(".py", ".rst", ".md")),
        G("mujocolab/mjlab", 0.025, ("src/mjlab/tasks/", "src/mjlab/envs/", "docs/"), licence="apache-2.0",
          exts=(".py", ".md", ".rst")),
        G("eureka-research/Eureka", 0.025, ("eureka/",), licence="mit", exts=(".py", ".txt", ".md"),
          generator="human + GPT-4 (Eureka prompts and generated reward functions)"),
        # --- robot task planning and agentic code generation (S9 10%)
        G("Princeton-Robot-Planning-and-Learning/kindergarden", 0.01, ("src/", "docs/"), licence="mit", exts=(".py", ".md"),
          lazy=True, max_files=300),
        G("AIRLab-POLIMI/BTGenBot-2", 0.012, ("dataset/", "model/", "training/"), licence="mit",
          exts=(".py", ".md", ".xml", ".json"), lazy=True, max_files=200),
        HF("AIRLab-POLIMI/btgenbot", 0.012, licence="mit", generator="LLM-generated task summaries of BehaviorTree.CPP XML (BTGenBot)",
           data_files="bt_dataset.json", status="new (robotics.md A17)", fn=_auto),
        G("BehaviorTree/BehaviorTree.CPP", 0.014, ("docs/", "examples/", "sample_nodes/"), licence="mit",
          exts=(".cpp", ".h", ".xml", ".md")),
        G("aiplan4eu/unified-planning", 0.012, ("docs/", "unified_planning/test/examples/"), licence="apache-2.0",
          exts=(".py", ".rst", ".md")),
        G("caelan/pddlstream", 0.01, ("examples/", "pddlstream/"), licence="gpl-3.0", exts=(".py", ".pddl")),
        # --- sim-to-real, perception and estimation for control (S10 9%)
        SE("dsp.stackexchange.com", 0.03, keep=r"\b(kalman|ekf|ukf|particle filter|sensor fusion|imu|accelerometer\w*|"
                                              r"gyro\w*|odometry|state estimation|slam|lidar|complementary filter|"
                                              r"attitude|quaternion\w*|gps|tracking filter|observer|system identification)\b",
           tag="sim2real-est"),
        G("AtsushiSakai/PythonRobotics", 0.02, licence="mit", exts=(".py", ".md", ".rst")),
        ARXIV(0.05, r"\b(sim-to-real|sim2real|domain randomi[sz]ation|system identification|state estimation|SLAM|"
                    r"visual odometry|sensor fusion|Kalman filter|reality gap)\b", 8, tag="sim2real-est"),
    ],
    # reconciled 2026-11-01 against docs/domain/sources/quantum.md (1-13). Rejected: samuellimabraz/quantum-assistant
    # (laundered CC-BY-SA Qiskit prose), the OAPEN book (PDF; licence unverified on the copyright page).
    "quantum": [
        SE("quantumcomputing.stackexchange.com", 0.12),
        G("Qiskit/documentation", 0.10, ("docs/guides/", "docs/tutorials/", "learning/courses/"),
          licence="apache-2.0 (code) / cc-by-sa-4.0 (prose)", exts=(".mdx", ".md", ".ipynb")),
        G("Qiskit/qiskit", 0.06, ("qiskit/", "crates/", "docs/"), licence="apache-2.0", exts=(".py", ".rs", ".rst", ".md")),
        G("Qiskit/qiskit-aer", 0.01, ("qiskit_aer/", "docs/"), licence="apache-2.0", exts=(".py", ".rst")),
        G("Qiskit/qiskit-ibm-runtime", 0.01, ("qiskit_ibm_runtime/", "docs/"), licence="apache-2.0", exts=(".py", ".rst")),
        G("Qiskit/qiskit-addon-sqd", 0.01, licence="apache-2.0", exts=(".py", ".rst", ".md")),
        G("PennyLaneAI/demos", 0.05, licence="apache-2.0", exts=(".py", ".md")),
        G("PennyLaneAI/pennylane", 0.04, ("pennylane/", "doc/"), licence="apache-2.0", exts=(".py", ".rst", ".md")),
        G("quantumlib/Cirq", 0.04, ("cirq-core/cirq/", "docs/"), licence="apache-2.0", exts=(".py", ".md")),
        G("quantumlib/Qualtran", 0.03, ("qualtran/",), licence="apache-2.0", exts=(".py", ".md")),
        G("quantumlib/OpenFermion", 0.02, ("src/openfermion/", "docs/"), licence="apache-2.0", exts=(".py", ".md")),
        G("quantumlib/ReCirq", 0.01, licence="apache-2.0", exts=(".py", ".md")),
        G("quantumlib/Stim", 0.04, ("doc/", "src/stim/", "glue/"), licence="apache-2.0", exts=(".md", ".cc", ".h", ".py")),
        G("oscarhiggott/PyMatching", 0.02, licence="apache-2.0", exts=(".py", ".cc", ".h", ".md", ".rst")),
        G("NVIDIA/cudaq-qec", 0.02, licence="apache-2.0", exts=(".py", ".cpp", ".h", ".md", ".rst")),
        G("NVIDIA/cuda-quantum", 0.03, ("docs/sphinx/", "docs/sphinx/examples/"), licence="apache-2.0",
          exts=(".rst", ".py", ".cpp", ".md"), lazy=True, max_files=400),
        G("openqasm/openqasm", 0.02, ("source/",), licence="apache-2.0", exts=(".rst", ".md", ".qasm")),
        G("munich-quantum-toolkit/bench", 0.01, licence="mit", exts=(".py", ".md")),
        G("microsoft/qdk", 0.03, ("katas/", "samples/"), licence="mit", exts=(".qs", ".md", ".py"), lazy=True, max_files=500),
        G("microsoft/QuantumKatas", 0.02, licence="mit", exts=(".qs", ".md")),
        G("errorcorrectionzoo/eczoo_data", 0.03, licence="cc-by-sa-4.0", exts=(".yml",)),
        ARXIV(0.24, r"\b(qubit|quantum error correction|surface code|stabilizer|quantum circuit|logical qubit|"
                    r"decoder|fault[- ]tolerant|variational quantum|quantum algorithm)\b", 8),
    ],
}

# ----------------------------------------------------------------------------- self-traces (PLAN "What makes each mask")
# ~15% of each bucket is reserved for the A192 spine's OWN agent traces on held-out domain prompts (coordinator,
# later). Until a file exists its rows flow to the other sub-areas by share.
SELFTRACE_SHARE = 0.15
SELF_GEN = "DeepSeek-V4.1-Flash A192 spine (self, agent traces)"
for _b in SOURCES:
    SOURCES[_b].insert(0, L(f"dom/selftraces/{_b}.jsonl", SELFTRACE_SHARE, _chat, licence="own model output",
                            generator=SELF_GEN, status="coordinator: A192 spine agent traces; absent until generated",
                            thinking=True))
# Tool-calling DSML rows in EVERY bucket (coordinator 2026-11-01): the A192 calibration rendered chat through
# base._messages, which drops tool_calls/tools/reasoning_content, so REAP never routed a DSML tool-call token and the
# served spine's tool calls degraded. Each bucket's `tool-use` sub-area (>= TOOL_SHARE of rows, checked >= 10% of
# tokens by s02) holds its bucket-specific tool-calling sets (mapped in _SA) plus these generic rows.
_TOOL_SPECIFIC = {"mlsys", "gamedev"}     # own tool-calling sets lead their sub-area (original run: also private transcripts)
for _b in SOURCES:
    SOURCES[_b][1:1] = ([TOOL(_b, 0.03, TOOL_RX[_b]), TOOL_ANY(0.01)] if _b in _TOOL_SPECIFIC
                        else [TOOL(_b, 0.10, TOOL_RX[_b]), TOOL_ANY(0.02)])

# ----------------------------------------------------------------------------- sub-areas (PLAN: stratified calibration)
# Every source belongs to exactly one sub-area of its bucket (first matching pattern; patterns match the source key:
# repo, hf_id or local path, plus `:config`). Topical sub-areas follow docs/domain/sources/<bucket>.md; chat-format
# material that spans topics (transcripts, issue threads, SE Q&A, chat SFT sets) is its own `agentic-chat` sub-area.
# Rows are packed per sub-area (a row never mixes sub-areas), every row records `subarea`/`sub_id`, each sub-area
# has its own held-out rows, and s03 banks per sub-area (bucket totals are sums over its sub-areas).
SUBAREA_FLOOR = 0.08
_SA = {
    "gamedev": [("selftraces", ["dom/selftraces/"]),
                ("tool-use", ["Nemotron-SFT-Agentic-v2", "Nemotron-Agentic-v1"]),
                ("agentic-chat", ["dom/threads/", "stack-exchange"]),
                ("bevy-engine", ["bevyengine/bevy"]),
                ("bevy-ecosystem", ["bevy_rapier", "avian", "leafwing", "bevy_egui", "bevy_asset_loader", "bevy_ecs_tilemap",
                                    "bevy-inspector", "bevy_new_2d"]),
                ("rust-language", ["rust-lang/"])],
    "mlsys": [("selftraces", ["dom/selftraces/"]),
              ("tool-use", ["Nemotron-SFT-CUDA", "Nemotron-SFT-Agentic-v2", "Nemotron-Agentic-v1"]),
              ("agentic-chat", ["dom/threads/", "stack-exchange", "Nemotron-SFT-CUDA", "drkernel"]),
              ("cuda-kernels", ["cutlass", "ThunderKittens", "cutile", "DeepGEMM", "FlashMLA", "DeepEP", "flash-attention",
                                "KernelBook", "AI-CUDA-Engineer", "gpu-mode/lectures", "gpu-glossary"]),
              ("compilers-dsl", ["triton-lang", "tilelang", "pytorch/pytorch", "flash-linear-attention"]),
              ("inference-serving", ["llama.cpp", "vllm-project/vllm", "sglang", "flashinfer", "TensorRT-LLM", "EAGLE"]),
              ("training-dist", ["torchtitan", "Megatron", "verl", "scaling-book"]),
              ("quantization", ["pytorch/ao", "llm-compressor", "Model-Optimizer"]),
              ("papers", ["arxiv_papers"])],
    "robotics": [("selftraces", ["dom/selftraces/"]),
                 ("tool-use", ["robot-session-logs", "Nemotron-SFT-Agentic-v2", "Nemotron-Agentic-v1"]),
                 # locomotion first: "mujocolab/mjlab" and "google-deepmind/mujoco" are substrings of its repo keys
                 ("locomotion-wbc", [":locomotion-wbc", "mjlab_playground", "foothold-prediction", "mujoco_playground"]),
                 ("sim-scene", ["google-deepmind/mujoco", "newton-physics", "genesis-world", "gz-sim", "robot_descriptions",
                                "wuji-description", "allex_model", "mujoco-skills", "urdf-skill"]),
                 ("ros2-middleware", [":ros2-middleware", "dom/threads/", "ros2_documentation", "docs.nav2.org",
                                      "control.ros.org", "ros2_control_demos", "navigation2", "gazebosim/docs", "ros2-mcp",
                                      "rosclaw", "ros2-agent-workflow"]),
                 ("motion-planning", ["ompl/ompl", "moveit", "curobo", "kevinzakka/mink", "pink-kinematics",
                                      "pybullet-planning", "robotics-toolbox"]),
                 ("control-mpc", [":control-mpc", "acados", "casadi/casadi", "crocoddyl", "pinocchio", "RobotLocomotion/drake",
                                  "python-control", "underactuated"]),
                 ("manipulation-vla", [":manipulation-vla", "RussTedrake/manipulation", "cap-x", "CaPtain", "HexaShell",
                                       "RoboHarness", "exploration/rax", "droid", "behavior-1k", "openpi", "Isaac-GR00T",
                                       "lerobot", "AgenticRoboRL"]),
                 ("rl-rewards", ["IsaacLab", "mujocolab/mjlab", "Eureka"]),
                 ("task-planning", ["robocode", "kindergarden", "BTGenBot", "btgenbot", "BehaviorTree.CPP",
                                    "unified-planning", "pddlstream"]),
                 ("sim2real-est", [":sim2real-est", "PythonRobotics"])],
    "quantum": [("selftraces", ["dom/selftraces/"]),
                ("tool-use", ["Nemotron-SFT-Agentic-v2", "Nemotron-Agentic-v1"]),
                ("agentic-chat", ["dom/threads/", "stack-exchange"]),
                ("sdk-docs", ["Qiskit/", "cuda-quantum", "openqasm", "microsoft/qdk", "QuantumKatas"]),
                ("variational-qml", ["pennylane", "PennyLaneAI/demos"]),
                ("algorithms", ["Cirq", "Qualtran", "OpenFermion", "ReCirq", "munich-quantum-toolkit"]),
                ("qec", ["Stim", "PyMatching", "cudaq-qec", "eczoo"]),
                ("papers", ["arxiv_papers"])],
}


def source_key(src: dict) -> str:
    return (src.get("repo") or src["hf_id"]) + (f":{src['config']}" if src.get("config") else "")


def subarea_of(bucket: str, src: dict) -> str:
    k = source_key(src)
    for name, pats in _SA[bucket]:
        if any(p in k for p in pats):
            return name
    raise KeyError(f"{bucket}: no sub-area for source {k}")


SUBAREAS = [f"{b}/{name}" for b in BUCKETS for name, _ in _SA[b]]     # FROZEN global order = s03 stat-bucket ids


def subarea_plan(bucket: str, rows: int, usable_fn=None) -> list[tuple[str, list[dict], int]]:
    """[(sub-area, sources, rows)] in _SA order. selftraces gets SELFTRACE_SHARE of the rows; every other sub-area
    gets max(SUBAREA_FLOOR of the bucket, its value share) in whole rows (largest remainder), value = summed weight."""
    groups: dict[str, list[dict]] = {name: [] for name, _ in _SA[bucket]}
    for s in SOURCES[bucket]:
        if usable_fn is None or usable_fn(s):
            groups[subarea_of(bucket, s)].append(s)
    groups = {k: v for k, v in groups.items() if v}
    n_self = round(SELFTRACE_SHARE * rows) if "selftraces" in groups else 0
    rest = rows - n_self
    topical = [k for k in groups if k != "selftraces"]
    floor = max(1, int(-(-SUBAREA_FLOOR * rows // 1)))           # ceil(8% of the bucket's rows)
    fl = {k: max(floor, int(-(-TOOL_SHARE * rows // 1))) if k == "tool-use" else floor for k in topical}
    assert sum(fl.values()) <= rest, (bucket, "sub-area floors exceed the bucket", fl, rest)
    w = {k: sum(s["weight"] for s in groups[k]) for k in topical}
    out = dict(fl)
    spare = rest - sum(fl.values())
    want = {k: max(0.0, w[k] / sum(w.values()) * rest - fl[k]) for k in topical}   # rows above the floor, by value
    tot = sum(want.values()) or 1.0
    raw = {k: want[k] / tot * spare for k in topical}
    for k in topical:
        out[k] += int(raw[k])
    for k in sorted(topical, key=lambda k: raw[k] - int(raw[k]), reverse=True)[: rows - n_self - sum(out.values())]:
        out[k] += 1
    plan = [("selftraces", groups["selftraces"], n_self)] if n_self else []
    return plan + [(k, groups[k], out[k]) for k in topical]


for _s in SOURCES["mlsys"]:
    if _s["hf_id"] == "GPUMODE/KernelBook":
        _s.update(data_files="dataset_permissive.parquet", stream=_hf_files)

# Licence / eval exclusions, restricted to the sources of the published buckets' catalogues
# (docs/domain/sources/{gamedev,mlsys,robotics,quantum}.md). Eval sets never feed calibration: every source named in
# table C (evaluation) of a bucket's catalogue is removed from that bucket's plan.
EXCLUDED = {
    "licence_nc_or_restricted": [
        "agibot-world/AgiBotWorld2026", "agibot-world/AgiBotWorldChallenge-2026",   # CC BY-NC-SA (robotics.md)
        "OpenGalaxea/Galaxea-Open-World-Dataset",                          # CC BY-NC-SA
        "nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim", "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL",  # CC BY-NC
        "robfiras/loco-mujoco-datasets",                                   # CC BY-NC-ND
        "URDF-Anything-plus/Dataset",                                      # CC BY-NC
        "zhouxueyang/LIBERO-Pro2.0-Train",                                 # CC BY-NC (train split)
        "ut-amrl/codebotler", "zichao22/robo-instruct",                    # CodeBotler licence bans ML training
    ],
    "canary_no_train": [],
    "too_coarse_or_unverified": [
        "flymin/stackexchange_20260331",            # card: none (H4 stack-exchange-preferences used instead)
        "xlang-ai/text2reward", "robot-agent-exploration/skills-lib",   # no licence
        "emgena/emgena_ros2_tf2_coordinate_transform_guard_mcp_teaser",  # advert teaser rows, per-row commercial EULA
        "tencent/Hy-Embodied-0.5-VLA-Data",         # 22 identical Chinese task lists; negligible text
        "jxu124/OpenX-Embodiment", "TianxingChen/RoboTwin2.0",  # RLDS TFRecord tars / zip archives: no text loader
    ],
    "gated_or_eval_only": [
        "SWE-bench/SWE-bench_Multilingual", "ByteDance-Seed/Multi-SWE-bench",   # gamedev.md C (Rust slices)
        "ScalingIntelligence/KernelBench",        # mlsys C (KernelBench base / KB-Verified)
        "capgym/cap-x", "waltstephen/CaPtain", "waltstephen/CaPtain-Harness",   # robotics C3, C4
        "tomsilver/robocode",                     # robotics C2
    ],
}

GENERATOR_POLICIES = ("with-claude", "no-claude")
DEFAULT_POLICY = "with-claude"


def licence_ok(lic: str | None) -> bool | None:
    """False for NC / no-derivatives / Commons-Clause; True for any named permissive, attribution,
    share-alike or copyleft licence (private masks); None when nothing usable is declared."""
    if not lic or lic.strip().lower() in ("other", "unknown"):
        return None
    low = lic.lower()
    toks = set(re.split(r"[^a-z0-9.]+", low))
    flat = low.replace("-", "").replace("_", "")
    if toks & {"nc", "nd", "busl"} or "noncommercial" in flat or "noderiv" in flat or "commonsclause" in flat:
        return False
    if low.startswith("unknown") or low.startswith("error"):
        return None
    return True


def usable(src: dict, allow_unverified: bool, policy: str = DEFAULT_POLICY) -> tuple[bool, str]:
    g = src["generator"].lower()
    if policy == "no-claude" and ("claude" in g or "anthropic" in g):
        return False, "policy no-claude excludes claude-generated"
    if g in ("unknown", "mixed") or (g == "unverified" and not allow_unverified):
        return False, f"generator {g}"
    excluded = {h for v in EXCLUDED.values() for h in v}
    if src["hf_id"] in excluded or src.get("repo") in excluded:
        return False, "listed in EXCLUDED"
    if licence_ok(src["licence"]) is False:
        return False, f"declared licence {src['licence']}"
    return True, "ok"


def rows_per_bucket(total_tokens: int = TOTAL_TOKENS, seq_len: int = SEQ_LEN) -> dict[str, int]:
    """Largest-remainder apportionment over THIS module's buckets (the base function reads base.BUCKETS)."""
    n = max(len(BUCKETS), round(total_tokens / seq_len))
    raw = {b: TOKEN_TARGET[b] * n for b in BUCKETS}
    out = {b: int(v) for b, v in raw.items()}
    for b in sorted(BUCKETS, key=lambda b: raw[b] - out[b], reverse=True)[: n - sum(out.values())]:
        out[b] += 1
    return out


def interleave(counts: dict[str, int]) -> list[str]:
    """Same smooth weighted round-robin as the base spec, over THIS module's bucket order."""
    total = sum(counts.values())
    credit = {b: 0.0 for b in counts}
    left = dict(counts)
    order = []
    for _ in range(total):
        for b in counts:
            if left[b]:
                credit[b] += counts[b] / total
        b = max((b for b in counts if left[b]), key=lambda b: (credit[b], -BUCKETS.index(b)))
        credit[b] -= 1.0
        left[b] -= 1
        order.append(b)
    return order


def interleave_keys(counts: dict[str, int]) -> list[str]:
    """Smooth weighted round-robin over arbitrary keys (sub-areas inside a pool), ties by insertion order."""
    keys = list(counts)
    total = sum(counts.values())
    credit = dict.fromkeys(keys, 0.0)
    left = dict(counts)
    order = []
    for _ in range(total):
        for k in keys:
            if left[k]:
                credit[k] += counts[k] / total
        k = max((k for k in keys if left[k]), key=lambda k: (credit[k], -keys.index(k)))
        credit[k] -= 1.0
        left[k] -= 1
        order.append(k)
    return order


def git_sources() -> list[dict]:
    return [s for b in BUCKETS for s in SOURCES[b] if s["hf_id"].startswith("git:")]
