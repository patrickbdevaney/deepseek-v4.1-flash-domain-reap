#!/usr/bin/env python3
"""Self-traces for the domain calibration slot (docs/domain/PLAN.md, "What makes each mask as good
as it can be", item 2; graph node D0t).

The A192 spine answers each bucket's task prompts ($V41_WORK/dom/prompts/<bucket>.jsonl) greedily
through our own server, and every answer becomes a `local:` chat row (id, sort, split, domain, subarea, generator,
thinking, n_tokens, messages; ops/dom/README.md), so calibration also sees the model's OWN routing on its own outputs
(own-output calibration: +3-4 points in the pruning literature).

Resumable: a prompt whose id is already in the output file is skipped. Single stream, because the
server is single-stream. Rows from prompts flagged `heldout` are written to <bucket>.heldout.jsonl.

RUN (only with the v41 server up and nothing else model-loading; detach it):
    setsid nohup python3 ops/dom/selftraces.py --url http://127.0.0.1:8000 \\
        --buckets gamedev mlsys ... --tokens-per-bucket 40000 > ops/logs/dom_selftraces.log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ops.lib import env  # noqa: E402
from ops.dom import render as T  # noqa: E402

PROMPTS = env.WORK / "dom" / "prompts"
OUT = env.WORK / "dom" / "selftraces"
GENERATOR = "self:A192"


def chat(url: str, messages: list[dict], max_tokens: int, thinking: bool, timeout: float) -> dict:
    body = {"model": "local", "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": False, "thinking_mode": "thinking" if thinking else "chat"}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def done_ids(*paths: Path) -> set[str]:
    ids = set()
    for p in paths:
        if p.exists():
            for line in open(p):
                try:
                    ids.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
    return ids


def row_for(bucket: str, p: dict, msg: dict, thinking: bool) -> dict:
    rid = f"{bucket}:self/{p.get('source', 'prompt')}/{hashlib.sha1(p['prompt'].encode()).hexdigest()[:12]}"
    asst = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("reasoning_content"):
        asst["reasoning_content"] = msg["reasoning_content"]
    messages = [{"role": "user", "content": p["prompt"]}, asst]
    return {"id": rid, "sort": hashlib.sha1(rid.encode()).hexdigest()[:16],
            "source_file": f"selftraces/{bucket}", "split": "heldout" if p.get("heldout") else "train",
            "split_key": rid, "domain": bucket, "subarea": p.get("subarea"), "generator": GENERATOR,
            "thinking": bool(asst.get("reasoning_content")), "n_tokens": T.n_tokens(messages, thinking),
            "messages": messages}


def run_bucket(args, bucket: str) -> None:
    src = PROMPTS / f"{bucket}.jsonl"
    if not src.exists():
        print(f"[selftraces] {bucket}: no prompt pool at {src}, skipped", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    out_tr, out_ho = OUT / f"{bucket}.jsonl", OUT / f"{bucket}.heldout.jsonl"
    seen = done_ids(out_tr, out_ho)
    have = sum(json.loads(l)["n_tokens"] for l in open(out_tr)) if out_tr.exists() else 0
    prompts = [json.loads(l) for l in open(src)]
    # interleave sub-areas so a token cap still covers every sub-area
    by_sub: dict[str, list] = {}
    for p in prompts:
        by_sub.setdefault(p.get("subarea") or "-", []).append(p)
    order = [q for grp in zip(*[iter(v) for v in by_sub.values()]) for q in grp]
    order += [p for v in by_sub.values() for p in v if p not in order]
    t0 = time.time()
    for i, p in enumerate(order):
        # alternate thinking and chat turns: a thinking turn that hits max_tokens has no answer at all
        thinking = args.thinking and (args.chat_every == 0 or i % args.chat_every != 0)
        if have >= args.tokens_per_bucket and not p.get("heldout"):
            continue
        rid = row_for(bucket, p, {"content": ""}, False)["id"]
        if rid in seen:
            continue
        try:
            r = chat(args.url, [{"role": "user", "content": p["prompt"]}], args.max_tokens, thinking,
                     args.timeout)
        except Exception as e:  # one bad prompt must not stop the bucket
            print(f"[selftraces] {bucket}: request failed ({type(e).__name__}: {e}); continuing", flush=True)
            continue
        msg = r["choices"][0]["message"]
        if r["choices"][0].get("finish_reason") == "length" and not (msg.get("content") or "").strip():
            # thinking-only, truncated: no answer to calibrate on; do not record it, so a rerun retries it
            print(f"[selftraces] {bucket}: truncated before answering, dropped", flush=True)
            continue
        row = row_for(bucket, p, msg, thinking)
        with open(out_ho if row["split"] == "heldout" else out_tr, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        seen.add(rid)
        if row["split"] == "train":
            have += row["n_tokens"]
        print(f"[selftraces] {bucket}: {have} train tokens, {time.time() - t0:.0f}s", flush=True)
        if have >= args.tokens_per_bucket and all(not q.get("heldout") for q in order):
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--buckets", nargs="+", required=True)
    ap.add_argument("--tokens-per-bucket", type=int, default=40_000)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--chat-every", type=int, default=2, help="every Nth prompt runs in chat mode (0: never)")
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    for b in args.buckets:
        run_bucket(args, b)
    print("SELFTRACES_DONE", flush=True)


if __name__ == "__main__":
    main()
