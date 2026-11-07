#!/usr/bin/env python3
"""D2 free-running gate (D2_METHOD item 9) for the top 3 masks per domain. GPU. NOT RUN YET.

Teacher-forced replay cannot see generation drift, loops or DSML breakage, so each finalist GENERATES ~200
held-out domain answers and the unpruned teacher then scores that text in one teacher pass:

  gen      GPU (server window). The mask's Thor build is served (v41_server --ckpt <build> --keepset <mask.json>;
           the build comes from the 384 pool via the phase-A assembly, `--build`), and answers the bucket's
           HELD-OUT prompt pool ($V41_WORK/dom/prompts/<bucket>.jsonl rows with heldout: true; never calibration
           prompts), greedy, pinned generation config (thinking every other prompt, --max-tokens), single stream.
           The served identity (/v1/models keepset sha) is asserted before the first request.
           -> d2/freerun/<bucket>/<mask>/answers.jsonl
  corpus   CPU. Render [user, assistant] through the checkpoint encoder (s02 Renderer), one row per answer
           (<= 8192 tokens), `answer_start` = first assistant token; writes an s03-format corpus
           $V41_WORK/dom/corpus_fr_<bucket>_<mask>/ (tokens/types/valid npy, rows.jsonl, manifest.json).
  teacher  GPU. s03 over that corpus as its own pass (V41_DOM=1 V41_DOM_PASS=fr_<bucket>_<mask>): the unpruned
           384-expert teacher, whose `teacher/` cache gives top-32 ids/log-probs and next-token NLL. The bank it
           also writes is ignored. Then `passes.py retire fr_...` to drop the stream.
  score    CPU. Over answer tokens only: teacher NLL of the generated token (mean, row-bootstrap CI), out-of-top-32
           rate (generated token not in the teacher's top-32), plus text checks: 4-gram loop rate, truncation rate,
           DSML envelope parse failures. Compared across the bucket's masks and A192 (same prompts, same config).
           -> d2/freerun/<bucket>/score.json

EXPECTED COST per mask: gen ~200 answers x ~1.5k tokens at ~25 tok/s ~ 3.5 h worst case (cap with --max-tokens
2048 and --n 200: ~1-2 h); teacher pass ~300k tokens at ~2,200 tok/s per layer x 40 layers ~ 1.5 h (D2_METHOD
"~1 h per mask"), NVMe scratch ~55 KB/token (~16 GB, released by retire), MemAvailable >= 108 GiB (s03 guard);
the server and s03 never co-run (unified memory): gen for all masks first, server stopped, then teacher passes.

  setsid nohup .venv/bin/python ops/dom/d2_gpu/freerun_gate.py gen --bucket gamedev --mask hope_w45_pf4_st1 \\
      --build $V41_WORK/thor/dom/gamedev_hope_w45_pf4_st1 > ops/logs/d2_freerun.log 2>&1 < /dev/null &
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
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ops.lib import env, keepset  # noqa: E402

D2 = env.WORK / "dom" / "d2"
FR = D2 / "freerun"
PROMPTS = env.WORK / "dom" / "prompts"


def _mask_file(bucket: str, mask: str) -> Path:
    return env.WORK / "s04" / "keepset.json" if mask == "A192" else D2 / bucket / f"{mask}.json"


def _pass_name(bucket: str, mask: str) -> str:
    return f"fr_{bucket}_{mask}"


def gen(a):
    from ops.dom.selftraces import chat
    mf = _mask_file(a.bucket, a.mask)
    _, sha = keepset.load(mf)
    out = FR / a.bucket / a.mask
    out.mkdir(parents=True, exist_ok=True)
    srv = None
    if a.build:
        for p in ("v41_server", "v41_decode", "llama-server", "vllm"):
            if subprocess.run(["pgrep", "-x", p], capture_output=True).returncode == 0:
                raise SystemExit(f"{p} running; refusing (unified memory)")
        if (env.WORK / "GPU_WINDOW.lock").exists():
            raise SystemExit("GPU_WINDOW.lock held")
        (env.WORK / "GPU_WINDOW.lock").write_text(f"{os.getpid()} {env.now()} d2 freerun gen {a.bucket}/{a.mask}\n")
        srv = subprocess.Popen([str(env.REPO / "cuda/build/v41_server"), "--ckpt", str(a.build), "--keepset", str(mf),
                                "--port", str(a.port), "--seqmax", str(a.seqmax)], stdout=open(out / "server.log", "w"),
                               stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{a.port}"
        _wait_identity(url, sha, a.timeout)
        prompts = [json.loads(l) for l in open(PROMPTS / f"{a.bucket}.jsonl")]
        prompts = [p for p in prompts if p.get("heldout") and not p.get("needs_tools")][:a.n]
        f = out / "answers.jsonl"
        done = {json.loads(l)["pid"] for l in open(f)} if f.exists() else set()
        for i, p in enumerate(prompts):
            pid = hashlib.sha1(p["prompt"].encode()).hexdigest()[:16]
            if pid in done:
                continue
            thinking = i % 2 == 1
            t0 = time.time()
            r = chat(url, [{"role": "user", "content": p["prompt"]}], a.max_tokens, thinking, a.timeout)
            ch = r["choices"][0]
            with open(f, "a") as fh:
                fh.write(json.dumps({"pid": pid, "mask": a.mask, "mask_sha256": sha, "subarea": p.get("subarea"),
                                     "thinking": thinking, "max_tokens": a.max_tokens, "prompt": p["prompt"],
                                     "message": ch["message"], "finish_reason": ch.get("finish_reason"),
                                     "secs": time.time() - t0}, ensure_ascii=False) + "\n")
    finally:
        if srv is not None:
            srv.terminate(); srv.wait(60)
            (env.WORK / "GPU_WINDOW.lock").unlink(missing_ok=True)


def _wait_identity(url: str, sha: str, timeout: float):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            ident = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=10).read())
            blob = json.dumps(ident)
            if sha[:16] not in blob and sha not in blob:
                raise SystemExit(f"served model does not report keep-set {sha[:16]}: {blob[:300]}")
            return
        except (OSError, ValueError):
            time.sleep(10)
    raise SystemExit("server did not come up")


def corpus(a):
    from ops.dom import corpus_spec_dom as spec
    from ops.s02_calib_corpus import Renderer
    R = Renderer()
    out = env.WORK / "dom" / f"corpus_{_pass_name(a.bucket, a.mask)}"
    out.mkdir(parents=True, exist_ok=True)
    ans = [json.loads(l) for l in open(FR / a.bucket / a.mask / "answers.jsonl")]
    toks, rows, off = [], [], 0
    sub0 = next(i for i, n in enumerate(spec.SUBAREAS) if n.startswith(a.bucket + "/"))
    for x in ans:
        msg = {"role": "assistant", "content": x["message"].get("content") or ""}
        if x["message"].get("reasoning_content"):
            msg["reasoning_content"] = x["message"]["reasoning_content"]
        user = [{"role": "user", "content": x["prompt"]}]
        full = R.text_ids(user + [msg], x["thinking"])
        pre = R.text_ids(user, x["thinking"])
        k = 0
        while k < min(len(pre), len(full)) and pre[k] == full[k]:
            k += 1
        full = full[:8192]
        if k >= len(full) - 1:
            continue
        sid = spec.SUBAREAS.index(f"{a.bucket}/{x['subarea']}") if x.get("subarea") and \
            f"{a.bucket}/{x['subarea']}" in spec.SUBAREAS else sub0
        rows.append({"row": len(rows), "bucket": a.bucket, "bucket_id": spec.BUCKETS.index(a.bucket), "subarea": x.get("subarea"),
                     "sub_id": sid, "offset": off, "length": len(full), "heldout": False, "n_valid": len(full),
                     "docs": [{"source": f"freerun:{a.mask}", "tokens": len(full)}], "images": [], "answer_start": k,
                     "pid": x["pid"], "finish_reason": x["finish_reason"]})
        toks += full; off += len(full)
    np.save(out / "tokens.npy", np.asarray(toks, np.int32))
    np.save(out / "types.npy", np.full(off, -1, np.int8))
    np.save(out / "valid.npy", np.ones(off, np.uint8))
    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ref = json.load(open(env.WORK / "dom" / "corpus" / "manifest.json"))
    env.atomic_json(out / "manifest.json", {**{k: ref[k] for k in ("spec_hash", "generator_policy", "revision", "draw")},
                                           "buckets": spec.BUCKETS, "subareas": spec.SUBAREAS,
                                           "pass": _pass_name(a.bucket, a.mask), "pass_buckets": [a.bucket],
                                           "rows": len(rows), "tokens_total": off, "tokens_valid": off,
                                           "order": [r["bucket"] for r in rows], "max_row_len": max(r["length"] for r in rows),
                                           "longctx_lengths": [], "source_manifest_ts": ref["ts"], "ts": env.now(),
                                           "freerun": {"mask": a.mask, "bucket": a.bucket}})
    print(f"{len(rows)} rows, {off} tokens -> {out}")


def teacher(a):
    """The s03 pass over the free-run corpus (same driver conventions as run_d1_passes.sh)."""
    p = _pass_name(a.bucket, a.mask)
    if (env.WORK / "GPU_WINDOW.lock").exists():
        raise SystemExit("GPU_WINDOW.lock held")
    (env.WORK / "GPU_WINDOW.lock").write_text(f"{os.getpid()} {env.now()} d2 freerun teacher {p}\n")
    try:
        d = env.WORK / "dom" / f"s03_{p}"; d.mkdir(parents=True, exist_ok=True)
        e = {**os.environ, "V41_DOM": "1", "V41_DOM_PASS": p}
        with open(d / "s03.log", "a") as log:
            rc = subprocess.run([sys.executable, str(env.OPS / "s03_saliency.py"), "run", "--kernels", "tilelang",
                                 "--engram-dir", a.engram_dir or sys.exit("teacher: set --engram-dir or V41_ENGRAM_DIR")], env=e, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            raise SystemExit(f"s03 failed rc={rc} (see {d / 's03.log'})")
        subprocess.run([sys.executable, str(env.OPS / "dom" / "passes.py"), "retire", p], env=e, check=True)
    finally:
        (env.WORK / "GPU_WINDOW.lock").unlink(missing_ok=True)


def score(a):
    from ops.lib import stats
    res = {}
    for m in a.masks:
        p = _pass_name(a.bucket, m)
        c = env.WORK / "dom" / f"corpus_{p}"
        t = env.WORK / "dom" / f"s03_{p}" / "teacher"
        rows = [json.loads(l) for l in open(c / "rows.jsonl")]
        tok = np.load(c / "tokens.npy", mmap_mode="r")
        ids = np.load(t / "top32_ids.npy", mmap_mode="r")
        nll = np.load(t / "nll_next.npy", mmap_mode="r")
        v_nll, v_out, cl, loops, trunc = [], [], [], [], 0
        for r in rows:
            o, n, k = r["offset"], r["length"], r["answer_start"]
            pos = np.arange(o + k - 1, o + n - 1)          # predicting answer token t+1 from position t
            nxt = np.asarray(tok[pos + 1])
            v_nll.append(np.asarray(nll[pos], np.float64))
            v_out.append(~(np.asarray(ids[pos]) == nxt[:, None]).any(1))
            cl.append(np.full(pos.size, r["row"]))
            a4 = [tuple(nxt[i:i + 4]) for i in range(max(0, nxt.size - 3))]
            loops.append(1 - len(set(a4)) / max(len(a4), 1))
            trunc += r.get("finish_reason") == "length"
        cl = np.concatenate(cl)
        txt = [json.loads(l) for l in open(FR / a.bucket / m / "answers.jsonl")]
        dsml_bad = sum(_dsml_broken(x["message"].get("content") or "") for x in txt)
        res[m] = {"answers": len(rows), "teacher_nll": stats.cluster_bootstrap_mean(np.concatenate(v_nll), cl),
                  "out_top32": stats.cluster_bootstrap_mean(np.concatenate(v_out).astype(float), cl),
                  "loop4_rate_mean": float(np.mean(loops)), "truncated": trunc, "dsml_broken": dsml_bad}
    env.atomic_json(FR / a.bucket / "score.json", res)
    print(json.dumps({m: {"nll": r["teacher_nll"]["mean"], "out32": r["out_top32"]["mean"], "loop": r["loop4_rate_mean"]}
                      for m, r in res.items()}, indent=1))


_DSML_OPEN = re.compile(r"<｜DSML｜function_calls>")
_DSML_CLOSE = re.compile(r"</｜DSML｜function_calls>")


def _dsml_broken(text: str) -> bool:
    return len(_DSML_OPEN.findall(text)) != len(_DSML_CLOSE.findall(text))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gen", "corpus", "teacher", "score"])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--mask", help="candidate name under d2/<bucket>/, or A192")
    ap.add_argument("--masks", nargs="*", help="score: masks to compare (include A192)")
    ap.add_argument("--build", type=Path, help="gen: Thor build dir of the mask (starts/stops the server itself)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seqmax", type=int, default=8192)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--engram-dir", default=os.environ.get("V41_ENGRAM_DIR"), help="teacher Engram shard dir (teacher step)")
    a = ap.parse_args()
    {"gen": gen, "corpus": corpus, "teacher": teacher, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
