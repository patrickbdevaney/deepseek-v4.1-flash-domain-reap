#!/usr/bin/env python3
"""EASY-EP baseline for the domain-mask study (docs/domain/EASYEP_BASELINE.md).

EASY-EP (Dong et al., arXiv 2504.06792; code github.com/RUCAIBox/EASYEP, pruning/expert_selection.py) scores every
routed expert i of layer l over the tokens t of a few in-domain demonstrations (input + model response):

    I(E_i^l) = sum_t  g_{i,t}^l * ||e_{i,t}^l||  *  max(1 - cos(h_t^l, h_t^l + r_t^l), 0)      (paper Eqs. 3, 7, 8)

g = the routing weight actually applied (normalised, x routed_scaling_factor), e = the expert's output, h = the
residual stream entering the MoE, r = the ROUTED experts' summed output (shared expert excluded). The code clamps the
token weight at 0 (paper Eq. 8 does not). Per layer the top-M experts are kept (uniform allocation, their App. G.2).
Demonstrations: 25 per domain (paper §5.1; code: seed-42 shuffle, sort by length descending, take 25; inf_new.py
skips inputs longer than 13,000 tokens).

Ours (decisions in the doc): from scratch, uniform 192/384, one mask per tier-1 bucket, our hard guard set forced in.
The per-token terms are NOT in the D1 banks (they hold per-sub-area sums of g*||f|| and per-token routing only), so
a recorder pass (s03 with V41_EASYEP_REC=1 over a small demo corpus) supplies them.

  demos            CPU: pick 25 demos per bucket from the D1 pass corpora -> $W/dom/d2/easyep/demos.json
  corpus [--group G --buckets ...]
                   CPU: write the demo corpus $W/dom/corpus_easyep[_G]/ (s03 format, one demo per row)
  plan             CPU: tokens, NVMe scratch, GPU-hour estimate of the recorder pass -> easyep/plan.json
  select           CPU: scores from the recorder output -> easyep/<bucket>/easyep.keep.json + easyep.json
  bank             CPU, runnable now: proxy without the token weight (s_t = 1) on the full D1 bucket corpus
                   -> easyep/<bucket>/easyep_bank.keep.json + easyep_bank.json  (ablation, not the baseline)

Recorder pass (GPU window only; nothing else on the GPU; ~5-6 h, see `plan`):
  V41_DOM=1 V41_DOM_PASS=easyep V41_EASYEP_REC=1 setsid nohup .venv/bin/python ops/s03_saliency.py run \\
      --kernels tilelang --engram-dir "/media/patrickd/Seagate Portable Drive/v41_1017_engram" \\
      > ~/models/v41-work/dom/s03_easyep/s03.log 2>&1 < /dev/null &

CPU steps (capped):  systemd-run --user --scope -p MemoryMax=2G nice -n 19 .venv/bin/python ops/dom/easyep.py <cmd>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ops.lib import env, keepset  # noqa: E402

STAGE = "easyep"
REPO = Path(__file__).resolve().parent.parent.parent
DOM = env.WORK / "dom"
OUT = DOM / "d2" / "easyep"
S04 = env.WORK / "s04"
A192_KEEP = S04 / "keepset.json"
ANCHORED = DOM / "d2" / "anchored"
E_PLAN = DOM / "e" / "plan.json"
TIER1 = ["gamedev", "mlsys", "robotics", "quantum"]   # this release's buckets
PUBLIC = ["gamedev", "mlsys", "robotics", "quantum"]
N_LAYERS, E, K, N_KEEP = 40, 384, 6, 192

# EASY-EP constants (expert_selection.py / inf_new.py / paper §5.1)
DEMO_N = 25
DEMO_MAX_TOKENS = 13_000
DEMO_SEED = 42
# ours
MIN_RESPONSE_TOKENS = 32          # a demo must carry a response (tokens inside assistant turns)
PAD_TO = 128                      # demo rows are padded (valid=0, EOS) to a multiple of this; pads are never scored
BOS, EOS, SYS, USER, ASST = 0, 1, 128799, 128803, 128804
TEXT = -1
SELFTRACE = "local:dom/selftraces/"
# the generic tool pair is filtered only by a keyword on the tool schema; README: "never used as prompt sources
# (off-domain)". Demonstrations must be in-domain tasks, so they are excluded here too.
GENERIC_TOOL = ("nvidia/Nemotron-SFT-Agentic-v2/tool_calling", "nvidia/Nemotron-Agentic-v1/tool_calling")
COS_COL = {"full": 0, "collapsed": 1}
# measured s03 rates (W/dom/s03_<p>/s03.log): p1 1,638,400 tok 12.1 h; t3 819,200 tok 7.3 h -> linear fit
S03_FIXED_H, S03_H_PER_MTOK = 2.5, 5.86
S03_BYTES_PER_TOKEN = 55_064 + 2 * 4 * 40 + 6 * 4 * 40     # D1 persistent scratch + recorder (cos 2 f32, gn 6 f32)


# ----------------------------------------------------------------------------- pure functions (tested)
def response_mask(t: np.ndarray) -> np.ndarray:
    """True on tokens inside an assistant turn: after <|Assistant|>, before the next User/System/BOS/EOS marker.
    Marker tokens themselves are False."""
    t = np.asarray(t)
    mk = np.isin(t, (ASST, USER, SYS, BOS, EOS))
    last = np.maximum.accumulate(np.where(mk, np.arange(len(t)), -1))
    out = np.zeros(len(t), bool)
    ok = last >= 0
    out[ok] = t[last[ok]] == ASST
    return out & ~mk


def split_docs(rows: list[dict], tokens: np.ndarray, bucket: str) -> list[dict]:
    """Reassemble the bucket's documents from packed rows. s02 packs a sub-area's documents back to back and a
    document that overflows continues on the NEXT row of the same sub-area (first fragment, same source, no BOS).
    -> [{source, frags: [(row, start, n)], tokens, heldout, subarea, sub_id, at_row_start, broken}]"""
    by: dict[str, list[dict]] = {}
    for r in rows:
        if r["bucket"] == bucket:
            by.setdefault(r["subarea"], []).append(r)
    out = []
    for sa, seq in by.items():
        cur = None
        for r in seq:
            pos = 0
            for i, d in enumerate(r["docs"]):
                a, n = r["offset"] + pos, int(d["tokens"])
                pos += n
                if n == 0:
                    continue
                if int(tokens[a]) == BOS:
                    if cur is not None:
                        out.append(cur)
                    cur = {"source": d["source"], "frags": [(r["row"], a, n)], "tokens": n, "heldout": bool(r["heldout"]),
                           "subarea": sa, "sub_id": r["sub_id"], "at_row_start": pos == n, "broken": False,
                           "calibration_only": bool(d.get("calibration_only"))}
                elif cur is not None and i == 0 and d["source"] == cur["source"]:
                    cur["frags"].append((r["row"], a, n))
                    cur["tokens"] += n
                    cur["heldout"] |= bool(r["heldout"])
                else:                      # a continuation we cannot attach: never use it
                    if cur is not None:
                        out.append(cur)
                    cur = {"source": d["source"], "frags": [(r["row"], a, n)], "tokens": n, "heldout": bool(r["heldout"]),
                           "subarea": sa, "sub_id": r["sub_id"], "at_row_start": False, "broken": True}
            if pos != r["length"]:
                raise SystemExit(f"row {r['row']}: docs cover {pos} of {r['length']} tokens")
        if cur is not None:
            out.append(cur)
    return out


def easyep_order(docs: list[dict], seed: int = DEMO_SEED) -> list[dict]:
    """EASY-EP's demo order: random.seed(42); shuffle; stable sort by length descending."""
    d = list(docs)
    random.Random(seed).shuffle(d)
    return sorted(d, key=lambda x: x["tokens"], reverse=True)


def truncate(d: dict, n: int) -> dict:
    """The document's first n tokens (its fragments trimmed)."""
    fr, left = [], n
    for r, s, k in d["frags"]:
        if left <= 0:
            break
        fr.append((r, s, min(k, left))); left -= k
    return {**d, "frags": fr, "tokens": n, "truncated_from": d["tokens"]}


def pick_demos(docs: list[dict], n: int = DEMO_N, long_docs: list[dict] = ()) -> list[dict]:
    """Self-traces (the model's own responses) first, then the other response-shaped documents, each tier in
    EASY-EP's order; take n. Only if that leaves fewer than n: over-long documents cut to DEMO_MAX_TOKENS
    (EASY-EP skipped them; a cut one is still prompt + response start), in the same order."""
    st = [d for d in docs if d["source"].startswith(SELFTRACE)]
    other = [d for d in docs if not d["source"].startswith(SELFTRACE)]
    out = (easyep_order(st) + easyep_order(other))[:n]
    return out + [truncate(d, DEMO_MAX_TOKENS) for d in easyep_order(list(long_docs))][:n - len(out)]


def token_weight(cos: np.ndarray) -> np.ndarray:
    """EASY-EP token contribution s_t = max(1 - cos, 0) (expert_selection.py)."""
    return np.maximum(1.0 - np.asarray(cos, np.float64), 0.0)


def easyep_scores(idx: np.ndarray, gn: np.ndarray, cos: np.ndarray | None, valid: np.ndarray,
                  n_experts: int = E) -> np.ndarray:
    """I(e) = sum_t sum_slots [idx_t,slot == e] * gn_t,slot * s_t over valid tokens; cos None -> s_t = 1."""
    v = np.asarray(valid, bool)
    w = np.asarray(gn, np.float64)[v]
    if cos is not None:
        w = w * token_weight(np.asarray(cos)[v])[:, None]
    return np.bincount(np.asarray(idx)[v].astype(np.int64).ravel(), weights=w.ravel(), minlength=n_experts)


def topk_ids(score: np.ndarray, k: int) -> np.ndarray:
    """Top-k ids, ties to the lower id (stable)."""
    return np.sort(np.argsort(-np.asarray(score, np.float64), kind="stable")[:k])


def select_layer(score: np.ndarray, guard: np.ndarray, n_keep: int = N_KEEP) -> tuple[list[int], dict]:
    """keep = guards UNION the best (n_keep - |guards|) non-guard experts by score.
    info: guards EASY-EP's own top-n_keep would drop, and the EASY-EP picks displaced to make room."""
    guard = np.unique(np.asarray(guard, int))
    assert len(guard) <= n_keep, f"{len(guard)} guards > {n_keep}"
    raw = topk_ids(score, n_keep)
    g = np.zeros(len(score), bool); g[guard] = True
    s = np.where(g, -np.inf, np.asarray(score, np.float64))
    fill = np.argsort(-s, kind="stable")[: n_keep - len(guard)]
    keep = np.sort(np.concatenate([guard, fill])).astype(int)
    assert len(set(keep.tolist())) == n_keep
    dropped = sorted(set(guard.tolist()) - set(raw.tolist()))
    displaced = sorted(set(raw.tolist()) - set(keep.tolist()))
    assert len(dropped) == len(displaced)
    return keep.tolist(), {"n_guard": int(len(guard)), "guards_dropped_by_easyep": dropped, "displaced": displaced}


def overlap(a: dict, b: dict) -> dict:
    inter = [len(set(a[str(L)]) & set(b[str(L)])) for L in range(len(a))]
    return {"mean_shared": float(np.mean(inter)), "min_shared": int(min(inter)), "max_shared": int(max(inter)),
            "mean_jaccard": float(np.mean([i / (2 * N_KEEP - i) for i in inter])), "per_layer": inter}


# ----------------------------------------------------------------------------- inputs
def pass_of(bucket: str) -> str:
    for c in sorted(DOM.glob("corpus_*")):
        m = c / "manifest.json"
        if m.exists() and bucket in json.load(open(m)).get("pass_buckets", []):
            return c.name[len("corpus_"):]
    raise SystemExit(f"{bucket}: no D1 pass corpus")


def eval_slugs() -> list[str]:
    """Dataset / repo names of every phase-E suite (ops/eval/suites/*.py `source = "hf:..."|"git:..."`)."""
    out = set()
    for f in sorted((REPO / "ops" / "eval" / "suites").glob("*.py")):
        for m in re.finditer(r'source(?:\s*,\s*licence)?\s*=\s*"(hf|git):([^"\s]+)', f.read_text()):
            s = m.group(2).rstrip("/")
            s = re.sub(r"^https?://github\.com/", "", s)
            out.add(s.lower())
            out.add(s.split("/")[-1].lower())
    return sorted(x for x in out if len(x) >= 6)


def _contaminated(text: str, slugs: list[str]) -> list[str]:
    t = text.lower()
    return [s for s in slugs if s in t]


def guards_all() -> tuple[np.ndarray, dict, dict]:
    """The pipeline's hard keep (same call as anchored.py / select_dom.py): super-experts + position protect."""
    from ops.dom import d2lib as D
    from ops.dom import select_dom as S
    av = D.available_buckets()
    avail = {b: av[b] for b in D.TIER1 if b in av}
    return S._guards_cached(D.DOM / "d2", avail)


def guard_ids(L: int, sup: np.ndarray, protect: dict) -> np.ndarray:
    return np.union1d(np.asarray(protect.get(str(L), []), int), np.flatnonzero(sup[L])).astype(int)


def anchored_pick(bucket: str) -> tuple[str | None, Path | None]:
    if E_PLAN.exists():
        p = json.load(open(E_PLAN)).get("picks", {})
        if bucket in p:
            return p[bucket], ANCHORED / bucket / f"{p[bucket]}.json"
    return None, None


# ----------------------------------------------------------------------------- demos / corpus
def demos(a) -> dict:
    slugs = eval_slugs()
    res = {"ts": env.now(), "rule": {"n": DEMO_N, "max_tokens": DEMO_MAX_TOKENS, "seed": DEMO_SEED,
                                     "min_response_tokens": MIN_RESPONSE_TOKENS, "order": "self-traces first, then "
                                     "other response-shaped train documents; each tier seed-42 shuffle + stable sort "
                                     "by length descending (EASY-EP expert_selection.py)",
                                     "excluded_sources": list(GENERIC_TOOL)},
           "eval_slugs": slugs, "buckets": {}}
    for b in a.buckets or TIER1:
        p = pass_of(b)
        C = DOM / f"corpus_{p}"
        rows = [json.loads(l) for l in open(C / "rows.jsonl")]
        tok = np.load(C / "tokens.npy", mmap_mode="r")
        st_file = DOM / "selftraces" / f"{b}.jsonl"
        # self-trace ids carry the prompt's origin source ("<b>:self/<source>/<hash>"); the text may MENTION a
        # benchmark (a mention is not an item), so only the origin is checked
        st_ids = [json.loads(l)["id"] for l in open(st_file)] if st_file.exists() else []
        st_hits = sorted({h for i in st_ids for h in _contaminated(i, slugs)})
        if st_hits:
            raise SystemExit(f"{b}: self-trace prompt origins match phase-E sources {st_hits}")
        docs = split_docs(rows, tok, b)
        stats = {"docs": len(docs), "broken": 0, "heldout": 0, "too_long": 0, "no_response": 0, "generic_tool": 0,
                 "eval_source": 0}
        ok, long_ok = [], []
        for d in docs:
            why = ("broken" if d["broken"] else "heldout" if d["heldout"] else
                   "too_long" if d["tokens"] > DEMO_MAX_TOKENS else
                   "generic_tool" if d["source"].startswith(GENERIC_TOOL) else
                   "eval_source" if _contaminated(d["source"], slugs) else None)
            if why in (None, "too_long"):
                t = np.concatenate([np.asarray(tok[s:s + n]) for _, s, n in d["frags"]])
                rm = response_mask(t)
                d["response_tokens"] = int(rm.sum())
                if why == "too_long":
                    if rm[:DEMO_MAX_TOKENS].sum() >= MIN_RESPONSE_TOKENS:
                        long_ok.append({**d, "response_tokens": int(rm[:DEMO_MAX_TOKENS].sum())})
                elif d["response_tokens"] < MIN_RESPONSE_TOKENS:
                    why = "no_response"
            if why:
                stats[why] += 1
            else:
                ok.append(d)
        pick = pick_demos(ok, long_docs=long_ok)
        for i, d in enumerate(pick):
            d["demo_id"] = f"{b}:{i:02d}:{p}:r{d['frags'][0][0]}@{d['frags'][0][1] - rows[d['frags'][0][0]]['offset']}"
        res["buckets"][b] = {"pass": p, "eligible": len(ok), "excluded": stats, "n": len(pick),
                             "tokens": int(sum(d["tokens"] for d in pick)),
                             "response_tokens": int(sum(d["response_tokens"] for d in pick)),
                             "selftraces": sum(d["source"].startswith(SELFTRACE) for d in pick),
                             "truncated": sum("truncated_from" in d for d in pick),
                             "demos": [{k: d[k] for k in ("demo_id", "source", "subarea", "sub_id", "tokens",
                                                          "response_tokens", "frags", "at_row_start")}
                                       | ({"truncated_from": d["truncated_from"]} if "truncated_from" in d else {})
                                       for d in pick]}
        if len(pick) < DEMO_N:
            env.log(f"{b}: only {len(pick)} demos (< {DEMO_N}): too few response-shaped train documents", STAGE, "WARN")
        x = res["buckets"][b]
        env.log(f"{b}: {x['n']} demos ({x['selftraces']} self-traces), {x['tokens']:,} tok, "
                f"{x['response_tokens']:,} response tok; eligible {len(ok)}; excluded {stats}", STAGE)
    env.atomic_json(OUT / "demos.json", res)
    return res


def corpus(a) -> None:
    dm = json.load(open(OUT / "demos.json"))
    from ops.dom import corpus_spec_dom as spec
    buckets = [b for b in spec.BUCKETS if b in (a.buckets or TIER1)]
    name = f"corpus_easyep_{a.group}" if a.group else "corpus_easyep"
    out = DOM / name
    if (out / "manifest.json").exists() and not a.force:
        raise SystemExit(f"{out} exists (--force rewrites it; never while its s03 pass runs)")
    out.mkdir(parents=True, exist_ok=True)
    src = {}
    plan, T = [], 0
    for b in buckets:
        for d in dm["buckets"][b]["demos"]:
            n = d["tokens"]
            L = -(-n // PAD_TO) * PAD_TO
            plan.append((b, d, T, n, L))
            T += L
    arrs = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=dt, shape=(T,))
            for k, dt in (("tokens", np.uint32), ("types", np.int8), ("valid", np.uint8))}
    man0 = None
    with open(out / "rows.jsonl.tmp", "w") as fh:
        for i, (b, d, off, n, L) in enumerate(plan):
            p = dm["buckets"][b]["pass"]
            if p not in src:
                C = DOM / f"corpus_{p}"
                src[p] = {k: np.load(C / f"{k}.npy", mmap_mode="r") for k in ("tokens", "types", "valid")}
                man0 = man0 or json.load(open(C / "manifest.json"))
            pos = off
            for _, s, k in d["frags"]:
                for kk in arrs:
                    arrs[kk][pos:pos + k] = src[p][kk][s:s + k]
                pos += k
            assert pos == off + n and int(arrs["tokens"][off]) == BOS
            arrs["tokens"][off + n:off + L] = EOS
            arrs["types"][off + n:off + L] = TEXT
            arrs["valid"][off + n:off + L] = 0
            fh.write(json.dumps({"row": i, "bucket": b, "bucket_id": spec.BUCKETS.index(b), "offset": off, "length": L,
                                 "heldout": False, "n_valid": int(arrs["valid"][off:off + L].sum()), "n_doc": n,
                                 "docs": [{"source": d["source"], "tokens": n}] + ([{"source": "pad", "tokens": L - n}] if L > n else []),
                                 "images": [], "subarea": d["subarea"], "sub_id": d["sub_id"], "demo_id": d["demo_id"],
                                 "src_pass": p, "src_frags": d["frags"]}) + "\n")
    for m in arrs.values():
        m.flush()
    (out / "rows.jsonl.tmp").replace(out / "rows.jsonl")
    # NOTE: no "pass_buckets" key, so d2lib.pass_dirs / replay_filter._pass_of never mistake this for a D1 pass
    env.atomic_json(out / "manifest.json", {
        **{k: man0[k] for k in ("spec_hash", "generator_policy", "revision", "draw")},
        "buckets": spec.BUCKETS, "subareas": spec.SUBAREAS, "easyep_buckets": buckets, "rows": len(plan),
        "tokens_total": T, "tokens_valid": int(sum(p_[3] for p_ in plan)), "max_row_len": max(p_[4] for p_ in plan),
        "longctx_lengths": [], "demos_ts": dm["ts"], "pad_to": PAD_TO, "ts": env.now()})
    env.log(f"{out}: {len(plan)} demo rows, {T:,} tokens (pads included), buckets {buckets}", STAGE)


def plan_cost(a) -> dict:
    dm = json.load(open(OUT / "demos.json"))
    per = {b: x["tokens"] for b, x in dm["buckets"].items()}
    tok = sum(-(-d["tokens"] // PAD_TO) * PAD_TO for x in dm["buckets"].values() for d in x["demos"])
    out = {"ts": env.now(), "tokens_by_bucket": per, "tokens_padded_total": tok,
           "scratch_gb": tok * S03_BYTES_PER_TOKEN / 1e9,
           "gpu_hours_one_pass": S03_FIXED_H + S03_H_PER_MTOK * tok / 1e6,
           "rate_source": "s03 logs: p1 1,638,400 tok in 12.1 h, t3 819,200 tok in 7.3 h -> 2.5 h fixed + 5.86 h/Mtok",
           "disk_free_gb": __import__("shutil").disk_usage(DOM).free / 1e9}
    env.atomic_json(OUT / "plan.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "tokens_by_bucket"}, indent=1))
    return out


# ----------------------------------------------------------------------------- selection
def _write(b: str, arm: str, scores: np.ndarray, sup, protect, prov: dict) -> dict:
    keep, info = {}, []
    for L in range(N_LAYERS):
        k, i = select_layer(scores[L], guard_ids(L, sup, protect))
        keep[str(L)] = k
        info.append(i)
    d = OUT / b
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{arm}.scores.npy", scores)
    sha = keepset.save(d / f"{arm}.keep.json", keep, arm=f"{b}:{arm}", method=prov["method"], n_keep_per_layer=N_KEEP)
    a192, a192_sha = keepset.load(A192_KEEP)
    rep = {"ts": env.now(), "bucket": b, "arm": arm, "sha256": sha, **prov,
           "guards": {"total": int(sum(x["n_guard"] for x in info)),
                      "dropped_by_easyep_total": int(sum(len(x["guards_dropped_by_easyep"]) for x in info)),
                      "dropped_by_easyep_mean_per_layer": float(np.mean([len(x["guards_dropped_by_easyep"]) for x in info])),
                      "per_layer": {str(L): x for L, x in enumerate(info)}},
           "overlap_A192": {"sha256": a192_sha, **overlap(keep, a192)}}
    pick, pf = anchored_pick(b)
    if pf is not None:
        pk, psha = keepset.load(pf)
        rep["overlap_anchored_pick"] = {"arm": pick, "sha256": psha, **overlap(keep, pk)}
    else:
        f4 = ANCHORED / b / "anc_k4.json"
        rep["overlap_anchored_pick"] = None
        if f4.exists():
            pk, psha = keepset.load(f4)
            rep["overlap_anchored_k4_no_pick"] = {"arm": "anc_k4", "sha256": psha, **overlap(keep, pk)}
    env.atomic_json(d / f"{arm}.json", rep)
    env.log(f"{b}:{arm} {sha[:8]}  shared with A192 {rep['overlap_A192']['mean_shared']:.1f}/192, guards EASY-EP "
            f"would drop {rep['guards']['dropped_by_easyep_total']} of {rep['guards']['total']}", STAGE)
    return rep


def select(a) -> None:
    """From the recorder pass: per bucket, sum over its demo rows of gn * max(1 - cos, 0)."""
    sup, protect, _ = guards_all()
    dirs = [d for d in sorted(DOM.glob("corpus_easyep*")) if (d / "manifest.json").exists()]
    if not dirs:
        raise SystemExit("no demo corpus (run `corpus`, then the recorder pass)")
    variants = {"easyep": ("full", True), "easyep_collapsed": ("collapsed", True), "easyep_nos": (None, False)}
    for C in dirs:
        man = json.load(open(C / "manifest.json"))
        S = DOM / ("s03_" + C.name[len("corpus_"):])
        led = json.load(open(S / "s03_ledger.json")) if (S / "s03_ledger.json").exists() else {}
        if not all(led.get(f"layer/L{L}", {}).get("done") for L in range(N_LAYERS)):
            raise SystemExit(f"{S}: recorder pass not complete")
        rows = [json.loads(l) for l in open(C / "rows.jsonl")]
        valid = np.load(C / "valid.npy", mmap_mode="r").astype(bool)
        T = man["tokens_total"]
        for b in [x for x in man["easyep_buckets"] if not a.buckets or x in a.buckets]:
            m = np.zeros(T, bool)
            for r in rows:
                if r["bucket"] == b:
                    m[r["offset"]:r["offset"] + r["length"]] = True
            m &= valid
            sc = {v: np.zeros((N_LAYERS, E)) for v in variants}
            cos_stats = []
            for L in range(N_LAYERS):
                idx = np.load(S / "routing" / f"L{L}_idx.npy", mmap_mode="r")
                gn = np.load(S / "easyep" / f"L{L}_gn.npy", mmap_mode="r")
                cos = np.load(S / "easyep" / f"L{L}_cos.npy", mmap_mode="r")
                ii, gg, cc = np.asarray(idx[m]), np.asarray(gn[m]), np.asarray(cos[m])
                assert np.isfinite(gg).all() and np.isfinite(cc).all(), f"L{L}: non-finite recorder values"
                for v, (col, use) in variants.items():
                    sc[v][L] = easyep_scores(ii, gg, cc[:, COS_COL[col]] if use else None, np.ones(len(ii), bool))
                cos_stats.append({"full_mean": float(cc[:, 0].mean()), "collapsed_mean": float(cc[:, 1].mean()),
                                  "s_full_mean": float(token_weight(cc[:, 0]).mean())})
            dm = json.load(open(OUT / "demos.json"))["buckets"][b]
            prov = {"demos": [{k: d[k] for k in ("demo_id", "source", "tokens", "response_tokens")} for d in dm["demos"]],
                    "demo_tokens": int(m.sum()), "recorder": str(S), "cos_by_layer": cos_stats,
                    "routing_check_vs_d1": routing_check(b, rows, S, dm)}
            for v, (col, use) in variants.items():
                prov["method"] = (f"EASY-EP: sum_t g*||e|| * max(1-cos,0), cos over the {col} stream" if use
                                  else "EASY-EP without token weight (s_t = 1) on the same demos")
                _write(b, v, sc[v], sup, protect, prov)


def routing_check(b: str, rows: list[dict], S: Path, dm: dict, layers=(0, 20, 39)) -> dict:
    """Demos whose first fragment started its D1 row see the same causal context in both passes: their routing
    ids should match D1's (agreement, not bit-equality: kernels may tile differently at another length)."""
    agree, n = 0, 0
    for r in rows:
        if r["bucket"] != b:
            continue
        d = next(x for x in dm["demos"] if x["demo_id"] == r["demo_id"])
        if not d["at_row_start"]:
            continue
        _, s, k = d["frags"][0]
        for L in layers:
            a = np.load(S / "routing" / f"L{L}_idx.npy", mmap_mode="r")[r["offset"]:r["offset"] + k]
            ref = np.load(DOM / f"s03_{r['src_pass']}" / "routing" / f"L{L}_idx.npy", mmap_mode="r")[s:s + k]
            agree += int((np.sort(np.asarray(a), 1) == np.sort(np.asarray(ref), 1)).all(1).sum()); n += k
    return {"layers": list(layers), "tokens": n, "top6_set_agreement": agree / n if n else None}


def bank(a) -> None:
    """Proxy, runnable from the D1 banks alone: sum of g*||f|| over the bucket's whole train corpus (bank `sum`,
    s_t = 1, 409,600 tokens instead of 25 demos). Isolates what the token weight and the demo restriction change."""
    from ops.dom import d2lib as D
    sup, protect, _ = guards_all()
    for b in a.buckets or TIER1:
        p = pass_of(b)
        sc = np.zeros((N_LAYERS, E))
        for L in range(N_LAYERS):
            with np.load(DOM / f"s03_{p}" / "bank" / f"L{L}.npz") as z:
                sa = [str(x) for x in z["subareas"]]
                ids = [i for i, n in enumerate(sa) if n.split("/")[0] == b]
                sc[L] = z["sum"][ids].sum(0)
        assert (sc.sum(1) > 0).all(), f"{b}: empty bank rows"
        _write(b, "easyep_bank", sc, sup, protect,
               {"method": "EASY-EP proxy: sum over the bucket's D1 train tokens of g*||f|| (bank `sum`), no token "
                          "weight, no demo restriction", "bank": str(DOM / f"s03_{p}" / "bank"), "n_layers": D.N_LAYERS})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["demos", "corpus", "plan", "select", "bank"])
    ap.add_argument("--buckets", nargs="*")
    ap.add_argument("--group", help="corpus: write corpus_easyep_<group> (split the recorder pass for disk)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"demos": demos, "corpus": corpus, "plan": plan_cost, "select": select, "bank": bank}[a.cmd](a)


if __name__ == "__main__":
    main()
