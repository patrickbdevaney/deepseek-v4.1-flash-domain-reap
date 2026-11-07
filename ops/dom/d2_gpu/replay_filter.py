#!/usr/bin/env python3
"""D2 replay FILTER (D2_METHOD item 7): teacher-routed held-out replay of the D2 candidates, kl32 / top-1 with
row-grouped bootstrap CIs. A filter, not the ranker: candidates whose CIs overlap are ties settled by the task
evals (E). GPU. NOT RUN YET: written for the next GPU window.

Reuses the s04 machinery unchanged (ops/lib/replay: proper re-routing over the survivors, layer-streamed, one
process per (layer, arm batch); ops/lib/metrics + stats: kl32 vs the cached s03 teacher top-32, cluster = row).
The domain runs execute with V41_DOM=1 V41_DOM_PASS=<pass>, so s03.Corpus / s03.D point at dom/corpus_<pass> and
dom/s03_<pass>: the teacher cache is dom/s03_<pass>/teacher and the Engram rows are the PARKED copy that
`passes.py retire` symlinked at dom/s03_<pass>/engram (cold store) -- nothing is recomputed.

Targets (one run dir each under $V41_WORK/dom/d2/replay/<target>):
  <bucket>        the bucket's held-out rows (default: all of them, stratified to --tokens), arms = A192 + shortlist
  poly_<pass>     --poly-rows held-out rows per tier-1 bucket of that pass, arms = A192 + polymath candidates
  general         A192 corpus, SCREEN split, general bucket rows (V41_DOM unset), arms = A192 + every shortlisted
                  candidate of every bucket + polymath; gives the "general kl32 <= A192 + 0.10" gate
  control         (once per pass, 1 row) all 384 experts must reproduce the teacher (gates.control), as s04 does

  plan      CPU: shortlist, rows, tokens, disk, expected GPU time per target -> replay/plan.json
  run       GPU: every target in the plan (or --targets), resumable; units are separate processes (s04 pattern)
  decide    CPU: per bucket kl32/top1/dNLL by bucket and by sub-area, paired vs A192 (row-cluster bootstrap),
            the D2 gate (domain d_kl32 CI hi < 0 vs A192 AND general kl32 <= A192 + 0.10), tie groups
            -> replay/filter.json

EXPECTED COST (measured s10/s04 replay rate ~1,800 tok/s per arm-layer on Thor, 2026-10-24):
  per arm   40 layers x tokens / 1,800 tok/s  -> 40,960 tokens (default --tokens) = ~15 min
  bucket    (1 + 6 shortlisted [+ shrink/fair]) arms ~ 1.75-2.25 h;   10 buckets ~ 18-22 h (original run)
  general   ~65 arms x 16k tokens ~ 6 h;   polymath 3 passes x 4 arms x ~16k-20k tokens ~ 1.5 h
  disk      41 KB/token/arm stream (replay.bytes_per_arm), arms batched to fit (orchestrate.batches);
            a bucket target at 40,960 tokens is ~1.7 GB/arm, ~12 GB for 7 arms; streams dropped after the head
  memory    as s04 arbitrate: one decoder layer (384 MXFP4 experts dequantized per arm) + one row on the GPU,
            peak ~25-30 GiB unified; requires the GPU window: MemAvailable >= 100 GiB (--min-avail-gib), no server,
            GPU_WINDOW.lock taken by this script. Never beside D1, a server or another model-loading job.

Run (detached, in a GPU window only):
  setsid nohup systemd-run --user --scope -p MemoryMax=110G .venv/bin/python ops/dom/d2_gpu/replay_filter.py run \\
      > ops/logs/d2_replay.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ops.lib import env, keepset, orchestrate, rowsets  # noqa: E402

STAGE = "d2_replay"
D2 = env.WORK / "dom" / "d2"
ROOT = D2 / "replay"
A192_KEEP = env.WORK / "s04" / "keepset.json"
TOK_PER_S = 1800.0
SHORTLIST = ["hope_w45_pf4_st1", "mass_w45_pf4_st1", "reap_w45_pf4_st1", "hope_w35_pf4_st1", "hope_w60_pf4_st1",
             "hope_w45_pf4_st0"]
GENERAL_MARGIN = 0.10
DEFAULT_W = 0.45
GENERAL_BACKUP = 2        # domain_ok arms beyond the tie set whose general kl32 is still measured


def _manifest() -> dict:
    return json.load(open(D2 / "manifest.json"))


def _pass_of(bucket: str) -> str:
    for c in sorted((env.WORK / "dom").glob("corpus_*")):
        m = c / "manifest.json"
        if m.exists() and bucket in json.load(open(m)).get("pass_buckets", []):
            return c.name[len("corpus_"):]
    raise SystemExit(f"{bucket}: no pass corpus")


def _strat(rows: list[dict], budget: int, key: str = "subarea") -> list[int]:
    """Round-robin over sub-areas (visit order inside each) until the token budget: every sub-area gets a row first."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r.get(key) or "-", []).append(r)
    out, used, k = [], 0, 0
    while used < budget and any(len(v) > k for v in by.values()):
        for v in by.values():
            if len(v) > k and used < budget:
                out.append(v[k]["row"]); used += v[k]["length"]
        k += 1
    return sorted(out)


def plan(a) -> dict:
    man = _manifest()
    cands = {(c["bucket"], c["name"]): c for c in man["candidates"]}
    buckets = a.buckets or sorted({b for b, _ in cands if b != "polymath"})
    a192 = keepset.load(A192_KEEP)
    targets = {}
    general_arms = [{"name": "A192", "keep_file": str(A192_KEEP), "keep_sha256": a192[1]}]
    for b in buckets:
        p = _pass_of(b)
        rows, _ = rowsets.load_rows(env.WORK / "dom" / f"corpus_{p}")
        held = [r for r in rows if r["heldout"] and r["bucket"] == b]
        ids = _strat(held, a.tokens)
        names = [n for n in (a.cands or SHORTLIST) if (b, n) in cands]
        # shrink/fair only at the default weight (2026-11-03: replay time; their w35/w60 variants cross two axes)
        names += [n for (bb, n) in cands if bb == b and n not in names and (n.endswith("_diag") or
                  (("_shr" in n or n.startswith("fair_")) and n.split("_")[1] == f"w{round(DEFAULT_W * 100)}"))]
        arms = [{"name": "A192", "keep_file": str(A192_KEEP), "keep_sha256": a192[1]}] + \
               [{"name": n, "keep_file": str(D2 / cands[(b, n)]["file"]), "keep_sha256": cands[(b, n)]["sha256"]} for n in names]
        T = sum(rows[i]["length"] for i in ids)
        targets[b] = {"kind": "domain", "pass": p, "rows": ids, "tokens": T, "arms": arms}
        general_arms += [{**x, "name": f"{b}:{x['name']}"} for x in arms[1:]]
    poly = [(n, c) for (b, n), c in cands.items() if b == "polymath"]
    for p in sorted({_pass_of(b) for b in buckets}):
        rows, man_p = rowsets.load_rows(env.WORK / "dom" / f"corpus_{p}")
        ids = []
        for b in man_p["pass_buckets"]:
            ids += _strat([r for r in rows if r["heldout"] and r["bucket"] == b], a.poly_rows * 8192)
        targets[f"control_{p}"] = {"kind": "control", "pass": p, "rows": ids[:1], "tokens": rows[ids[0]]["length"],
                                   "arms": [{"name": "control", "keep_file": None, "keep_sha256": None}]}
        if poly:
            targets[f"poly_{p}"] = {"kind": "domain", "pass": p, "rows": sorted(ids),
                                    "tokens": sum(rows[i]["length"] for i in ids),
                                    "arms": [general_arms[0]] + [{"name": f"polymath:{n}", "keep_file": str(D2 / c["file"]),
                                                                  "keep_sha256": c["sha256"]} for n, c in poly]}
    general_arms += [{"name": f"polymath:{n}", "keep_file": str(D2 / c["file"]), "keep_sha256": c["sha256"]} for n, c in poly]
    rows9, man9 = rowsets.load_rows(env.WORK / "corpus")
    screen = [r for r in rowsets.split_rows(rows9, man9["spec_hash"], "screen") if r["bucket"] == "general"]
    gids = sorted(r["row"] for r in screen)
    targets["general"] = {"kind": "general", "pass": None, "rows": gids, "tokens": sum(rows9[i]["length"] for i in gids),
                          "arms": general_arms}
    for t in targets.values():                  # identical keep-sets are replayed once (polymath rr4/rr6 == control)
        seen, arms, alias = {}, [], {}
        for x in t["arms"]:
            k = x["keep_sha256"]
            if k is not None and k in seen:
                alias[x["name"]] = seen[k]
                continue
            seen.setdefault(k, x["name"]) if k is not None else None
            arms.append(x)
        t["arms"], t["aliases"] = arms, alias
        t["gpu_hours"] = len(t["arms"]) * 40 * t["tokens"] / TOK_PER_S / 3600
        t["disk_gib_per_arm"] = t["tokens"] * (4 * 5120 * 2 + 16) / 2**30
    out = {"ts": env.now(), "manifest_ts": man["ts"], "tokens_budget": a.tokens, "targets": targets,
           "total_gpu_hours": sum(t["gpu_hours"] for t in targets.values())}
    env.atomic_json(ROOT / "plan.json", out)
    for k, t in targets.items():
        print(f"{k:24s} {len(t['rows']):3d} rows {t['tokens']:>8,} tok {len(t['arms']):3d} arms ~{t['gpu_hours']:.1f} GPU-h")
    print(f"total ~{out['total_gpu_hours']:.1f} GPU-h")
    return out


def _env_for(t: dict) -> dict:
    e = {k: v for k, v in os.environ.items() if k not in ("V41_DOM", "V41_DOM_PASS")}
    if t["kind"] != "general":
        e.update(V41_DOM="1", V41_DOM_PASS=t["pass"])
    return e


def _sub(argv: list[str], t: dict, a, *, reclaim: bool) -> None:
    """One unit/target subprocess in the target's env; reclaim after it only when asked or memory is short."""
    env.log(f"start {' '.join(argv)}", STAGE)
    t0 = time.time()
    rc = subprocess.run([sys.executable, __file__, *argv, "--kernels", a.kernels, "--min-avail-gib",
                         str(a.min_avail_gib)], env=_env_for(t)).returncode
    if reclaim or env.mem_available_gib() < a.min_avail_gib:
        env.reclaim(STAGE, expect_gib=a.min_avail_gib)
    if rc:
        raise SystemExit(f"{' '.join(argv)} failed rc={rc}; re-run resumes")
    env.log(f"done {' '.join(argv)} in {time.time() - t0:.0f}s", STAGE)


def _run_layer_major(a, pl: dict, keys: list[str]) -> None:
    """All targets of one env group (one pass, or general) advance layer by layer together: each layer's teacher
    weights are read once from the backbone (on the cold store, ~110 s per layer) and the other targets' units
    hit the page cache. drop_caches runs at layer boundaries and whenever MemAvailable < --min-avail-gib.
    Needs every arm's stream of the group on disk at once (checked); no arm batching."""
    from ops.lib import replay
    for k in keys:
        _sub(["target", k, "--phase", "prep"], pl["targets"][k], a, reclaim=True)
    runs = {k: json.load(open(ROOT / k / "run.json")) for k in keys}
    need = sum(len(r["arms"]) * replay.bytes_per_arm(pl["targets"][k]["tokens"]) for k, r in runs.items())
    free = orchestrate.disk_free(ROOT)
    if need > free - (a.disk_margin_gib << 30):
        raise SystemExit(f"layer-major group {keys}: streams {need / 2**30:.0f} GiB > free {free / 2**30:.0f} GiB "
                         f"- {a.disk_margin_gib} GiB margin; split the group with --targets")
    for L in range(40):
        for k in keys:
            led = ROOT / k / "ledger.json"
            done = json.load(open(led)) if led.exists() else {}
            if all(_led_done(done, f"{x['name']}/L{L}") for x in runs[k]["arms"]):
                continue
            _sub(["unit", k, "layer", "--layer", str(L)], pl["targets"][k], a, reclaim=False)
        env.reclaim(STAGE, expect_gib=a.min_avail_gib)
    for k in keys:
        _sub(["target", k, "--phase", "finish"], pl["targets"][k], a, reclaim=True)


def _led_done(led: dict, key: str) -> bool:
    return bool(led.get(key, {}).get("done"))


def run(a):
    pl = json.load(open(ROOT / "plan.json"))
    lock = env.WORK / "GPU_WINDOW.lock"
    if lock.exists():
        raise SystemExit(f"GPU_WINDOW.lock held: {lock.read_text().strip()}")
    if env.mem_available_gib() < a.min_avail_gib:
        raise SystemExit(f"MemAvailable {env.mem_available_gib():.0f} GiB < {a.min_avail_gib}")
    lock.write_text(f"{os.getpid()} {env.now()} d2 replay filter\n")
    try:
        if a.layer_major:
            groups: dict = {}
            for k in a.targets or pl["targets"]:
                t = pl["targets"][k]
                groups.setdefault("general" if t["kind"] == "general" else t["pass"], []).append(k)
            for g in sorted(groups, key=lambda g: g == "general"):          # general last (contender pruning)
                _run_layer_major(a, pl, sorted(groups[g], key=lambda k: (not k.startswith("control_"), k)))
            return
        order = sorted(pl["targets"], key=lambda k: (not k.startswith("control_"), k == "general", k))
        for k in a.targets or order:
            t = pl["targets"][k]
            rc = subprocess.run([sys.executable, __file__, "target", k, "--kernels", a.kernels,
                                 "--min-avail-gib", str(a.min_avail_gib), "--max-batch", str(a.max_batch)],
                                env=_env_for(t)).returncode
            if rc:
                raise SystemExit(f"target {k} failed rc={rc}; re-run resumes")
    finally:
        lock.unlink(missing_ok=True)


def _contenders(pl: dict, draws: int, backup: int = GENERAL_BACKUP) -> set | None:
    """General-target arms that can still matter for the gate, from the finished domain targets: per bucket, the
    domain_ok arms (d_kl32 vs A192 CI hi < 0) tied with the lowest-kl32 one, plus the next `backup` domain_ok arms
    in kl32 order (so a contender failing the general gate has a successor measured). Diagnostic and polymath arms
    are always kept. An arm that fails domain_ok cannot pass the gate, so its general kl32 is not needed -- except
    in a bucket with NO domain_ok arm (underpowered at 5 rows), where the 1 + `backup` lowest-kl32 arms are kept.
    None = some domain target is unfinished: measure every arm."""
    from ops.lib import metrics
    keep = {"A192"}
    for k, t in pl["targets"].items():
        if t["kind"] != "domain":
            continue
        if k.startswith("poly_"):
            keep |= {x["name"] for x in t["arms"] if x["name"] != "A192"}
            continue
        if not (ROOT / k / "run.json").exists() or \
                not all((ROOT / k / x["name"].replace(":", "__") / "metrics").exists() for x in t["arms"]):
            return None
        C = metrics.corpus(env.WORK / "dom" / f"corpus_{t['pass']}")
        T = metrics.teacher(env.WORK / "dom" / f"s03_{t['pass']}")
        data = {x["name"]: metrics.load_arm(ROOT / k, x["name"].replace(":", "__"), t["rows"], C, T) for x in t["arms"]}
        ok = [n for n in data if n != "A192" and not n.endswith("_diag") and metrics.compare(data[n], data["A192"], n_draws=draws)["d_kl32"]["hi"] < 0]
        ok.sort(key=lambda n: float(np.mean(data[n]["kl32"])))
        tied = [n for n in ok[1:] if not metrics.compare(data[n], data[ok[0]], n_draws=draws)["d_kl32"]["excludes_zero"]]
        if not ok:
            # 5 rows per target: a real in-domain gain can miss CI hi < 0 (one bucket: every arm -0.03..-0.08
            # kl32, CI crossing 0). Measure general for the 1 + backup lowest-kl32 arms anyway so a later,
            # better-powered domain verdict (E evals) is not left without its general half.
            ok = sorted((n for n in data if n != "A192" and not n.endswith("_diag")),
                        key=lambda n: float(np.mean(data[n]["kl32"])))[:1 + backup]
            env.log(f"general contenders: {k} has no domain_ok arm; keeping lowest-kl32 {ok}", STAGE)
            chosen = list(ok)
        else:
            chosen = [ok[0]] + tied
        chosen += [n for n in ok if n not in chosen][:backup]
        chosen += [x["name"] for x in t["arms"] if x["name"].endswith("_diag")]
        keep |= {f"{k}:{n}" for n in chosen}
    return keep


def target(a):
    """One target, in a process whose V41_DOM / V41_DOM_PASS already match it (set by `run`)."""
    from ops.lib import replay
    s03 = replay.s03mod()
    pl = json.load(open(ROOT / "plan.json"))
    t = pl["targets"][a.name]
    assert (t["kind"] == "general") == (os.environ.get("V41_DOM") != "1"), "env does not match the target"
    root = ROOT / a.name
    if t["kind"] == "general" and not a.all_general:
        keep = _contenders(pl, a.draws)
        if keep is not None:
            t = {**t, "arms": [x for x in t["arms"] if x["name"] in keep]}
            env.atomic_json(ROOT / "general_arms.json", {"ts": env.now(), "arms": [x["name"] for x in t["arms"]],
                                                         "rule": _contenders.__doc__})
            env.log(f"general: {len(t['arms'])} contender arms of {len(pl['targets'][a.name]['arms'])}", STAGE)
    if a.phase == "finish":
        run_ = replay.Run(root)
        names = [x["name"] for x in run_.arms]
        todo = [n for n in names if not run_.led.done(f"{n}/head")]
        if todo:
            orchestrate.run_unit(Path(__file__), ["unit", a.name, "head", "--arms", ",".join(todo)], stage=STAGE,
                                 gpu=True, min_avail_gib=a.min_avail_gib)
        run_ = replay.Run(root)
        for n in names:
            replay.drop_streams(run_, n)
        return
    arms = []
    for x in t["arms"]:
        keep = None
        if x["keep_file"]:
            keep, _ = keepset.load(Path(x["keep_file"]), expect=x["keep_sha256"])
        arms.append({"name": x["name"].replace(":", "__"), "keep": keep})
    man = json.load(open(s03.CORPUS / "manifest.json"))
    run_ = replay.prepare(root, t["rows"], arms, man["spec_hash"])
    names = [x["name"] for x in run_.arms]
    todo = [n for n in names if not run_.led.done(f"{n}/head")]
    me = Path(__file__)
    if not run_.led.done("_embed"):
        orchestrate.run_unit(me, ["unit", a.name, "embed", "--kernels", a.kernels], stage=STAGE, gpu=True,
                             min_avail_gib=a.min_avail_gib)
    if a.phase == "prep":
        return
    per = replay.bytes_per_arm(run_.T)
    for b in orchestrate.batches(todo, per, orchestrate.disk_free(root), 60 << 30, cap=a.max_batch):
        al = ",".join(b)
        for L in range(s03.N_LAYERS):
            if all(run_.led.done(f"{n}/L{L}") for n in b):
                continue
            orchestrate.run_unit(me, ["unit", a.name, "layer", "--layer", str(L), "--arms", al, "--kernels", a.kernels],
                                 stage=STAGE, gpu=True, min_avail_gib=a.min_avail_gib)
        orchestrate.run_unit(me, ["unit", a.name, "head", "--arms", al], stage=STAGE, gpu=True, min_avail_gib=a.min_avail_gib)
        run_ = replay.Run(root)
        for n in b:
            replay.drop_streams(run_, n)


def unit(a):
    from ops.lib import replay
    run_ = replay.Run(ROOT / a.name)
    arms = a.arms.split(",") if a.arms else None
    if a.unit == "embed":
        replay.unit_embed(run_, a.kernels)
    elif a.unit == "layer":
        replay.unit_layer(run_, a.layer, a.kernels, 8192, arms)
    else:
        replay.unit_head(run_, arms)


def decide(a):
    from ops.lib import gates, metrics, stats
    pl = json.load(open(ROOT / "plan.json"))
    g = gates.load()[0]
    res = {"rule": "D2 gate: domain d_kl32 vs A192 CI hi < 0 (row-cluster bootstrap) AND general kl32 mean <= A192 "
                   f"general kl32 + {GENERAL_MARGIN}. Among gate-passing candidates, every one whose paired d_kl32 vs "
                   "the lowest-kl32 passing candidate has a CI containing 0 is TIED with it: ties go to the task evals "
                   "(D2_METHOD item 7: kl32 is a filter, not the ranker).", "control": {}, "general": None, "buckets": {}}
    for k, t in pl["targets"].items():
        if t["kind"] == "control":
            C = metrics.corpus(env.WORK / "dom" / f"corpus_{t['pass']}")
            T = metrics.teacher(env.WORK / "dom" / f"s03_{t['pass']}")
            d = metrics.load_arm(ROOT / k, "control", t["rows"], C, T)
            res["control"][k] = gates.control(metrics.summarize(d, n_draws=a.draws), g)
    gen = {}
    if "general" in pl["targets"] and (ROOT / "general" / "run.json").exists():
        t = pl["targets"]["general"]
        C, T = metrics.corpus(env.WORK / "corpus"), metrics.teacher(env.WORK / "s03")
        for x in t["arms"]:
            n = x["name"].replace(":", "__")
            if (ROOT / "general" / n / "metrics").exists():
                gen[x["name"]] = metrics.summarize(metrics.load_arm(ROOT / "general", n, t["rows"], C, T),
                                                   n_draws=a.draws)["overall"]
        res["general"] = gen
    for k, t in pl["targets"].items():
        if t["kind"] != "domain" or not (ROOT / k / "run.json").exists():
            continue
        C = metrics.corpus(env.WORK / "dom" / f"corpus_{t['pass']}")
        T = metrics.teacher(env.WORK / "dom" / f"s03_{t['pass']}")
        sub = {r["row"]: r.get("subarea") for r in C["rows"]}
        data = {}
        for x in t["arms"]:
            n = x["name"].replace(":", "__")
            if (ROOT / k / n / "metrics").exists():
                data[x["name"]] = metrics.load_arm(ROOT / k, n, t["rows"], C, T)
        if "A192" not in data:
            continue
        base = data["A192"]
        out = {"arms": {}}
        for n, d in data.items():
            s = metrics.summarize(d, n_draws=a.draws)
            r = {"overall": s["overall"], "by_subarea": {}}
            sa = np.array([sub[int(i)] for i in d["row"]], dtype=object)
            for q in sorted(set(sa.tolist())):
                m = sa == q
                r["by_subarea"][q] = {"kl32": stats.cluster_bootstrap_mean(d["kl32"][m], d["row"][m], n_draws=a.draws),
                                      "top1": stats.cluster_bootstrap_mean(d["agree"][m].astype(float), d["row"][m],
                                                                           n_draws=a.draws)}
            if n != "A192":
                r["vs_A192"] = metrics.compare(d, base, n_draws=a.draws)
                gk = gen.get(n if ":" in n else f"{k}:{n}")
                ga = gen.get("A192")
                r["general_kl32"] = gk["kl32"]["mean"] if gk else None
                r["general_ok"] = (gk["kl32"]["mean"] <= ga["kl32"]["mean"] + GENERAL_MARGIN) if gk and ga else None
                r["general_measured"] = gk is not None      # not measured = not a contender (see _contenders)
                r["domain_ok"] = bool(r["vs_A192"]["d_kl32"]["hi"] < 0)
                r["gate"] = bool(r["domain_ok"] and r["general_ok"])
            out["arms"][n] = r
        out["diagnostic"] = {n: {"kl32": r["overall"]["kl32"], "vs_A192": r.get("vs_A192"), "general_kl32": r.get("general_kl32")}
                             for n, r in out["arms"].items() if n.endswith("_diag")}
        passing = sorted([n for n, r in out["arms"].items() if r.get("gate") and not n.endswith("_diag")],
                         key=lambda n: out["arms"][n]["overall"]["kl32"]["mean"])
        out["passing"] = passing
        if passing:
            best = passing[0]
            out["ties_with_best"] = [best] + [n for n in passing[1:]
                                             if not metrics.compare(data[n], data[best], n_draws=a.draws)["d_kl32"]["excludes_zero"]]
        res["buckets"][k] = out
    env.atomic_json(ROOT / "filter.json", res)
    for k, o in res["buckets"].items():
        print(f"{k}: passing {o.get('passing')} ties {o.get('ties_with_best')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "target", "unit", "decide"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("unit", nargs="?", choices=["embed", "layer", "head"])
    ap.add_argument("--buckets", nargs="*")
    ap.add_argument("--cands", nargs="*", help=f"shortlist override (default {SHORTLIST} + any shrink/fair)")
    ap.add_argument("--tokens", type=int, default=40_960, help="held-out token budget per bucket target")
    ap.add_argument("--poly-rows", type=int, default=1, help="held-out rows per bucket in a polymath target")
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--arms")
    ap.add_argument("--kernels", choices=["tilelang", "torch"], default="tilelang")
    ap.add_argument("--min-avail-gib", type=float, default=100.0)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--all-general", action="store_true", help="target general: measure every arm, not only contenders")
    ap.add_argument("--phase", choices=["all", "prep", "finish"], default="all", help="target: prep = run dir + embed")
    ap.add_argument("--layer-major", action="store_true", help="run: all targets of a pass advance layer by layer")
    ap.add_argument("--disk-margin-gib", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "target":
        env.assert_no_gpu_server(STAGE)
    {"plan": plan, "run": run, "target": target, "unit": unit, "decide": decide}[a.cmd](a)


if __name__ == "__main__":
    main()
