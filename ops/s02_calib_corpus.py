#!/usr/bin/env python3
"""s02: build the token-balanced calibration corpus (spec and final mix: ops/corpus_spec.py).

Rows. Eight pools are packed into 8,192-token rows, one pool per row. The `longctx` pool holds
whole documents of 32k-128k tokens, one per row and never concatenated, so rows are RAGGED: all
per-token arrays are flat over the corpus and rows.jsonl gives each row's offset and length.

Output, under $V41_WORK/corpus/ (default $HOME/v41-work/corpus):
  tokens.npy   [T] uint32   token ids (vocab 129,280 does not fit uint16)
  types.npy    [T] int8     -1 text, 0..3 image_processor IMAGE_START/IMAGE/NEW_LINE/END
  valid.npy    [T] uint8    1 for real tokens, 0 for the tail pad of a vision row
  rows.jsonl   one record per row IN VISIT ORDER: row, bucket, offset, length, heldout, docs, images
  sources.json per source: declared + live licence, generator, status, rows read / kept, tokens
  manifest.json achieved token mix, counts, spec hash; written last (it marks completion)
  images/      original image bytes for vision rows (s03 re-reads them with load_image)

Rows are written in the smooth interleaved order of corpus_spec.interleave(). The last 10% of each
pool's rows are the frozen held-out split: s03 runs them through the teacher for top-32 logits
and routing but never lets them touch a saliency accumulator.

Packing. Text documents are packed back to back in 8k pools (a document that overflows continues
on the next row of the same pool). Vision samples are never split: an image span must sit inside
one prefill, so a sample that does not fit pads the row out (valid=0) and starts the next.

Sources. `hf_id` is a HF dataset, `local:<path>` a jsonl under $V41_WORK (s02b self-generated
traces), or `git:<repo>@<ref>` an upstream git ref read with `git show` (whole-repo long rows).

Modes:
  --dry-run   list sources, usability, declared licences and the row/token plan; no network
  --probe     per source: live card licence + a small streamed read to measure yield; no corpus
  (default)   build; resumable per pool through ops/logs/s02_ledger.json

RUN (CPU and network only; detach it):
    python3 ops/s02_calib_corpus.py --dry-run
    setsid nohup python3 ops/s02_calib_corpus.py --probe > ops/logs/s02_probe.log 2>&1 &
    setsid nohup python3 ops/s02_calib_corpus.py > ops/logs/s02.log 2>&1 &
Needs: datasets, tokenizers, pillow, huggingface_hub; torch on CPU only because the vision pool
calls the reference inference/image_processor.plan_image_grid.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops import corpus_spec as spec  # noqa: E402
from ops.lib import env  # noqa: E402

# Domain-REAP track (docs/domain/PLAN.md, ops/dom/README.md): V41_DOM=1 swaps in the 6-bucket domain spec
# and writes under $V41_WORK/dom; unset, the spec and every path are exactly the original ones.
DOM = os.environ.get("V41_DOM") == "1"
if DOM:
    from ops.dom import corpus_spec_dom as spec  # noqa: E402,F811

STAGE = "s02"
# V41_DOM_DRAW=2: the dose-response draw (PLAN): same buckets at 2x rows, every source resuming after the rows the
# first draw read, into $V41_WORK/dom/corpus_draw2 (build the first draw first).
DRAW = os.environ.get("V41_DOM_DRAW", "1") if DOM else "1"
DRAW_SCALE = 2 if DRAW == "2" else 1
OUT = (env.WORK / "dom" / ("corpus_draw2" if DRAW == "2" else "corpus")) if DOM else env.WORK / "corpus"
LEDGER = env.LOGS / (("dom_s02_draw2_ledger.json" if DRAW == "2" else "dom_s02_ledger.json") if DOM else "s02_ledger.json")
PROBE = env.LOGS / ("dom_s02_probe.json" if DOM else "s02_probe.json")
TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = -1, 0, 1, 2, 3
CFG = json.load(open(env.INFERENCE / "config.json"))


def pool_sources(bucket: str) -> list[dict]:
    if bucket == "vision":
        return spec.VISION_SOURCES
    if bucket == "longctx":
        return spec.LONG_SOURCES
    return spec.SOURCES[bucket]


def spec_hash() -> str:
    blob = json.dumps({"target": spec.TOKEN_TARGET, "seq": spec.SEQ_LEN, "total": spec.TOTAL_TOKENS,
                       "long": [spec.LONG_MIN, spec.LONG_MAX],
                       "sources": {b: [(s["hf_id"], s["config"], s["split"], s["weight"]) for s in pool_sources(b)]
                                   for b in spec.BUCKETS}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def first_draw_reads() -> dict | None:
    """Second draw only: rows each source read in the first draw (sources.json keys are bucket/name)."""
    if DRAW != "2":
        return None
    p = env.WORK / "dom" / "corpus" / "sources.json"
    if not (env.WORK / "dom" / "corpus" / "manifest.json").exists():
        raise SystemExit("draw 2 needs the finished first draw ($V41_WORK/dom/corpus/manifest.json)")
    return {k: v.get("read", 0) + v.get("skipped_first_draw", 0) for k, v in json.load(open(p)).items()}


def pool_hash(bucket: str, a) -> str:
    """Domain track: identity of one pool's source list (+ generator policy), so a reconciled pool is rebuilt alone."""
    blob = json.dumps({"policy": a.generator_policy, "seq": spec.SEQ_LEN, "draw": DRAW,
                       "plan": [(sa, n) for sa, _, n in spec.subarea_plan(bucket, spec.rows_per_bucket(a.total_tokens)[bucket])],
                       "rows": spec.rows_per_bucket(a.total_tokens)[bucket],
                       "src": [(s["hf_id"], s["config"], s["split"], s["weight"], s.get("include"), s.get("exts"),
                                s.get("exclude"), s.get("max_files"), s.get("data_files")) for s in pool_sources(bucket)],
                       # local jsonl content (self-traces, transcripts, threads, owner repos): a pool goes stale when one
                       # lands or changes, so `--pools <all>` after a trace window rebuilds exactly the affected pools
                       "local": [(s["hf_id"], (env.WORK / s["hf_id"][6:]).stat().st_size if (env.WORK / s["hf_id"][6:]).exists()
                                  else None) for s in pool_sources(bucket) if s["hf_id"].startswith("local:")],
                       # live-licence handling that is not part of the source tuple (2026-11-01 GPL/GFDL NC fix)
                       "lic": [(s["hf_id"], s.get("licence_ignore"), s.get("live_licence_override")) for s in pool_sources(bucket)
                               if s.get("licence_ignore") or s.get("live_licence_override")] + ["gpl-nc-fix-1"]},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def dry_run(allow_unverified: bool, policy: str, total: int) -> None:
    rows = spec.rows_per_bucket(total)
    print(f"spec {spec_hash()}: ~{total:,} tokens; generator policy {policy}")
    print("pool plan (8k rows; longctx = token budget in 8k units, filled with whole 32k-128k docs):")
    for b in spec.BUCKETS:
        print(f"  {b:12s} share {spec.TOKEN_TARGET[b]:.2f}  {rows[b]:4d} x 8k = {rows[b] * spec.SEQ_LEN:>10,} tokens")
    for b in spec.BUCKETS:
        for s in pool_sources(b):
            ok, why = spec.usable(s, allow_unverified, policy)
            print(f"  {b:12s} {'USE ' if ok else 'SKIP'} w={s['weight']:.2f} {s['hf_id']}"
                  f"{'/' + s['config'] if s['config'] else ''}{':' + s['split'] if s['split'] else ''}"
                  f"  licence={s['licence']}  generator={s['generator']}  status={s['status']}"
                  + ("" if ok else f"  ({why})"))
    print("excluded:", json.dumps(spec.EXCLUDED, indent=1))


def live_licence(src: dict) -> str | None:
    if "live_licence_override" in src:          # domain spec: a dual-licence repo read only in its permissive part
        return src["live_licence_override"]
    if not src["hf_id"].startswith(("local:", "git:")):
        try:
            from huggingface_hub import HfApi
            card = HfApi().dataset_info(src["hf_id"]).card_data or {}
            lic = card.get("license")
            return ",".join(lic) if isinstance(lic, list) else lic
        except Exception as e:
            return f"error: {type(e).__name__}"
    if src["hf_id"].startswith("git:"):
        repo, ref = git_ref(src)
        if hasattr(spec, "git_licence"):          # domain spec: classify every LICENSE/COPYING file + record the sha
            return spec.git_licence(repo, ref, src.get("licence_ignore", ()))
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
            r = subprocess.run(["git", "-C", repo, "show", f"{ref}:{name}"], capture_output=True, text=True)
            if r.returncode == 0:
                head = r.stdout[:400].lower()
                return "mit" if "mit license" in head or "permission is hereby granted" in head else \
                    "bsd-3-clause" if "redistribution and use" in head else f"see {name}"
        return None
    return src["licence"]


def git_ref(src: dict) -> tuple[str, str]:
    repo, ref = src["hf_id"][4:].split("@")
    return os.path.expanduser(repo), ref


def stream(src: dict):
    if "stream" in src:                          # domain spec: per-file git rows, HF data_files subsets
        return src["stream"](src)
    hid = src["hf_id"]
    if hid.startswith("local:"):
        p = env.WORK / hid[6:]
        if not p.exists():
            return iter(())
        return (json.loads(l) for l in open(p))
    if hid.startswith("git:"):
        return git_repo_docs(src)
    from datasets import load_dataset
    kw = {"split": src["split"], "streaming": True}
    return iter(load_dataset(hid, src["config"], **kw) if src["config"] else load_dataset(hid, **kw))


def git_repo_docs(src: dict):
    """Yield whole-repo text slices of the upstream ref: files grouped by top-level directory,
    each file prefixed with its path, so one row is one coherent subtree."""
    repo, ref = git_ref(src)
    files = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", ref],
                           capture_output=True, text=True, check=True).stdout.split()
    files = [f for f in files if f.endswith(spec.REPO_EXTS)]
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(f.split("/")[0] if "/" in f else ".", []).append(f)
    for top, fs in sorted(groups.items()):
        parts = []
        for f in sorted(fs):
            r = subprocess.run(["git", "-C", repo, "show", f"{ref}:{f}"], capture_output=True)
            if r.returncode == 0 and b"\0" not in r.stdout[:4096]:
                parts.append(f"// ===== {f} =====\n" + r.stdout.decode("utf-8", "replace"))
        if parts:
            yield {"text": "\n".join(parts), "_doc": f"{src['hf_id']}:{top}"}


class Renderer:
    """Document -> token ids through the checkpoint's own encoder and tokenizer."""

    def __init__(self):
        sys.path.insert(0, str(env.ENCODING))
        import encoding as E
        from ops.lib.tok import Tok
        self.E, self.tok = E, Tok(env.SRC)
        self.image_token_id = CFG["image_token_id"]
        self.pad_id = self.tok.convert_tokens_to_ids(E.eos_token)   # pad slots carry valid=0

    def text_ids(self, doc, thinking: bool) -> list[int] | None:
        if doc is None:
            return None
        if isinstance(doc, str):
            return self.tok.encode(self.E.bos_token + doc)
        prompt = self.E.encode_messages(doc, thinking_mode="thinking" if thinking else "chat")
        return self.tok.encode(prompt)

    def vision_ids(self, messages) -> tuple[list[int], list[int], list[dict]]:
        """Expand each image placeholder into its span exactly as image_processor.prepare_vl_inputs
        lays it out, using the reference plan_image_grid (s03 re-derives it with load_image and
        asserts equality)."""
        from PIL import Image
        prompt, mm = self.E.encode_messages(messages, thinking_mode="chat", return_multi_modal_data=True)
        ids = self.tok.encode(prompt)
        imgs = iter(mm["images"])
        out, types, spans = [], [], []
        for t in ids:
            if t != self.image_token_id:
                out.append(t); types.append(TEXT)
                continue
            rec = next(imgs)
            with Image.open(rec["url"]) as im:
                w, h = im.size
            n_h, n_w = plan_grid(w, h)
            ty = [IMAGE_START] + ([IMAGE] * n_w + [IMAGE_NEW_LINE]) * n_h + [IMAGE_END]
            spans.append({"start": len(out), "path": rec["url"], "n_llm_h": n_h, "n_llm_w": n_w, "n": len(ty)})
            out += [self.image_token_id] * len(ty); types += ty
        return out, types, spans


class _VisionArgs:
    vision_patch_size = CFG["vision_patch_size"]
    vision_downsample_ratio = CFG["vision_downsample_ratio"]
    vision_max_n_token = CFG["vision_max_n_token"]
    vision_min_pixels = CFG["vision_min_pixels"]
    vision_max_wh_ratio = CFG["vision_max_wh_ratio"]


def plan_grid(w, h):
    if str(env.INFERENCE) not in sys.path:
        sys.path.insert(0, str(env.INFERENCE))
    from image_processor import plan_image_grid  # the reference function, not a copy
    n_h, n_w, _, _ = plan_image_grid(w, h, _VisionArgs)
    return n_h, n_w


def vision_messages(row: dict, img_dir: Path, key: str) -> list[dict] | None:
    """the_cauldron / Docmatix: images + texts[{user, assistant}];
    LLaVA-OneVision: image + conversations[{from, value}] with an <image> marker."""
    from PIL import Image
    imgs = row.get("images") or ([row["image"]] if row.get("image") is not None else [])
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return None
    paths = []
    for k, im in enumerate(imgs[:4]):
        if isinstance(im, dict):          # undecoded Image feature: {"bytes": ..., "path": ...}
            im = Image.open(io.BytesIO(im["bytes"]))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        p = img_dir / f"{key}_{k}.png"
        p.write_bytes(buf.getvalue())
        paths.append(str(p))
    blocks = [{"type": "image", "url": p} for p in paths]
    if isinstance(row.get("texts"), list) and row["texts"]:
        msgs = []
        for i, t in enumerate(row["texts"][:6]):
            u, a = t.get("user"), t.get("assistant")
            if not u or not a:
                continue
            content = (blocks if not msgs else []) + [{"type": "text", "text": u}]
            msgs += [{"role": "user", "content": content}, {"role": "assistant", "content": a}]
        return msgs or None
    conv = spec._messages(row, "conversations")
    if conv and conv[0]["role"] == "user":
        conv[0] = {"role": "user", "content": blocks + [{"type": "text", "text": conv[0]["content"].replace("<image>", "").strip()}]}
        return conv
    return None


def url_filter(src: dict):
    st = src["status"]
    if not st.startswith("filtered:"):
        return None
    pat = spec.SCIENCE_URL if "SCIENCE" in st else spec.FINANCE_URL if "FINANCE" in st else spec.KNOWLEDGE_URL
    return re.compile(pat)


def build_pool(bucket: str, budget: int, R: Renderer, a, rep: dict, srcs=None, min_fill=0.9, skip=None,
               key_prefix="", prompts=None) -> list[dict]:
    """Fill one pool to `budget` tokens. Returns rows: {tokens, types, valid, docs, images}.
    Domain track only: `srcs` (one sub-area's sources), `min_fill` 0 (the caller checks the whole pool), `skip`
    (rows of each source already read by the first draw), `key_prefix` (bucket/ in sources.json keys) and `prompts`
    (a list that collects each kept chat document's first user turn with the pool row it starts in)."""
    S = spec.SEQ_LEN
    long_pool = bucket == "longctx"
    if srcs is None:
        srcs = [s for s in pool_sources(bucket) if spec.usable(s, a.allow_unverified, a.generator_policy)[0]]
    wsum = sum(s["weight"] for s in srcs)
    rows: list[dict] = []
    cur = None
    img_dir = OUT / "images"; img_dir.mkdir(parents=True, exist_ok=True)

    def new_row():
        r = {"tokens": [], "types": [], "valid": [], "docs": [], "images": []}
        rows.append(r)
        return r

    def filled():
        return sum(sum(r["valid"]) for r in rows)

    deficit = 0
    for si, s in enumerate(srcs):
        name = s["hf_id"] + (f"/{s['config']}" if s["config"] else "") + (f":{s['split']}" if s["split"] else "")
        # a short source's shortfall is spread over the sources after it by weight (2026-10-19: rolling it all
        # into the next one gave SWE-rebench 524k of the ~600k agentic pool when selfgen was empty)
        w_left = sum(x["weight"] for x in srcs[si:])
        quota = int(budget * s["weight"] / wsum + deficit * s["weight"] / w_left)
        deficit -= deficit * s["weight"] / w_left
        live = live_licence(s)
        rec = rep.setdefault(key_prefix + name, {"bucket": bucket, "licence_declared": s["licence"], "licence_live": live,
                                    "licence_live_ok": spec.licence_ok(live), "generator": s["generator"],
                                    "status": s["status"], "read": 0, "kept": 0, "tokens": 0})
        if spec.licence_ok(live) is False:
            rec["skipped"] = f"live licence {live} is NC/share-alike"
            deficit += quota
            env.log(f"{bucket}: SKIP {name}: live licence {live}", STAGE, "WARN")
            continue
        flt = url_filter(s)
        got, t0 = 0, time.time()
        try:
            it = stream(s)
            if skip and skip.get(key_prefix + name):            # second draw: start after the first draw's rows
                it = itertools.islice(it, skip[key_prefix + name], None)
                rec["skipped_first_draw"] = skip[key_prefix + name]
            for row in it:
                if got >= quota:
                    break
                rec["read"] += 1
                if flt is not None and not flt.search(str(row.get("url", ""))):
                    continue
                doc = None
                try:
                    if bucket == "vision":
                        msgs = vision_messages(row, img_dir, f"{si}_{rec['read']}")
                        ids, ty, spans = R.vision_ids(msgs) if msgs else (None, None, [])
                    elif s["fn"] is None or s["hf_id"].startswith("git:"):
                        ids, ty, spans = R.text_ids(row.get("text"), False), None, []
                    else:
                        doc = s["fn"](row)
                        th = s["thinking"] if not isinstance(row, dict) or "thinking" not in row else bool(row["thinking"])
                        ids, ty, spans = R.text_ids(doc, th), None, []
                except Exception:
                    ids = None
                if not ids:
                    if rec["read"] > 2000 and rec["kept"] == 0:
                        break  # dead source: 2,000 rows, zero usable (GLM: media sources die silently)
                    continue
                if long_pool:
                    if len(ids) < spec.LONG_MIN:
                        continue
                    ids = ids[: max(spec.LONG_MIN, min(spec.LONG_MAX, quota - got))]   # don't overshoot the quota
                    rows.append({"tokens": ids, "types": [TEXT] * len(ids), "valid": [1] * len(ids),
                                 "docs": [{"source": name, "tokens": len(ids), "doc": row.get("_doc")}], "images": []})
                    got += len(ids)
                elif bucket == "vision":
                    if len(ids) > S:
                        continue                                  # an image span would be cut
                    if cur is None or len(cur["tokens"]) + len(ids) > S:
                        if cur is not None:                       # pad this row out (valid=0)
                            pad = S - len(cur["tokens"])
                            cur["tokens"] += [R.pad_id] * pad; cur["types"] += [TEXT] * pad; cur["valid"] += [0] * pad
                        cur = new_row()
                    cur["images"] += [{**sp, "start": sp["start"] + len(cur["tokens"])} for sp in spans]
                    cur["tokens"] += ids; cur["types"] += ty; cur["valid"] += [1] * len(ids)
                    cur["docs"].append({"source": name, "tokens": len(ids)})
                    got += len(ids)
                else:
                    # one document may not dominate a source: cap at two rows and at the quota left
                    # (2026-10-19: whole SWE-rebench patches took 524k against a 224k quota)
                    ids = ids[: min(2 * S, quota - got)]
                    tool = DOM and isinstance(doc, list) and any(m.get("tool_calls") for m in doc)
                    if prompts is not None and isinstance(doc, list) and not s.get("calibration_only"):
                        u = next((m.get("content") for m in doc if m.get("role") == "user"), None)
                        if isinstance(u, str) and u.strip():
                            start = len(rows) - 1 if cur is not None and len(cur["tokens"]) < S else len(rows)
                            prompts.append({"source": name, "prompt": u.strip(), "row": start, "tools": _doc_tools(doc) if tool else None})
                    pos = 0
                    while pos < len(ids):
                        if cur is None or len(cur["tokens"]) == S:
                            cur = new_row()
                        k = min(S - len(cur["tokens"]), len(ids) - pos)
                        cur["tokens"] += ids[pos:pos + k]; cur["types"] += [TEXT] * k; cur["valid"] += [1] * k
                        cur["docs"].append({"source": name, "tokens": k, **({"tool": True} if tool else {}),
                                            **({"sa": True, "calibration_only": True} if DOM and s.get("calibration_only") else {})})
                        pos += k; got += k
                rec["kept"] += 1
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["tokens"] += got
        rec["seconds"] = round(time.time() - t0, 1)
        rec["yield"] = rec["kept"] / max(1, rec["read"])
        deficit += max(0, quota - got)
        env.log(f"{bucket}: {name} +{got:,} tok (quota {quota:,}), kept {rec['kept']}/{rec['read']}"
                f"{' ERROR ' + rec['error'] if 'error' in rec else ''}", STAGE)
    if not long_pool:
        # 8k pools: whole rows only; drop a trailing partial text row, pad a trailing vision row
        if rows and len(rows[-1]["tokens"]) < S:
            if bucket == "vision":
                r = rows[-1]; pad = S - len(r["tokens"])
                r["tokens"] += [R.pad_id] * pad; r["types"] += [TEXT] * pad; r["valid"] += [0] * pad
            else:
                rows.pop()
        rows = rows[: budget // S]
    if filled() < min_fill * budget:
        raise RuntimeError(f"{bucket}: only {filled():,}/{budget:,} tokens; sources too thin. Fix the spec rather "
                           "than letting another pool absorb the share")
    return rows


def build_pool_dom(bucket: str, n_rows: int, R: Renderer, a, rep: dict, skip=None) -> list[dict]:
    """Domain track: stratified pool. Rows are packed per sub-area (spec.subarea_plan: selftraces share, >= floor per
    topical sub-area, rest by value); a sub-area's shortfall (e.g. absent selftraces) flows to the later sub-areas by
    planned rows; each sub-area holds out its own last rows; rows are interleaved smoothly across sub-areas."""
    S = spec.SEQ_LEN
    # a git source whose clone is absent would silently hand its quota on: refuse unless prep.py recorded it FAILED
    pm = env.WORK / "dom" / "prep_manifest.json"
    failed = {k for k, v in (json.load(open(pm)).get("clone", {}) if pm.exists() else {}).items()
              if isinstance(v, dict) and "error" in v}
    absent = [x["repo"] for x in spec.SOURCES[bucket] if x["hf_id"].startswith("git:")
              and not any((Path(spec.git_paths(x)[0]) / h).exists() for h in ("HEAD", ".git/HEAD", ".git"))  # bare or working tree
              and x["repo"] not in failed]
    if absent:
        raise RuntimeError(f"{bucket}: git sources not cloned yet (run ops/dom/prep.py clone): {absent}")
    plan = spec.subarea_plan(bucket, n_rows, lambda s: spec.usable(s, a.allow_unverified, a.generator_policy)[0])
    by_sa: dict[str, list[dict]] = {}
    cand: list[dict] = []
    deficit = carry = 0.0
    for i, (sa, srcs, n) in enumerate(plan):
        left = sum(m for _, _, m in plan[i:])
        extra = deficit * n / left
        deficit -= extra
        # carry the rounding remainder: rounding each share alone gave 0.8 -> 1 row in 10 sub-areas, so pools came
        # out at 52 rows instead of 50 (two buckets, 2026-11-01)
        target = n + extra + carry
        want = int(round(target))
        carry = target - want
        pr: list[dict] = []
        # +1 row of budget: build_pool drops a trailing partial row, which cost every sub-area one row
        rows = build_pool(bucket, (want + 1) * S, R, a, rep, srcs=srcs, min_fill=0.0, skip=skip, key_prefix=f"{bucket}/",
                          prompts=pr)[:want] if want else []
        for r in rows:
            r["subarea"] = sa
        k = max(1, round(len(rows) * spec.HELDOUT_ROWS_FRACTION)) if rows else 0
        for j, r in enumerate(rows):
            r["heldout"] = j >= len(rows) - k
        cand += [{**p, "subarea": sa, "heldout": rows[p["row"]]["heldout"]} for p in pr if p["row"] < len(rows)]
        by_sa[sa] = rows
        deficit += max(0, want - len(rows))
        env.log(f"{bucket}/{sa}: {len(rows)}/{want} rows (planned {n}), {k} held out", STAGE)
    order = spec.interleave_keys({sa: len(r) for sa, r in by_sa.items() if r})
    nxt = dict.fromkeys(by_sa, 0)
    pool = []
    for sa in order:
        pool.append(by_sa[sa][nxt[sa]]); nxt[sa] += 1
    if len(pool) < 0.9 * n_rows:
        raise RuntimeError(f"{bucket}: only {len(pool)}/{n_rows} rows; sources too thin. Fix the spec")
    # every bucket keeps >= 10% of its tokens as tool-calling DSML documents (coordinator 2026-11-01)
    tool_tok = sum(d["tokens"] for r in pool for d in r["docs"] if d.get("tool"))
    all_tok = sum(sum(r["valid"]) for r in pool)
    env.log(f"{bucket}: tool-calling DSML tokens {tool_tok:,}/{all_tok:,} = {tool_tok / max(1, all_tok):.1%}", STAGE)
    if tool_tok < 0.10 * all_tok:
        raise RuntimeError(f"{bucket}: tool-calling tokens {tool_tok / max(1, all_tok):.1%} < 10%; fix the tool-use sources")
    if DRAW == "1":
        write_prompts(bucket, cand, plan, R, rep)
    return pool


PROMPTS_PER_BUCKET = 300
PROMPT_MAX_TOKENS = 2048
# generic (not bucket-specific) tool-calling sets: calibration rows yes, self-trace task prompts no
GENERIC = ("nvidia/Nemotron-SFT-Agentic-v2", "nvidia/Nemotron-Agentic-v1")


def eval_ids(bucket: str) -> set[str]:
    """HF dataset ids / GitHub repos named in table C (evaluation) of docs/domain/sources/<bucket>.md."""
    src = env.REPO / "docs" / "domain" / "sources"
    md = src / f"{bucket}.md"
    if not md.exists():
        return set()
    txt = md.read_text()
    m = re.search(r"^## C\b.*?(?=^## |\Z)", txt, re.M | re.S)
    sec = m.group(0) if m else ""
    return set(re.findall(r"huggingface\.co/datasets/([\w.-]+/[\w.-]+)", sec)) | \
        set(re.findall(r"github\.com/([\w.-]+/[\w.-]+)", sec))


def _doc_tools(doc: list) -> list | None:
    """The tool schema a tool-calling document was rendered with (system-turn `tools`), so its prompt stays answerable."""
    for m in doc:
        if m.get("tools"):
            return m["tools"]
    return None


# A prompt that presumes a live session (Claude Code tools, a working tree, a web tool) cannot be answered without tools.
_LIVE = re.compile(r"\b(WebSearch|WebFetch|TodoWrite|use the \w+ tool|in this repo(sitory)?|working directory|"
                   r"run (the )?(tests|command)s?|/home/\w+|~/[\w.-]+/)", re.I)
TOOL_PROMPT_SHARE = 0.2    # coordinator: mostly self-contained prompts, a minority carrying a tool schema


def extra_prompt_candidates(bucket: str, plan, rep: dict, need: int, chat: set[str] | None = None) -> list[dict]:
    """More first-user-turn candidates when the pool's own chat documents give too few: each chat/instruction source
    of the bucket is read past the rows the pool consumed (sources.json `read`), so these documents are NOT in the
    corpus (`pool_row` None, `heldout` False). At most 3,000 rows are scanned per source. Only sources that gave chat
    documents in the pool (`chat`) are read: re-streaming a text source past a large `read` offset (papers behind a
    keyword filter) costs tens of minutes for nothing."""
    srcs = [(sa, x) for sa, xs, _ in plan if sa != "selftraces" for x in xs
            if not x["hf_id"].startswith("git:") and x["fn"] is not None and not any(g in x["hf_id"] for g in GENERIC)
            and not x.get("calibration_only")
            and (chat is None or x["hf_id"] + (f"/{x['config']}" if x["config"] else "") + (f":{x['split']}" if x["split"] else "") in chat)]
    per = -(-need * 2 // max(1, len(srcs)))
    out = []
    for sa, x in srcs:
        name = x["hf_id"] + (f"/{x['config']}" if x["config"] else "") + (f":{x['split']}" if x["split"] else "")
        start = rep.get(f"{bucket}/{name}", {}).get("read", 0)
        got = 0
        try:
            for row in itertools.islice(stream(x), start, start + 3000):
                try:
                    doc = x["fn"](row)
                except Exception:
                    continue
                if not isinstance(doc, list):
                    continue
                u = next((m.get("content") for m in doc if m.get("role") == "user"), None)
                if isinstance(u, str) and u.strip():
                    tl = _doc_tools(doc) if any(m.get("tool_calls") for m in doc) else None
                    out.append({"source": name, "prompt": u.strip(), "row": None, "subarea": sa, "heldout": False, "tools": tl})
                    got += 1
                    if got >= per:
                        break
        except Exception as e:
            env.log(f"{bucket}: extra prompts from {name}: {type(e).__name__}: {str(e)[:200]}", STAGE, "WARN")
    return out


def write_prompts(bucket: str, cand: list[dict], plan, R: Renderer, rep: dict | None = None) -> None:
    """Self-trace prompt pool (coordinator): ~PROMPTS_PER_BUCKET first user turns from the pool's chat documents,
    stratified by sub-area (by planned rows), round-robin over sources inside a sub-area, deduplicated, at most
    PROMPT_MAX_TOKENS tokens, never from a table-C eval source or from self-traces; `heldout` marks prompts whose
    document sits in a held-out corpus row (do not reuse those for training traces)."""
    ev = eval_ids(bucket)
    seen, ok = set(), []
    for p in cand:
        if "dom/selftraces/" in p["source"] or any(e in p["source"] for e in ev) or any(g in p["source"] for g in GENERIC):
            continue
        key = hashlib.sha1(re.sub(r"\s+", " ", p["prompt"].lower()).encode()).hexdigest()
        if key in seen or len(p["prompt"]) < 40:
            continue
        seen.add(key)
        if len(R.tok.encode(p["prompt"])) > PROMPT_MAX_TOKENS:
            continue
        ok.append(p)
    if len(ok) < PROMPTS_PER_BUCKET and rep is not None:      # top up from documents outside the corpus
        for p in extra_prompt_candidates(bucket, plan, rep, PROMPTS_PER_BUCKET - len(ok), {p["source"] for p in cand}):
            if any(e in p["source"] for e in ev) or len(p["prompt"]) < 40:
                continue
            key = hashlib.sha1(re.sub(r"\s+", " ", p["prompt"].lower()).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            if len(R.tok.encode(p["prompt"])) <= PROMPT_MAX_TOKENS:
                ok.append(p)
    planned = {sa: n for sa, _, n in plan if sa != "selftraces"}
    # self-contained prompts (no tool schema, no live-session assumptions) form the majority; prompts of tool-calling
    # documents keep their tool schema (`tools`) and are capped at TOOL_PROMPT_SHARE
    with_tools = [p for p in ok if p.get("tools")]
    plain = [p for p in ok if not p.get("tools") and not _LIVE.search(p["prompt"])]
    n_tool = min(len(with_tools), round(TOOL_PROMPT_SHARE * PROMPTS_PER_BUCKET))
    out = _stratify(plain, planned, PROMPTS_PER_BUCKET - n_tool)
    out += _stratify(with_tools, planned, n_tool)    # strict cap: a short plain pool is not back-filled with tool prompts
    d = env.WORK / "dom" / "prompts"; d.mkdir(parents=True, exist_ok=True)
    dropped_live = sum(1 for p in ok if not p.get("tools") and _LIVE.search(p["prompt"]))
    with open(d / f"{bucket}.jsonl.tmp", "w") as fh:
        for p in out:
            fh.write(json.dumps({"bucket": bucket, "subarea": p["subarea"], "source": p["source"], "prompt": p["prompt"],
                                 "heldout": p["heldout"], "pool_row": p["row"], "in_corpus": p["row"] is not None,
                                 "needs_tools": bool(p.get("tools")), "tools": p.get("tools")}) + "\n")
    os.replace(d / f"{bucket}.jsonl.tmp", d / f"{bucket}.jsonl")
    env.log(f"{bucket}: {len(out)} self-trace prompts ({sum(p['heldout'] for p in out)} from held-out rows; "
            f"{sum(bool(p.get('tools')) for p in out)} with a tool schema; {len(cand)} candidates, {len(ok)} after "
            f"dedup/length/eval filters, {dropped_live} live-session prompts dropped) -> {d / (bucket + '.jsonl')}", STAGE)


def _stratify(cands: list[dict], planned: dict, n: int) -> list[dict]:
    """n prompts spread over sub-areas by planned rows (thin sub-areas first, unused quota flows on), round-robin over
    the sources inside a sub-area."""
    if n <= 0 or not cands:
        return []
    by_sa: dict[str, dict[str, list[dict]]] = {}
    for p in cands:
        by_sa.setdefault(p["subarea"], {}).setdefault(p["source"], []).append(p)
    tot = sum(planned.get(sa, 0) for sa in by_sa) or 1
    quota = {sa: max(1, round(n * planned.get(sa, 0) / tot)) for sa in by_sa}
    out, spare = [], 0
    for sa in sorted(by_sa, key=lambda x: sum(map(len, by_sa[x].values()))):
        q = quota[sa] + spare
        srcs = [list(v) for v in by_sa[sa].values()]
        got = []
        while len(got) < q and any(srcs):
            for v in srcs:
                if v and len(got) < q:
                    got.append(v.pop(0))
        spare = q - len(got)
        out += got
    return out[:n]


def probe(a) -> None:
    out = {}
    for b in spec.BUCKETS:
        for s in pool_sources(b):
            name = s["hf_id"] + (f"/{s['config']}" if s["config"] else "") + (f":{s['split']}" if s["split"] else "")
            rec = {"bucket": b, "usable": spec.usable(s, a.allow_unverified, a.generator_policy),
                   "licence_declared": s["licence"], "licence_live": live_licence(s), "generator": s["generator"],
                   "status": s["status"]}
            try:
                ok = seen = 0
                row = {}
                for row in stream(s):
                    seen += 1
                    if b == "vision":
                        ok += bool(row.get("images") or row.get("image") is not None)
                    elif s["fn"] is not None:
                        ok += bool(s["fn"](row))
                    else:
                        ok += bool(row.get("text"))
                    if seen >= a.probe_rows:
                        break
                rec.update(seen=seen, yield_=ok / max(1, seen), columns=sorted(row.keys()) if seen else [])
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            out[name] = rec
            env.log(f"probe {name}: {json.dumps({k: v for k, v in rec.items() if k != 'columns'}, default=str)}", STAGE)
    env.atomic_json(PROBE, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-rows", type=int, default=64)
    ap.add_argument("--allow-unverified", action="store_true",
                    help="include sources whose generator is 'unverified' (only after checking their cards)")
    ap.add_argument("--generator-policy", choices=spec.GENERATOR_POLICIES, default=getattr(spec, "DEFAULT_POLICY", "no-claude"))
    ap.add_argument("--total-tokens", type=int, default=spec.TOTAL_TOKENS)
    if DOM:
        ap.add_argument("--pools", nargs="*", default=None,
                        help="domain track: build only these pools and skip corpus assembly (re-run without it to assemble)")
    a = ap.parse_args()
    if a.dry_run:
        return dry_run(a.allow_unverified, a.generator_policy, a.total_tokens)
    if a.probe:
        return probe(a)

    OUT.mkdir(parents=True, exist_ok=True)
    led = env.Ledger(LEDGER)
    h = spec_hash()
    if DOM:   # domain track: each pool carries its own source hash and is rebuilt alone when its source list changes
        if led.get("spec", {}).get("generator_policy") not in (None, a.generator_policy):
            raise SystemExit(f"ledger was built with policy {led.get('spec')['generator_policy']}; move {OUT} aside first")
    elif led.get("spec", {}).get("hash") not in (None, h):
        raise SystemExit(f"ledger was built with spec {led.get('spec')['hash']}, current {h}; move {OUT} aside first")
    led.mark("spec", hash=h, generator_policy=a.generator_policy, allow_unverified=a.allow_unverified)
    plan = spec.rows_per_bucket(a.total_tokens)
    R = Renderer()
    rep = json.load(open(OUT / "sources.json")) if (OUT / "sources.json").exists() else {}
    for b in spec.BUCKETS:
        part = OUT / f"pool_{b}.json"
        if DOM:
            if a.pools is not None and b not in a.pools:
                continue
            ph = pool_hash(b, a)
            if led.done(f"pool:{b}") and part.exists() and led.get(f"pool:{b}", {}).get("hash") == ph:
                continue
            rep = {k: v for k, v in rep.items() if v.get("bucket") != b}     # a rebuilt pool re-counts its sources
        elif led.done(f"pool:{b}") and part.exists():
            continue
        if DOM:
            rows = build_pool_dom(b, plan[b] * DRAW_SCALE, R, a, rep, skip=first_draw_reads())
        else:
            rows = build_pool(b, plan[b] * spec.SEQ_LEN, R, a, rep)
        env.atomic_json(part, rows)
        env.atomic_json(OUT / "sources.json", rep)
        led.finish(f"pool:{b}", rows=len(rows), tokens=sum(sum(r["valid"]) for r in rows), **({"hash": ph} if DOM else {}))
    if DOM and a.pools is not None:
        env.log(f"pools {a.pools} built; corpus assembly skipped (--pools)", STAGE)
        return
    if DOM:
        stale = [b for b in spec.BUCKETS if led.get(f"pool:{b}", {}).get("hash") != pool_hash(b, a)]
        assert not stale, f"pools out of date with the spec: {stale}"

    pools = {b: json.load(open(OUT / f"pool_{b}.json")) for b in spec.BUCKETS}
    order = spec.interleave({b: len(pools[b]) for b in spec.BUCKETS})
    T = sum(len(r["tokens"]) for b in pools for r in pools[b])
    mm = {k: np.lib.format.open_memmap(OUT / f"{k}.npy", mode="w+", dtype=d, shape=(T,))
          for k, d in (("tokens", np.uint32), ("types", np.int8), ("valid", np.uint8))}
    nxt = {b: 0 for b in spec.BUCKETS}
    off = 0
    tok_by = {b: 0 for b in spec.BUCKETS}
    with open(OUT / "rows.jsonl.tmp", "w") as fh:
        for row, b in enumerate(order):
            i = nxt[b]; nxt[b] += 1
            r = pools[b][i]
            n = len(r["tokens"])
            mm["tokens"][off:off + n] = r["tokens"]; mm["types"][off:off + n] = r["types"]; mm["valid"][off:off + n] = r["valid"]
            n_held = max(1, round(len(pools[b]) * spec.HELDOUT_ROWS_FRACTION))
            rec = {"row": row, "bucket": b, "bucket_id": spec.BUCKETS.index(b), "offset": off, "length": n,
                   "heldout": i >= len(pools[b]) - n_held, "n_valid": int(sum(r["valid"])),
                   "docs": r["docs"], "images": r["images"]}
            if DOM:   # per-sub-area held-out flag (set by build_pool_dom) and the sub-area stat-bucket id for s03
                rec.update(heldout=r["heldout"], subarea=r["subarea"], sub_id=spec.SUBAREAS.index(f"{b}/{r['subarea']}"))
            fh.write(json.dumps(rec) + "\n")
            tok_by[b] += int(sum(r["valid"]))
            off += n
    os.replace(OUT / "rows.jsonl.tmp", OUT / "rows.jsonl")
    for m in mm.values():
        m.flush()
    total = sum(tok_by.values())
    lens = [len(r["tokens"]) for r in pools.get("longctx", [])]
    manifest = {"spec_hash": h, "rows": len(order), "tokens_total": T, "tokens_valid": total, "order": order,
                "achieved_share": {b: tok_by[b] / total for b in spec.BUCKETS}, "target_share": spec.TOKEN_TARGET,
                "longctx_lengths": lens, "max_row_len": max(len(r["tokens"]) for b in pools for r in pools[b]),
                "generator_policy": a.generator_policy, "revision": env.REVISION, "ts": env.now()}
    if DOM:
        manifest["buckets"] = spec.BUCKETS          # s03 asserts this order before banking anything
        manifest["subareas"] = spec.SUBAREAS
        manifest["draw"] = DRAW
        manifest["achieved_subarea_rows"] = {sa: sum(1 for b in pools for r in pools[b] if f"{b}/{r['subarea']}" == sa)
                                             for sa in spec.SUBAREAS}
        manifest["owner_repo_holdout"] = spec.OWNER_HOLDOUT     # owner hold-out: nothing on/after the cutoff is used
        manifest["tool_token_share"] = {b: sum(d["tokens"] for r in pools[b] for d in r["docs"] if d.get("tool"))
                                        / max(1, sum(sum(r["valid"]) for r in pools[b])) for b in spec.BUCKETS}
        # Stack Exchange (H4 pre-2024): calibration only, never fine-tune data, share-alike (docs carry the same flags)
        manifest["calibration_only_tokens"] = {b: sum(d["tokens"] for r in pools[b] for d in r["docs"]
                                                      if d.get("calibration_only")) for b in spec.BUCKETS}
        manifest["calibration_only_policy"] = spec.SE_USE
    env.atomic_json(OUT / "manifest.json", manifest)
    env.log(f"corpus done: {len(order)} rows, {total:,} tokens, achieved "
            + ", ".join(f"{b} {v:.3f}" for b, v in manifest["achieved_share"].items()), STAGE)


if __name__ == "__main__":
    main()
    if DOM:   # HF streaming leaves download threads behind; interpreter shutdown then hung 6 min and aborted (one
        sys.stdout.flush(); sys.stderr.flush()   # bucket, 2026-11-01) after the pool was already saved
        os._exit(0)
