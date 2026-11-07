"""D2 domain-mask selection library (numpy only, CPU). docs/domain/D2_METHOD.md, items 1-7 and 10.

Inputs, read-only:
  A192 (original 9-bucket) bank   $V41_WORK/s03/bank/L*.npz, routing $V41_WORK/s03/routing/L*_{idx,w}.npy,
                                  corpus $V41_WORK/corpus (rows.jsonl: bucket_id, heldout; valid.npy)
  domain pass banks               $V41_WORK/dom/s03_<pass>/bank/L*.npz  (rows = the spec's SUB-AREAS, full
                                  `buckets` list, only the pass's sub-areas non-zero), routing alongside,
                                  corpus $V41_WORK/dom/corpus_<pass> (rows.jsonl: sub_id, heldout)
  guards                          $V41_WORK/s04/position_protect.json + s04/candidates.json super-experts

WHY F IS RECONSTRUCTED. s03 banks ONE pooled HOPE matrix per layer and pass (F_sum over every bucket of the
pass, hooks.FFNStats). Per-source F (needed to normalise each source before weighting, item 2) is not in the
bank. It is rebuilt here from the banked routing (every token's 6 expert ids and gates, held-out rows included)
with s = g * nbar_a(e), nbar_a(e) = the bank's mean ||f|| of expert e on that sub-area (bucket for A192), and
the DIAGONAL replaced by the exact banked second moment (sq = sum of s^2). The same approximation is what
select.image_saliency already uses. `validate_F` measures it against the banked pooled F: relative Frobenius
error and the Jaccard of the HOPE keep-sets solved on banked vs reconstructed F.

Normalisation (item 2): every source's per-layer vector (REAP sum, routing mass) and F is divided by its own
total, so the mix weights act on shares, not on token counts or score scale. No candidate subtracts or divides
by a general score (item 3): general enters only additively and as a protected slice.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ops.lib import drift, env
from ops.lib import select as SEL

E, K_KEEP = 384, 192
N_LAYERS = int(os.environ.get("V41_D2_NLAYERS", "40"))     # tests shrink this; production is 40
WORK = env.WORK
DOM = WORK / "dom"
D2 = DOM / "d2"
A192_BANK = WORK / "s03"
A192_CORPUS = WORK / "corpus"
S04 = WORK / "s04"

# Domain buckets selected by default (the published subset; the original run selected ten tier-1 buckets).
TIER1 = ["gamedev", "mlsys", "robotics", "quantum"]
ORIG9 = ["general", "multilingual", "agentic", "code", "vision", "math", "science", "finance", "longctx"]
OTHER6 = [b for b in ORIG9 if b not in ("general", "code", "agentic")]
W_GRID = (0.35, 0.45, 0.6)
PF_GRID = (0.02, 0.04, 0.06)
CRITERIA = ("hope", "reap", "mass")
ST_GRID = (1, 0)                  # selftraces sub-area weight in the domain source
SHARED_SA = ("selftraces", "tool-use", "agentic-chat")   # every bucket has these; split by row parity in halves
DEFAULT = dict(crit="hope", w=0.45, pf=0.04, st=1)
ENTROPY_REF = 0.85 * np.log2(E)   # 7.30 bits
POLY_K = (4, 6)
SHRINK_GRID = (0.25, 0.5, 0.75)
# Diagnostic only, never shipped: measures what the general share costs in-domain (owner request 2026-11-02).
# Above 0.6 the non-domain sources keep the w=0.45 proportions, scaled to 1 - w. Names end in "_diag".
DIAG_W, DIAG_BUCKETS = 0.8, ("gamedev",)
STABILITY_FLOOR = 0.9


def cand_name(crit: str, w: float, pf: float, st: int, extra: str = "") -> str:
    return f"{crit}_w{round(w * 100)}_pf{round(pf * 100)}_st{st}{extra}"


DEFAULT_NAME = cand_name(**DEFAULT)


def mix_weights(w: float, diag: bool = False) -> dict[str, float]:
    """{source: weight}; 'DOMAIN' is the domain bucket. code+agentic 0.25 split evenly (each normalised on its
    own), general 0.15, the other six original buckets share 1 - w - 0.40 evenly. diag allows w up to DIAG_W."""
    if diag and w > 0.6:
        base = mix_weights(0.45)
        out = {s: v * (1 - w) / 0.55 for s, v in base.items() if s != "DOMAIN"}
        return {"DOMAIN": w, **out}
    assert 0 < w <= 0.6 + 1e-9, "D2_METHOD item 4: nothing above 0.6"
    rest = 1.0 - w - 0.40
    out = {"DOMAIN": w, "code": 0.125, "agentic": 0.125, "general": 0.15}
    for b in OTHER6:
        out[b] = rest / len(OTHER6)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    return out


def unit(x: np.ndarray) -> np.ndarray:
    s = float(x.sum())
    return x / s if s > 0 else np.zeros_like(x)


def jaccard(a, b) -> float:
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / max(len(a | b), 1)


# ----------------------------------------------------------------------------- inputs
def pass_dirs() -> dict[str, dict]:
    """Every domain pass with a corpus dir: {pass: {bank, routing, corpus, buckets, complete}}."""
    out = {}
    for c in sorted(DOM.glob("corpus_*")):
        if not (c / "manifest.json").exists() or c.name.startswith("corpus_draw2"):
            continue
        p = c.name[len("corpus_"):]
        man = json.load(open(c / "manifest.json"))
        d = DOM / f"s03_{p}"
        log = d / "s03.log"
        done = log.exists() and "s03 complete" in log.read_text(errors="replace")
        layers = all((d / "bank" / f"L{L}.npz").exists() for L in range(N_LAYERS))
        out[p] = {"pass": p, "s03": d, "bank": d / "bank", "routing": d / "routing", "corpus": c,
                  "buckets": man["pass_buckets"], "complete": bool(done and layers)}
    return out


def available_buckets(require_complete: bool = True, bank_override: dict | None = None) -> dict[str, dict]:
    """bucket -> pass info for passes whose s03 completed (all 40 layers banked)."""
    out = {}
    for p, info in pass_dirs().items():
        if bank_override and p in bank_override:
            info = {**info, **bank_override[p]}
        if not info["complete"] and (require_complete or not (info["bank"] / "L0.npz").exists()):
            continue
        for b in info["buckets"]:
            out[b] = info
    return out


class Corpus:
    """rows.jsonl + valid.npy of one corpus; token -> stat-row id (sub_id or bucket_id) and held-out flag."""

    def __init__(self, d: Path, key: str):
        self.rows = [json.loads(l) for l in open(Path(d) / "rows.jsonl")]
        T = sum(r["length"] for r in self.rows)
        v = np.load(Path(d) / "valid.npy", mmap_mode="r")
        assert v.shape[0] == T, (d, v.shape, T)
        self.valid = np.asarray(v).astype(bool)
        self.T = T
        self.key = key
        self.sid = np.empty(T, np.int32)
        self.held = np.zeros(T, bool)
        self.parity = np.zeros(T, np.int8)      # row parity WITHIN its stat row (0 = A half, 1 = B half)
        seen: dict[int, int] = {}
        for r in self.rows:
            s = slice(r["offset"], r["offset"] + r["length"])
            k = int(r[key])
            self.sid[s] = k
            self.held[s] = bool(r["heldout"])
            if not r["heldout"]:
                self.parity[s] = seen.get(k, 0) % 2
                seen[k] = seen.get(k, 0) + 1


_CORPORA: dict = {}


def corpus_cached(d: Path, key: str) -> "Corpus":
    k = (str(d), key)
    if k not in _CORPORA:
        _CORPORA[k] = Corpus(d, key)
    return _CORPORA[k]


def mean_norm(bank: dict) -> np.ndarray:
    """[n_rows, E] mean ||f|| per stat row, falling back to the all-row mean where the row never routed e."""
    allm = bank["nrm"].sum(0) / np.maximum(bank["cnt"].sum(0), 1)
    return np.where(bank["cnt"] > 0, bank["nrm"] / np.maximum(bank["cnt"], 1), allm[None])


def load_routing(d: Path, L: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.load(Path(d) / f"L{L}_idx.npy", mmap_mode="r")
    w = np.load(Path(d) / f"L{L}_w.npy", mmap_mode="r")
    assert idx.shape[0] == T and w.shape == idx.shape, (d, L, idx.shape, T)
    return idx, w


def _F_of(idx: np.ndarray, s: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """sum over tokens of the K x K outer products of s, scattered to (idx_a, idx_b). float64."""
    F = np.zeros(E * E)
    for i in range(0, idx.shape[0], chunk):
        ii = idx[i:i + chunk].astype(np.int64)
        ss = s[i:i + chunk]
        flat = (ii[:, :, None] * E + ii[:, None, :]).ravel()
        F += np.bincount(flat, weights=(ss[:, :, None] * ss[:, None, :]).ravel(), minlength=E * E)
    return F.reshape(E, E)


def layer_sources(bank: dict, C: Corpus, routing_dir: Path, L: int, stat_ids, *, halves: bool = False,
                  shared_ids=(), exact_diag: bool = True) -> dict:
    """Per stat row a (sub-area or A192 bucket): reconstructed F_a [E,E] on calibration tokens; with
    `halves`, also F of the even-row half (key (a, 0)) for the ids in shared_ids (odd half = full - even);
    held-out approximate saliency ho[a] [E] = sum over held-out valid tokens of g * nbar; ho_mass[a] = sum g."""
    idx, w = load_routing(routing_dir, L, C.T)
    nb = mean_norm(bank)
    calib = C.valid & ~C.held
    held = C.valid & C.held
    out = {"F": {}, "F_even": {}, "ho": {}, "ho_mass": {}}
    for a in stat_ids:
        m = calib & (C.sid == a)
        ii = np.asarray(idx[m]).astype(np.int64)
        ss = np.asarray(w[m], np.float64) * nb[a][ii]
        F = _F_of(ii, ss)
        if exact_diag:
            F[np.diag_indices(E)] = bank["sq"][a]
        out["F"][a] = F
        if halves and a in shared_ids:
            ev = C.parity[m] == 0
            Fe = _F_of(ii[ev], ss[ev])
            if exact_diag:     # exact second moment is not split by row: scale it by the half's share of the approx diag
                d_full, d_ev = np.diag(_F_of_diag(ii, ss)), np.diag(_F_of_diag(ii[ev], ss[ev]))
                Fe[np.diag_indices(E)] = bank["sq"][a] * np.where(d_full > 0, d_ev / np.maximum(d_full, 1e-300), 0.0)
            out["F_even"][a] = Fe
        h = held & (C.sid == a)
        hi = np.asarray(idx[h]).astype(np.int64)
        hw = np.asarray(w[h], np.float64)
        out["ho"][a] = np.bincount(hi.ravel(), weights=(hw * nb[a][hi]).ravel(), minlength=E)
        out["ho_mass"][a] = np.bincount(hi.ravel(), weights=hw.ravel(), minlength=E)
    return out


def _F_of_diag(ii: np.ndarray, ss: np.ndarray) -> np.ndarray:
    return np.diag(np.bincount(ii.ravel(), weights=(ss * ss).ravel(), minlength=E))


def routing_gat_check(bank: dict, C: Corpus, routing_dir: Path, L: int, stat_ids) -> float:
    """max relative error between banked gat[a] and the routing gate sum over calibration tokens of row a
    (routing w is stored fp16, so ~1e-3 is expected). Proves the token -> stat-row mapping."""
    idx, w = load_routing(routing_dir, L, C.T)
    calib = C.valid & ~C.held
    worst = 0.0
    for a in stat_ids:
        m = calib & (C.sid == a)
        g = np.bincount(np.asarray(idx[m]).astype(np.int64).ravel(), weights=np.asarray(w[m], np.float64).ravel(),
                        minlength=E)
        ref = bank["gat"][a]
        worst = max(worst, float(np.abs(g - ref).sum() / max(ref.sum(), 1e-30)))
    return worst


def guards(L: int, superc: np.ndarray, protect: dict) -> np.ndarray:
    return np.union1d(np.asarray(protect.get(str(L), []), int), np.flatnonzero(superc[L])).astype(int)


def load_guards(dom_fmax: np.ndarray | None = None) -> tuple[np.ndarray, dict, dict]:
    """(superc bool [40, E], position protect {L: ids}, info). Super-experts = s04's recorded set UNION
    drift.super_expert_candidates on the A192 bank fmax maxed with every domain pass fmax (item 1: the set is
    calibration-independent in principle; the union makes sure no domain-only spike is missed)."""
    cj = json.load(open(S04 / "candidates.json"))
    sup = np.zeros((N_LAYERS, E), bool)
    for L, ids in cj["guards"]["super_expert_candidates"].items():
        sup[int(L), ids] = True
    info = {"s04_super": int(sup.sum())}
    if dom_fmax is not None:
        fa = np.stack([SEL.load_bank(A192_BANK, L)["fmax"] for L in range(N_LAYERS)])
        extra = drift.super_expert_candidates(np.maximum(fa, dom_fmax))
        info["added_from_domain_fmax"] = int((extra & ~sup).sum())
        sup |= extra
    protect = json.load(open(S04 / "position_protect.json"))["protect"]
    info["position_protect"] = int(sum(len(v) for v in protect.values()))
    info["super_total"] = int(sup.sum())
    return sup, protect, info


# ----------------------------------------------------------------------------- per-layer selection
class LayerInputs:
    """Everything one layer needs for every candidate of every bucket: normalised per-source vectors and F."""

    def __init__(self, L: int, avail: dict[str, dict], guard: np.ndarray, *, buckets, halves: bool = True,
                 a192: tuple | None = None):
        self.L, self.guard = L, guard
        # A192 original 9
        b9 = SEL.load_bank(A192_BANK, L)
        names9 = [str(x) for x in b9["buckets"]]
        assert names9 == ORIG9, names9
        C9 = a192 if a192 is not None else corpus_cached(A192_CORPUS, "bucket_id")
        s9 = layer_sources(b9, C9, A192_BANK / "routing", L, range(9))
        self.src = {}          # name -> {"reap": [E], "mass": [E], "F": [E,E]} UNnormalised
        for i, b in enumerate(ORIG9):
            self.src[b] = {"reap": b9["sum"][i], "mass": b9["gat"][i], "F": s9["F"][i]}
        self.orig_cal = b9["sum"]                         # [9, E] calibration REAP per original bucket
        self.orig_ho = np.stack([s9["ho"][i] for i in range(9)])
        self.gen_protect_P = b9["sum"][ORIG9.index("general")][None]
        # domain passes
        self.sub = {}          # bucket -> {subarea name: {"id", "reap", "mass", "F", "F_even", "ho"}}
        self.subnames = None
        self.entropy = {}
        banks = {}
        for b in buckets:
            info = avail[b]
            if info["pass"] not in banks:
                bk = SEL.load_bank(info["s03"], L)
                banks[info["pass"]] = bk
            bk = banks[info["pass"]]
            sa = [str(x) for x in bk["subareas"]]
            self.subnames = sa
            ids = [i for i, n in enumerate(sa) if n.split("/")[0] == b and bk["cnt"][i].sum() > 0]
            assert ids, f"L{L}: bucket {b} has no live sub-area in pass {info['pass']}"
            C = corpus_cached(info["corpus"], "sub_id")
            shared = [i for i in ids if sa[i].split("/", 1)[1] in SHARED_SA]
            s = layer_sources(bk, C, info["routing"], L, ids, halves=halves, shared_ids=shared)
            self.sub[b] = {sa[i].split("/", 1)[1]: {"id": i, "reap": bk["sum"][i], "mass": bk["gat"][i], "F": s["F"][i],
                                                     "F_even": s["F_even"].get(i), "ho": s["ho"][i],
                                                     "cnt": bk["cnt"][i]} for i in ids}

    # --- sources
    def domain(self, b: str, st: int = 1, part: str = "full", topical_half: dict | None = None) -> dict:
        """The domain source of bucket b: sub-areas summed (their own corpus proportions). st=0 drops selftraces.
        part: full | A | B (stability halves: topical sub-areas by `topical_half`, shared ones by row parity)."""
        reap, mass, F = np.zeros(E), np.zeros(E), np.zeros((E, E))
        for n, d in self.sub[b].items():
            if n == "selftraces" and not st:
                continue
            if part == "full":
                reap += d["reap"]; mass += d["mass"]; F += d["F"]
            elif n in SHARED_SA:
                Fe = d["F_even"]
                F += Fe if part == "A" else d["F"] - Fe
            elif topical_half[n] == part:
                F += d["F"]
        return {"reap": reap, "mass": mass, "F": F}

    def mixed(self, b: str, w: float, st: int = 1, dom: dict | None = None, diag: bool = False) -> dict:
        dom = dom or self.domain(b, st)
        out = {"reap": np.zeros(E), "mass": np.zeros(E), "F": np.zeros((E, E))}
        for s, wt in mix_weights(w, diag).items():
            src = dom if s == "DOMAIN" else self.src[s]
            if wt == 0:
                continue
            for k in out:
                out[k] += wt * unit(src[k])
        return out

    def gen_slice(self, pf: float) -> np.ndarray:
        return SEL.pool_protect(self.gen_protect_P, pf, np.array([True]))

    def solve(self, crit: str, score: dict, protect: np.ndarray) -> list[int]:
        prot = np.union1d(self.guard, protect).astype(int)
        if crit == "hope":
            pr = SEL.prune_set("hope", n_keep=K_KEEP, protect=prot, F=score["F"], E=E)
        else:
            pr = SEL.prune_set(crit, n_keep=K_KEEP, protect=prot, crit={crit: score[crit]}, E=E)
        keep = SEL.keep_from_pruned(pr, E)
        assert len(keep) == K_KEEP and set(prot.tolist()) <= set(keep), (self.L, crit)
        return keep

    def fair(self, b: str, w: float, st: int, pf: float) -> list[int]:
        """Domain-fair allocation (item 6, only used if routing is near-flat): each source gets round(weight x
        free slots) of the slots, filled from its own normalised REAP ranking, round-robin in weight order."""
        dom = self.domain(b, st)
        keep = set(np.union1d(self.guard, self.gen_slice(pf)).tolist())
        free = K_KEEP - len(keep)
        ws = sorted(mix_weights(w).items(), key=lambda kv: -kv[1])
        quota = {s: int(round(wt * free)) for s, wt in ws}
        ranks = {s: list(np.argsort(-unit((dom if s == "DOMAIN" else self.src[s])["reap"]), kind="stable")) for s, _ in ws}
        while len(keep) < K_KEEP:
            moved = False
            for s, _ in ws:
                if quota[s] <= 0 or len(keep) >= K_KEEP:
                    continue
                while ranks[s] and ranks[s][0] in keep:
                    ranks[s].pop(0)
                if ranks[s]:
                    keep.add(int(ranks[s].pop(0))); quota[s] -= 1; moved = True
            if not moved:
                for s in quota:
                    quota[s] += 1
        return sorted(int(x) for x in keep)

    # --- retained mass of a keep-set
    def retained(self, keep: list[int], buckets) -> dict:
        m = np.zeros(E, bool); m[keep] = True
        out = {"cal": {}, "ho": {}}
        for b in buckets:
            for n, d in self.sub[b].items():
                out["cal"][f"{b}/{n}"] = float(d["reap"][m].sum() / max(d["reap"].sum(), 1e-30))
                out["ho"][f"{b}/{n}"] = float(d["ho"][m].sum() / max(d["ho"].sum(), 1e-30)) if d["ho"].sum() > 0 else None
        for i, b in enumerate(ORIG9):
            out["cal"][f"A192/{b}"] = float(self.orig_cal[i][m].sum() / max(self.orig_cal[i].sum(), 1e-30))
            out["ho"][f"A192/{b}"] = float(self.orig_ho[i][m].sum() / max(self.orig_ho[i].sum(), 1e-30)) \
                if self.orig_ho[i].sum() > 0 else None
        return out


def entropy_bits(cnt: np.ndarray) -> float:
    p = cnt / max(cnt.sum(), 1e-30)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
