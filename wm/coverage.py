"""Bounded evidence for choosing world-model data; no features, fitting, or model rollouts.

    python -m wm.coverage --help

Inventory reads sampled NPZ metadata and NPY headers only. Report processes whole shards of an immutable
manifest under explicit row/shard budgets. Neither output establishes training readiness or unseen-level skill.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from wm.buffer import transitions, chains
from wm.dataset import Dataset, norm, file_hash
from wm.features import AIR, group_of, platform_features, wind

PROTOCOL = "coverage-v1"
ONE_STEP = ("flag_ground", "flag_air", "flag_wall", "death_label", "finish_label", "landing_label", "blocked_label")
K_STEP = ("full_chains", "terminal_short_death", "terminal_short_finish", "terminal_short_other",
          "chain_landing_label", "chain_blocked_label")


def _rates(counts, names, denominator):
    return {name: counts[name] / denominator if denominator else None for name in names}


class _Bucket:
    def __init__(self):
        self.raw, self.proxy, self.metadata = Counter(), Counter(), Counter()
        self.groups, self.first_group_counts, self.group_outcomes = Counter(), {}, {}
        self.shards, self.agent_ids = set(), set()

    def add(self, entry, agent, core, proxy, metadata, outcome):
        group = entry["agents"][str(agent)]["group"]
        self.raw.update(core)
        self.proxy.update(proxy)
        self.metadata.update(metadata)
        self.groups[group] += 1
        self.first_group_counts.setdefault(group, core)
        self.group_outcomes.setdefault(group, set()).add(outcome)
        self.shards.add(entry["id"])
        self.agent_ids.add((entry["source"], agent))

    def result(self):
        unique = Counter()
        for value in self.first_group_counts.values():
            unique.update(value)
        n, u, tr = self.raw["agent_episodes"], len(self.groups), self.raw["transitions"]
        return {"shards_with_agents": len(self.shards), "pool_agent_ids": len(self.agent_ids),
                "unique_exact_trace_groups": u, "repeat_factor": n / u if u else None,
                "duplicate_episode_fraction": (n - u) / n if n else None,
                "raw_core_counts": dict(self.raw), "trace_dedup_core_counts": dict(unique),
                "one_step_label_rates": _rates(self.raw, ONE_STEP, tr),
                "dedup_one_step_label_rates": _rates(unique, ONE_STEP, unique["transitions"]),
                "chain_rates": _rates(self.raw, K_STEP, self.raw["chains"]),
                "metadata_outcome_counts": dict(self.metadata),
                "groups_with_conflicting_known_metadata_outcomes": sum(
                    len(v - {"unknown"}) > 1 for v in self.group_outcomes.values()),
                "proxy_counts": dict(self.proxy),
                "platform_present_rate": self.proxy["platform_logged_present"] / self.proxy["platform_context_known"]
                                          if self.proxy["platform_context_known"] else None,
                "platform_near_200_rate": self.proxy["platform_near_200"] / self.proxy["platform_context_known"]
                                         if self.proxy["platform_context_known"] else None,
                "wind_nonzero_rate": self.proxy["wind_nonzero"] / self.proxy["wind_context_known"]
                                     if self.proxy["wind_context_known"] else None}


def _proxies(s, world):
    """Per eligible transition at its starting frame. No platform support/landing inference."""
    names = ("platform_context_known", "platform_logged_present", "platform_near_200", "wind_context_known", "wind_nonzero")
    out = {name: np.zeros(len(s), bool) for name in names}
    wf = {int(w["f"]): w for w in world if "f" in w}
    frames, fans, known, upto = sorted(wf), [], False, 0
    for frame in sorted(set(s["frame"].tolist())):
        while upto < len(frames) and frames[upto] <= frame:
            rec = wf[frames[upto]]
            if isinstance(rec.get("fx"), dict) and "fans" in rec["fx"]:
                fans = rec["fx"]["fans"]
                known = isinstance(fans, list)
            upto += 1
        if frame not in wf:
            continue                           # missing world context is not observed zero exposure
        idx = np.nonzero(s["frame"] == frame)[0]
        rec = wf[frame]
        pf = rec.get("pf") or []                # collector omits an empty platform list
        out["platform_context_known"][idx] = True
        out["platform_logged_present"][idx] = bool(pf)
        if pf:
            out["platform_near_200"][idx] = platform_features(s["x"][idx], s["y"][idx], pf, None, 0)[:, 5] > 0
        if known:
            out["wind_context_known"][idx] = True
            fx, fy = wind(s["x"][idx], s["y"][idx], fans)
            out["wind_nonzero"][idx] = np.hypot(fx, fy) > 0
    return out


def _timeline(world):
    records = {int(w["f"]): w for w in world if "f" in w}
    c = Counter(world_frames=len(records))
    for rec in records.values():
        dt = rec.get("dt")
        if dt is None:
            c["dt_missing"] += 1
        elif not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
            c["dt_invalid"] += 1
        else:
            c["dt_known"] += 1
            c["dt_outside_locked_16_667ms"] += int(abs(dt - 16.667) > 0.01)
    return c


def _metadata(meta, agent, last_status):
    statuses = meta.get("status")
    status = statuses[agent] if isinstance(statuses, list) and 0 <= agent < len(statuses) else None
    known = status in (0, 1, 2)
    outcome = {0: "alive", 1: "death", 2: "finish"}.get(status, "unknown")
    return Counter({"metadata_known": int(known), "metadata_unknown": int(not known),
                    "metadata_" + outcome: 1,
                    "terminal_outcome_without_terminal_row": int(status in (1, 2) and last_status == 0),
                    "terminal_status_mismatch": int(status in (1, 2) and last_status in (1, 2) and status != last_status),
                    "partial_episodes": int(bool(meta.get("partial"))),
                    "metadata_timeout_alive": int(status == 0 and meta.get("why") == "time")}), outcome


def report(dataset, max_shards, max_rows, step=1, max_seconds=None):
    if step < 1:
        raise ValueError("step must be positive")
    started, budget = time.monotonic(), {}
    total, buckets, timeline, timelines = _Bucket(), {}, Counter(), {}
    for meta, rows, world in dataset.iter_shards(max_shards=max_shards, max_rows=max_rows,
                                                max_seconds=max_seconds, budget=budget):
        entry = dataset.entry(meta)
        tr = transitions(rows)
        s, nxt = tr["s"], tr["s2"]
        groups = group_of(s)
        labels = {"flag_ground": groups == 0, "flag_air": groups == 1, "flag_wall": groups == 2,
                  "death_label": nxt["status"] == 1, "finish_label": nxt["status"] == 2,
                  "landing_label": ((s["flags"] & AIR) != 0) & ((nxt["flags"] & AIR) == 0),
                  "blocked_label": (np.abs(s["vx"]) > 40) & (np.abs(nxt["vx"]) < .25 * np.abs(s["vx"]))}
        proxies = _proxies(s, world)
        ch = chains(rows, step) if step > 1 else None
        c = _timeline(world)
        timeline.update(c)
        timelines.setdefault((entry["source"], entry["level"]), Counter()).update(c)
        for a in np.unique(rows["agent"]):
            agent = int(a)
            r = rows[rows["agent"] == a]
            last_status = int(r[np.argmax(r["frame"])]["status"])
            mask = s["agent"] == a
            n = int(mask.sum())
            core = Counter(raw_rows=len(r), agent_episodes=1, transitions=n,
                           last_row_alive=int(last_status == 0), last_row_death=int(last_status == 1),
                           last_row_finish=int(last_status == 2),
                           alive_at_record_end=int(np.count_nonzero(tr["truncated_rows"]["agent"] == a)))
            core.update({name: int(values[mask].sum()) for name, values in labels.items()})
            if ch is not None:
                cm = ch["s"]["agent"] == a
                full = ch["full"]
                core.update(chains=int(cm.sum()), full_chains=int((cm & full).sum()),
                            terminal_short_death=int((cm & ~full & (ch["s2"]["status"] == 1)).sum()),
                            terminal_short_finish=int((cm & ~full & (ch["s2"]["status"] == 2)).sum()),
                            terminal_short_other=int((cm & ~full & ~np.isin(ch["s2"]["status"], [1, 2])).sum()),
                            chain_landing_label=int(ch["landing"][cm].sum()),
                            chain_blocked_label=int(ch["blocked"][cm].sum()))
            proxy = Counter({name: int(values[mask].sum()) for name, values in proxies.items()})
            proxy.update(platform_context_unknown=n - proxy["platform_context_known"],
                         wind_context_unknown=n - proxy["wind_context_known"])
            metadata, outcome = _metadata(meta, agent, last_status)
            key = (entry["source"], entry["level"], entry["agents"][str(agent)]["split"])
            buckets.setdefault(key, _Bucket()).add(entry, agent, core, proxy, metadata, outcome)
            total.add(entry, agent, core, proxy, metadata, outcome)
    processed = set(budget["processed_ids"])
    result = {"protocol": PROTOCOL, "dataset": dataset.reference(), "step": step,
              "requested_budgets": {"max_shards": max_shards, "max_rows": max_rows, "max_seconds": max_seconds},
              "budget": budget, "measurement_complete_manifest": len(processed) == len(dataset.entries),
              "manifest_complete_listed_sources": dataset.document["coverage"]["complete"],
              "unprocessed_ids": [e["id"] for e in dataset.entries if e["id"] not in processed],
              "elapsed_seconds": round(time.monotonic() - started, 6), "training_ready": "not established",
              "totals": total.result(),
              "source_level_split": [{"source": k[0], "level": k[1], "split": k[2], **v.result()}
                                     for k, v in sorted(buckets.items())],
              "world_timeline_counts": dict(timeline),
              "source_level_world_timeline": [{"source": k[0], "level": k[1], **dict(v)}
                                              for k, v in sorted(timelines.items())],
              "definitions": {
                  "agent_episodes": "one (shard, agent) occurrence; partial/censored recordings included",
                  "pool_agent_ids": "distinct (source, pool index); not independent experience",
                  "transitions": "one-frame same-agent consecutive rows with a recorded source action, as wm.buffer.transitions",
                  "one_step_denominator": "eligible one-frame transitions at their source row",
                  "flag_groups": "wm.features.group_of: WALL first, then AIR, otherwise ground; bit proxies, not contact truth",
                  "landing_blocked": "AIR to not-AIR; abs(vx)>40 then abs(next_vx)<0.25*abs(vx); label heuristics",
                  "dedup": "one copy per full exact-trace group within each bucket; global groups may span sources",
                  "metadata_outcomes": "raw meta.status[agent] evidence; never inserted as a synthetic terminal row or deduplicated",
                  "alive_at_record_end": "transitions.truncated_rows proxy; not proof of a wave timeout",
                  "chains": "K-frame chains, full or terminal-short death/finish; missing time-limit tails may produce no chain",
                  "chain_denominator": "eligible K-frame/terminal-short chains, not one-frame transitions",
                  "platform": "a platform was logged; nearest centre <200 units; no support/landing-success inference",
                  "platform_absence": "present world record without pf means no platform logged (collector omits empty lists)",
                  "wind": "features.wind nonzero force after fx.fans becomes known, carried forward within the shard",
                  "proxy_denominators": "eligible transitions with world/fan context; unknown context excluded from rates",
                  "proxy_dedup": "not provided: world timeline is not part of the exact-trace group hash",
                  "dt_denominator": "unique recorded world frames, once per shard; not agent rows or simulation proof",
                  "limits": "only processed whole shards; no sampling/oversampling, feature fitting, rollout or readiness guarantee"}}
    return result


def _save(path, result):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="bounded metadata/header samples, no state arrays")
    sources = inventory.add_mutually_exclusive_group(required=True)
    sources.add_argument("--data", nargs="+")
    sources.add_argument("--source-request", help="JSON with a sources list; budgets still explicit on the CLI")
    inventory.add_argument("--max-candidates", type=int, required=True)
    inventory.add_argument("--per-source-cap", type=int, required=True)
    inventory.add_argument("--seed", type=int, default=0)
    inspect = sub.add_parser("report", help="whole-shard physics/data-label coverage within a manifest")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--max-shards", type=int, required=True)
    inspect.add_argument("--max-rows", type=int, required=True)
    inspect.add_argument("--max-seconds", type=float)
    inspect.add_argument("--step", type=int, default=1)
    for p in (inventory, inspect):
        p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        raise SystemExit("coverage output already exists: " + args.out)
    if args.command == "inventory":
        request = json.loads(Path(args.source_request).read_text(encoding="utf-8")) if args.source_request else None
        result = Dataset.catalog(request["sources"] if request else args.data,
                                 args.max_candidates, args.seed, args.per_source_cap)
        if request is not None:
            result["source_request"] = {"path": norm(args.source_request), "sha256": file_hash(args.source_request),
                                        "declared_model_sha256": request.get("model_sha256")}
    else:
        result = report(Dataset.load(args.manifest), args.max_shards, args.max_rows, args.step, args.max_seconds)
    _save(args.out, result)
    print(json.dumps({"protocol": result["protocol"], "out": norm(args.out),
                      "elapsed_seconds": result["elapsed_seconds"], "budget": result.get("budget"),
                      "metadata_inspected": result.get("metadata_inspected")}, indent=2))


if __name__ == "__main__":
    main()
