"""Fixed validation evidence for a bounded step-4 experiment; never trains or controls the game.

API: prepare_roster(dataset, build(..., provenance=True)), write_json_exclusive(path, roster),
then evaluate(dataset, cached_features, roster, checkpoint_path, role=..., out=...).
Historical reference is restricted to the explicitly named post-checkpoint 2-10 recording, keys=0.
It is a descriptive control, never a strict validation/unknown-training-lineage exemption.
Opt-in tile-shapes-v1 uses a distinct v2 document contract and encoding-bound resource limits;
the default key-tile v1 roster/cache contract is unchanged. max_rows bounds feature rows, not raw archive
rows (the dataset builder separately bounds raw decoding). No cross-encoding historical comparison exists.
Every frozen cache's "prior" is gravity-v1 (CACHE_PRIOR). A checkpoint trained against another prior protocol
is evaluated only with that prior for the same rows (checkpoint_prior: rebuilt by wm.features.build and
checked field by field against the cache); otherwise the mismatch is an error.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from wm.dataset import canonical, digest, load_model, norm
from wm.features import EVENTS, LevelGeom, layout, model_layout, model_prior
from wm.protocol import GRAVITY_PRIOR, KEY_TILES_PROTOCOL, PRIOR_PROTOCOLS, feature_usage
from wm.rollout import World, rollout_batched

PROTOCOL = "fixed-step4-experiment-v1"
PROTOCOL_V2 = "fixed-step4-experiment-v2"
HORIZONS = (8, 16, 32, 48)
HISTORICAL_RUN = "collect_2-10_explore_01"
DATA_FIELDS = ("X", "raw", "prior", "yb", "full", "validation", "origin")
FEATURE_PROTOCOL = KEY_TILES_PROTOCOL       # this experiment's frozen 321-column cache encoding
CACHE_PRIOR = GRAVITY_PRIOR                 # the physics prior of every frozen cache's "prior" field


def _sealed(body):
    return dict(body, sha256=digest(canonical(body)))


def _verify_seal(document):
    if document.get("sha256") != digest(canonical({k: v for k, v in document.items() if k != "sha256"})):
        raise ValueError("experiment document hash mismatch")


def write_json_exclusive(path, document):
    """Never overwrite evidence, even when the same experiment is repeated."""
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def read_roster(path):
    with open(path, encoding="utf-8") as f:
        document = json.load(f)
    _verify_seal(document)
    if document.get("protocol") not in (PROTOCOL, PROTOCOL_V2):
        raise ValueError("unknown experiment roster protocol")
    return document


def data_hash(data):
    """Bind cache values, alignment, dtype and shape without expanding paths per sample."""
    h = hashlib.sha256()
    for key in DATA_FIELDS:
        a = np.ascontiguousarray(data[key])
        if a.dtype.hasobject:
            raise ValueError("object arrays cannot identify experiment data")
        h.update(canonical([key, a.dtype.str, list(a.shape)]))
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _resource_limits(max_shards, max_rows):
    for value, cap, name in ((max_shards, 160, "max_shards"), (max_rows, 2000000, "max_rows")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 < value <= cap:
            raise ValueError(f"{name} must be a positive integer <= {cap}")


def _validate_data(dataset, data, feature_protocol=FEATURE_PROTOCOL, max_shards=100, max_rows=410000):
    _resource_limits(max_shards, max_rows)
    width = layout(feature_protocol).n_in(4)
    n = len(data["X"])
    if len(dataset.entries) > max_shards or n > max_rows:
        raise ValueError("experiment exceeds its frozen shard/feature-row scope")
    if any(len(data[k]) != n for k in DATA_FIELDS) or data["X"].shape != (n, width):
        raise ValueError(f"step-4 experiment needs aligned {width}-column {feature_protocol} feature data")
    origin = data["origin"]
    if origin.shape != (n, 4) or origin.dtype.kind not in "iu":
        raise ValueError("origin must be integer [entry index, agent, start frame, keys]")
    if data["validation"].dtype != np.bool_ or data["full"].dtype != np.bool_:
        raise ValueError("validation and full masks must be boolean")
    if np.any(origin < 0) or np.any(origin[:, 0] >= len(dataset.entries)):
        raise ValueError("origin is outside this snapshot")
    for ei in np.unique(origin[:, 0]):
        e = dataset.entries[int(ei)]
        at = origin[:, 0] == ei
        for agent in np.unique(origin[at, 1]):
            owner = e["agents"].get(str(int(agent)))
            m = at & (origin[:, 1] == agent)
            if owner is None or owner["split"] == "excluded" or e["level"] in dataset.excluded or \
                    np.any(data["validation"][m] != (owner["split"] == "validation")):
                raise ValueError("feature membership disagrees with manifest owner")
    return origin


def prepare_roster(dataset, data, historical_source=None, seed=0, max_starts=128, per_level=8,
                   feature_protocol=FEATURE_PROTOCOL, max_shards=100, max_rows=410000):
    """Freeze val rows and rollout starts before training. Does not predict or decode recordings.

    Rollout candidates have two complete nonterminal four-frame chains (at f and f+4).
    Later horizons may be terminal/missing-data censored. 2-10 reserves half its level quota for each
    initial-input wind stratum when available; other levels are seeded, then levels round-robin.
    """
    if not 0 < max_starts <= 128 or not 0 < per_level <= 8:
        raise ValueError("start budgets must be within 128 total and 8 per level")
    origin = _validate_data(dataset, data, feature_protocol, max_shards, max_rows)
    version = PROTOCOL if (feature_protocol == KEY_TILES_PROTOCOL and max_shards == 100 and max_rows == 410000) else PROTOCOL_V2
    if historical_source and feature_protocol != KEY_TILES_PROTOCOL:
        raise ValueError("historical reference is restricted to the old key-tile encoding; no cross-encoding comparison")
    val = np.flatnonzero(data["validation"])
    if not len(val) or not np.any(~data["validation"]):
        raise ValueError("experiment needs training and validation samples")
    historical_source = norm(historical_source) if historical_source else None
    if historical_source and (historical_source not in dataset.sources or
                              Path(historical_source).name != "wm_data" or
                              Path(historical_source).parent.name != HISTORICAL_RUN):
        raise ValueError("historical reference source must be the explicit 2-10 exploration recording")
    # Duplicate full traces affect sample weights even when their split matches: require an explicit decision.
    groups = [a["group"] for e in dataset.entries for a in e["agents"].values() if a["split"] != "excluded"]
    if len(groups) != len(set(groups)):
        raise ValueError("exact replay groups repeat; stop for an explicit selection decision")
    wind = np.any(data["X"][:, 6:8] != 0, axis=1)
    nonterminal = ~(data["yb"][:, EVENTS["death"]].astype(bool) | data["yb"][:, EVENTS["finish"]].astype(bool))
    eligible = val[data["full"][val] & nonterminal[val]]
    by = {tuple(origin[i, :3]): int(i) for i in eligible}
    if len(by) != len(eligible):
        raise ValueError("duplicate feature origins")
    candidates = [int(i) for i in eligible if (int(origin[i, 0]), int(origin[i, 1]), int(origin[i, 2]) + 4) in by]
    rng = np.random.default_rng(seed)
    buckets, available = {}, {}
    for level in sorted({dataset.entries[int(origin[i, 0])]["level"] for i in candidates}):
        ids = [i for i in candidates if dataset.entries[int(origin[i, 0])]["level"] == level]
        rng.shuffle(ids)
        available[level] = {"wind_nonzero": sum(bool(wind[i]) for i in ids),
                            "wind_zero": sum(not bool(wind[i]) for i in ids)}
        if level == "2-10":
            fan, zero = [i for i in ids if wind[i]], [i for i in ids if not wind[i]]
            quota = per_level // 2
            picked = fan[:quota] + zero[:per_level - quota]
            picked += [i for i in ids if i not in picked][:per_level - len(picked)]
        else:
            picked = ids[:per_level]
        buckets[level] = picked
    chosen = []
    for k in range(per_level):
        for level in sorted(buckets):
            if k < len(buckets[level]) and len(chosen) < max_starts:
                i = buckets[level][k]
                ei, agent, frame, keys = map(int, origin[i])
                e = dataset.entries[ei]
                chosen.append({"sample": i, "entry": ei, "agent": agent, "frame": frame, "keys": keys,
                               "level": level, "group": e["agents"][str(agent)]["group"],
                               "wind_nonzero": bool(wind[i])})
    reference = [int(i) for i in val if historical_source and
                 dataset.entries[int(origin[i, 0])]["source"] == historical_source and
                 dataset.entries[int(origin[i, 0])]["level"] == "2-10" and origin[i, 3] == 0]
    selected = {level: {"wind_nonzero": sum(c["level"] == level and c["wind_nonzero"] for c in chosen),
                        "wind_zero": sum(c["level"] == level and not c["wind_nonzero"] for c in chosen),
                        "unique_groups": len({c["group"] for c in chosen if c["level"] == level})}
                for level in sorted(buckets)}
    body = {"protocol": version, "manifest_sha256": dataset.hash, "feature_protocol": feature_protocol,
                    "data_sha256": data_hash(data), "origin_columns": ["entry", "agent", "frame", "keys"],
                    "seed": int(seed), "one_step_indices": val.tolist(), "historical_indices": reference,
                    "historical_source": historical_source, "starts": chosen,
                    "budgets": {"max_starts": max_starts, "per_level": per_level}, "candidate_strata": available,
                    "selected_strata": selected,
                    "selection": "seeded candidates, 2-10 half wind/zero quota, levels round-robin",
                    "wind_definition": "nonzero initial model-input wind columns 6/7; zero is not proof of known zero physical force",
                    "limits": ["exact full traces only; shared prefixes/local correlation/shared tiles may remain",
                               "no unseen-level or unknown-training-lineage assurance",
                               "open-loop position is survivor-only with terminal/missing evidence censored",
                               "geometry includes recorded tile changes only up to each start; no imagined key pickup"]}
    if version == PROTOCOL_V2:
        body["encoding"] = {"protocol": feature_protocol, "step": 4, "n_in": layout(feature_protocol).n_in(4)}
        body["budgets"].update(max_shards=int(max_shards), max_rows=int(max_rows))
        body["open_loop_key_policy"] = "censor_recorded_key_change"
    return _sealed(body)


def _summary(values):
    values = np.asarray(values, np.float64)
    if len(values) and not np.all(np.isfinite(values)):
        raise ValueError("non-finite prediction metric")
    return {"n": len(values), "median": float(np.median(values)) if len(values) else None,
            "p90": float(np.percentile(values, 90)) if len(values) else None}


def _events(probability, truth):
    panel = {}
    for name in ("death", "finish"):
        y, p = truth[:, EVENTS[name]] > .5, probability[:, EVENTS[name]] > .5
        tp, positives, predicted = int(np.sum(y & p)), int(y.sum()), int(p.sum())
        panel[name] = {"n": len(y), "true_positive": tp, "actual_positive": positives,
                       "predicted_positive": predicted, "precision": tp / predicted if predicted else None,
                       "recall": tp / positives if positives else None}
    survivor = (truth[:, EVENTS["death"]] <= .5) & (truth[:, EVENTS["finish"]] <= .5)
    false = int(np.sum(survivor & (probability[:, EVENTS["death"]] > .5)))
    panel["survivor_death_false_alarm"] = {"n": int(survivor.sum()), "false_positive": false,
                                         "rate": false / int(survivor.sum()) if survivor.any() else None}
    return panel


def checkpoint_prior(dataset, data, protocol, feature_protocol=FEATURE_PROTOCOL, log=lambda *a: None):
    """{"protocol", "values"}: the prior of `protocol` for exactly the cache's rows, rebuilt by wm.features.build
    from the same manifest (step 4, every row, provenance) and accepted only when every other cached field
    (DATA_FIELDS) is identical, so the values align row by row; else an error."""
    from wm.features import build
    if protocol not in PRIOR_PROTOCOLS:
        raise ValueError(f"unknown prior protocol {protocol!r}")
    d = build(dataset.sources, dataset=dataset, step=4, frac=1, max_samples=0, provenance=True,
              protocol=feature_protocol, prior_protocol=protocol, log=log)
    for key in DATA_FIELDS:
        if key == "prior":
            continue
        if d[key].dtype != data[key].dtype or d[key].shape != data[key].shape or not np.array_equal(d[key], data[key]):
            raise ValueError(f"rebuilt {key} differs from the frozen cache; the prior cannot be aligned")
    return {"protocol": protocol, "values": d["prior"]}


def _one_step(model, dataset, data, idx, batch_size, deadline, key_strata=False):
    pos, vel, probabilities = [], [], []
    with torch.no_grad():
        for b in range(0, len(idx), batch_size):
            if time.perf_counter() > deadline:
                raise TimeoutError("experiment evaluation time budget reached")
            ids = idx[b:b + batch_size]
            p = model.predict(torch.as_tensor(data["X"][ids], dtype=torch.float32))
            mean, probability = p["mean"].numpy(), p["p"].numpy()
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(probability)):
                raise ValueError("non-finite model prediction")
            residual = data["prior_used"][ids] + mean * model.res_scale - data["raw"][ids]
            pos.extend(np.hypot(residual[:, 0], residual[:, 1]).tolist())
            vel.extend(np.hypot(residual[:, 2], residual[:, 3]).tolist())
            probabilities.append(probability)
    pos, vel = np.asarray(pos), np.asarray(vel)
    probability = np.concatenate(probabilities) if probabilities else np.empty((0, data["yb"].shape[1]))
    full = data["full"][idx]
    wind = np.any(data["X"][idx, 6:8] != 0, axis=1)
    level2 = np.array([dataset.entries[int(e)]["level"] == "2-10" for e in data["origin"][idx, 0]], bool)
    strata = {"overall": full, "wind_nonzero": full & wind, "wind_zero": full & ~wind,
              "2-10": full & level2, "2-10_wind_nonzero": full & level2 & wind, "2-10_wind_zero": full & level2 & ~wind}
    counts = {"all": len(idx), "full": int(full.sum()), "terminal_short": int((~full).sum())}
    if key_strata:
        owned = data["origin"][idx, 3] != 0
        strata.update(keys_zero=full & ~owned, keys_owned=full & owned)
        counts.update(keys_zero=int((~owned).sum()), keys_owned=int(owned.sum()),
                      keys_zero_full=int((full & ~owned).sum()), keys_owned_full=int((full & owned).sum()))
    return {"primary_full_four_frames": {name: {"position": _summary(pos[m]), "velocity": _summary(vel[m])}
                                         for name, m in strata.items()},
            "events_threshold": .5, "events_full": _events(probability[full], data["yb"][idx][full]),
            "events_terminal_short": _events(probability[~full], data["yb"][idx][~full]),
            "sample_counts": counts}


def _case_inputs(case, rows, world, geom, censor_key_change=False):
    """Only real consecutive alive truth/action/world prefixes; never pad after death or missing evidence."""
    f0, agent = case["frame"], case["agent"]
    by = {int(r["frame"]): r for r in rows if int(r["agent"]) == agent}
    wf = {int(w["f"]): w for w in world}
    if f0 not in by or int(by[f0]["keys"]) != case["keys"] or by[f0]["status"] != 0:
        raise ValueError("rollout roster start no longer matches captured state")
    geom.reset()
    fans, keep, known_fans = [], None, False
    for f in sorted(wf):
        if f > f0:
            break
        rec = wf[f]
        for ci, word in rec.get("tc", []) or []:
            geom.set_tile(ci, word)
        keep = rec.get("keep", keep)
        if "fx" in rec:
            fans, known_fans = rec["fx"].get("fans", []), "fans" in rec["fx"]
        keep = rec.get("keep", keep)
    acts, dts, worlds, reason = [], [], [], "horizon_limit"
    for j in range(max(HORIZONS)):
        f = f0 + j
        a, b, rec = by.get(f), by.get(f + 1), wf.get(f)
        if a is None or b is None:
            reason = "missing_row"
            break
        if a["status"] != 0 or b["status"] != 0:
            reason = "terminal_censored"
            break
        if censor_key_change and (int(a["keys"]) != case["keys"] or int(b["keys"]) != case["keys"]):
            reason = "key_mask_changed"
            break
        if a["action"] < 0:
            reason = "missing_action"
            break
        if rec is None or not isinstance(rec.get("dt"), (float, int)) or not np.isfinite(rec["dt"]) or rec["dt"] <= 0:
            reason = "missing_world_or_dt"
            break
        if "fx" in rec:
            fans, known_fans = rec["fx"].get("fans", []), "fans" in rec["fx"]
        acts.append(int(a["action"]))
        dts.append(float(rec["dt"]))
        keep = rec.get("keep", keep)
        worlds.append(World.from_record(rec, wf.get(f - 1), keep, fans))
    valid = [h for h in HORIZONS if h <= len(acts)]
    K = max(valid, default=0)
    return by, np.array([by[f0]]), np.array([acts[:K]], np.int64), np.array(dts[:K]), worlds[:K], valid, reason, known_fans


def _open_loop(model, dataset, roster, cases, deadline):
    errors = {str(h): [] for h in HORIZONS}
    counts = {str(h): {"selected_starts": len(cases), "survivors_scored": 0, "censored": {}} for h in HORIZONS}
    wanted = {}
    for case in cases:
        wanted.setdefault(case["entry"], []).append(case)
    sources = sorted({dataset.entries[ei]["source"] for ei in wanted})
    snapshots = (item for source in sources for item in dataset.iter_shards(source, split="validation"))
    for meta, rows, world in snapshots:
        ei = meta["_dataset_index"]
        if ei not in wanted:
            continue
        e = dataset.entries[ei]
        geom = LevelGeom(*dataset.level(e["source"], e["level"]), protocol=model_layout(model).protocol)
        for case in wanted[ei]:
            if time.perf_counter() > deadline:
                raise TimeoutError("experiment open-loop evaluation time budget reached")
            by, s0, actions, dts, worlds, valid, reason, _ = _case_inputs(
                case, rows, world, geom, censor_key_change=roster["protocol"] == PROTOCOL_V2)
            # Historical role must not acquire keys later in the measured prefix either.
            if e["source"] == roster["historical_source"] and any(
                    int(by[f]["keys"]) != 0 for f in range(case["frame"], case["frame"] + actions.shape[1] + 1)):
                raise ValueError("historical reference cases require keys=0 throughout their measured prefix")
            prediction = rollout_batched(model, geom, s0, actions, dts, worlds, deadline=deadline) if valid else None
            if prediction is not None and prediction["steps"] * model.step < max(valid):
                raise TimeoutError("experiment rollout stopped before its fixed horizon")
            for h in HORIZONS:
                row = counts[str(h)]
                if h not in valid:
                    row["censored"][reason] = row["censored"].get(reason, 0) + 1
                    continue
                truth = by[case["frame"] + h]
                x, y = prediction["x"][:, 0, h // model.step].mean(), prediction["y"][:, 0, h // model.step].mean()
                errors[str(h)].append(float(np.hypot(x - truth["x"], y - truth["y"])))
                row["survivors_scored"] += 1
    return {h: dict(position=_summary(errors[h]), **counts[h]) for h in errors}


def evaluate(dataset, data, roster, checkpoint, role="strict-model", out=None,
             expected_checkpoint_sha256=None, batch_size=2048, max_seconds=120, prior=None):
    """Evaluate immutable cache/roster. No generic compatibility bypass; old reference is separately scoped.
    prior: checkpoint_prior(...) for a checkpoint whose prior protocol is not the cache's (CACHE_PRIOR); one-step
    residuals then use it (open-loop rollouts always apply the checkpoint's own prior).

    Strict output includes a matched historical subset so its 2-10 numbers use EXACTLY the reference cases.
    Real rollout/prediction only happens when this function is explicitly called, never on import/prepare.
    """
    started = time.perf_counter()
    if batch_size <= 0 or batch_size > 8192 or not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("positive evaluation budget and batch size <=8192 required")
    if out is not None and Path(out).exists():
        raise FileExistsError(out)
    _verify_seal(roster)
    version = roster.get("protocol")
    feature_protocol = roster.get("feature_protocol")
    if version not in (PROTOCOL, PROTOCOL_V2) or roster.get("manifest_sha256") != dataset.hash or \
            (version == PROTOCOL and feature_protocol != FEATURE_PROTOCOL) or roster.get("data_sha256") != data_hash(data):
        raise ValueError("roster does not match this manifest/feature snapshot")
    limits = {} if version == PROTOCOL else {"max_shards": roster["budgets"].get("max_shards"),
                                            "max_rows": roster["budgets"].get("max_rows")}
    _validate_data(dataset, data, feature_protocol=feature_protocol, **limits)
    # Recompute the deterministic contract: edited, resealed case lists cannot introduce training agents.
    expected = prepare_roster(dataset, data, historical_source=roster["historical_source"], seed=roster["seed"],
                              max_starts=roster["budgets"]["max_starts"], per_level=roster["budgets"]["per_level"],
                              feature_protocol=feature_protocol, **limits)
    if expected != roster:
        raise ValueError("roster selection differs from deterministic validation contract")
    if role == "historical-reference" and feature_protocol != KEY_TILES_PROTOCOL:
        raise ValueError("historical reference is restricted to the old key-tile encoding; no cross-encoding comparison")
    model, identity = load_model(checkpoint)
    if expected_checkpoint_sha256 and identity["sha256"] != expected_checkpoint_sha256:
        raise ValueError("checkpoint bytes differ from requested identity")
    width = layout(feature_protocol).n_in(4)
    if model.step != 4 or model.mu.numel() != width or model_layout(model).protocol != feature_protocol:
        raise ValueError(f"this experiment requires a step-4, {width}-input {feature_protocol} checkpoint")
    model.eval()
    pp = model_prior(model)
    if pp == CACHE_PRIOR:
        if prior is not None:
            raise ValueError("the cache already holds this checkpoint's prior; no replacement is accepted")
        prior_values = data["prior"]
    else:
        if prior is None or prior.get("protocol") != pp:
            raise ValueError(f"checkpoint prior {pp} differs from the cache's {CACHE_PRIOR}; pass checkpoint_prior "
                             f"for {pp}")
        prior_values = np.asarray(prior["values"])
        if prior_values.shape != data["prior"].shape or prior_values.dtype != data["prior"].dtype or \
                not np.all(np.isfinite(prior_values)):
            raise ValueError("checkpoint prior values do not align with the cache")
    data = dict(data, prior_used=prior_values)
    if role == "strict-model":
        dataset.require_model(model)
        idx = np.asarray(roster["one_step_indices"], np.int64)
        cases = roster["starts"]
    elif role == "historical-reference":
        if (model.info or {}).get("dataset", {}).get("protocol") == "trace-split-v1":
            raise ValueError("a strict checkpoint cannot be disguised as historical reference")
        if not roster["historical_source"] or not roster["historical_indices"]:
            raise ValueError("historical reference needs its explicit keys=0 post-checkpoint source allowlist")
        idx = np.asarray(roster["historical_indices"], np.int64)
        allowed = set(idx.tolist())
        cases = [c for c in roster["starts"] if c["sample"] in allowed]
    else:
        raise ValueError("unknown experiment model role")
    deadline = started + max_seconds
    one = _one_step(model, dataset, data, idx, batch_size, deadline, key_strata=version == PROTOCOL_V2)
    loop = _open_loop(model, dataset, roster, cases, deadline)
    matched = None
    if role == "strict-model" and roster["historical_indices"]:
        matched_idx = np.asarray(roster["historical_indices"], np.int64)
        matched_cases = [c for c in cases if c["sample"] in set(matched_idx.tolist())]
        matched = {"one_step": _one_step(model, dataset, data, matched_idx, batch_size, deadline,
                                        key_strata=version == PROTOCOL_V2),
                   "open_loop": _open_loop(model, dataset, roster, matched_cases, deadline)}
    body = {"protocol": version, "role": role, "checkpoint": identity,
                      "checkpoint_metadata": model.info, "features": feature_usage(model.info),
                      "prior": {"protocol": pp, "cache_protocol": CACHE_PRIOR,
                                "values_sha256": hashlib.sha256(np.ascontiguousarray(prior_values)).hexdigest()},
                      "manifest_sha256": dataset.hash, "roster_sha256": roster["sha256"],
                      "data_sha256": roster["data_sha256"], "one_step": one, "open_loop": loop,
                      "matched_historical": matched, "elapsed_seconds": time.perf_counter() - started,
                      "limits": roster["limits"] + ["event threshold 0.5 is not a calibration claim",
                                 "historical-reference provenance is recorded evidence, not cryptographic no-training assurance"]}
    if version == PROTOCOL_V2:
        body.update(encoding=roster["encoding"], budgets=roster["budgets"],
                    open_loop_key_policy=roster["open_loop_key_policy"])
    result = _sealed(body)
    if out is not None:
        write_json_exclusive(out, result)
    return result
