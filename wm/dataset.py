"""Immutable, bounded gameplay snapshots and per-agent exact-trace splits.

CLI consumers must choose --manifest or --legacy-split. Legacy weights cannot be relabelled as clean by
attaching a new manifest. Trace identity ignores agent/wave/reward, canonicalizes signed zero, and rejects
non-finite physical state. It does not detect shared prefixes, near duplicates, or semantic similarity.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

PROTOCOL = "trace-split-v1"
LEGACY = "legacy-global-mod5"
ROOT = Path(__file__).resolve().parent.parent
TRACE_FIELDS = ("frame", "x", "y", "vx", "vy", "flags", "state", "status", "keys", "held", "action")


def norm(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _source_ids(dirs):
    from plan.demos import excluded_demo_folder
    sources = sorted({norm(d) for d in dirs})
    rejected = [d for d in sources if excluded_demo_folder(d)]
    if rejected:
        raise ValueError("user-excluded demo folders cannot enter a manifest: " + ", ".join(rejected))
    return sources


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def file_hash(path):
    with open(path, "rb") as f:
        h = hashlib.file_digest(f, "sha256")
    return h.hexdigest()


def load_model(path):
    from wm.model import WorldModel
    data = Path(path).read_bytes()
    return WorldModel.load(io.BytesIO(data)), {"path": norm(path), "sha256": digest(data)}


def geometry_id(words, meta, level):
    # ZIP headers and unrelated recording metadata are not geometry identity.
    geometry = {k: meta.get(k) for k in ("ox", "oy", "spawn", "goals", "objs", "portalTypes")}
    return digest(canonical({"level": level, "shape": list(words.shape), "geometry": geometry}) +
                  np.asarray(words, dtype="<u4").tobytes())


def trace_id(rows, geometry):
    rows = np.sort(rows, order="frame")
    if len(np.unique(rows["frame"])) != len(rows):
        raise ValueError("an agent has duplicate frame numbers")
    columns = []
    for name in TRACE_FIELDS:
        a = rows[name]
        if a.dtype.kind == "f":
            if not np.isfinite(a).all():
                raise ValueError("non-finite physical state in trace: " + name)
            a = np.where(a == 0, 0.0, a)       # -0 and +0 have the same physical meaning
        columns.append(a.tolist())
    return digest(canonical({"geometry": geometry, "fields": TRACE_FIELDS, "columns": columns}))


def split_of(group):
    return "validation" if int(group, 16) % 5 == 0 else "train"


def _read_npz(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return z["rows"].copy(), json.loads(str(z["meta"]))


def _rows_header(z):
    """Read only the NPY header from a ZIP member; never materialize the state array."""
    with z.open("rows.npy") as f:
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, _ = reader(f)
    if len(shape) != 1:
        raise ValueError("gameplay rows must be a one-dimensional array")
    return int(shape[0])


def _positive_budget(value, name):
    if value is not None and (not np.isfinite(value) or value <= 0):
        raise ValueError(name + " must be finite and positive")


def _level(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return z["words"].copy(), json.loads(str(z["meta"]))


def _checked(path, expected):
    data = Path(path).read_bytes()
    if digest(data) != expected:
        raise ValueError("dataset content changed: " + str(path))
    return data


class Dataset:
    def __init__(self, document, path=None):
        self.document, self.path = document, norm(path) if path else None
        body = {k: v for k, v in document.items() if k != "sha256"}
        if document.get("protocol") != PROTOCOL or document.get("sha256") != digest(canonical(body)):
            raise ValueError("invalid dataset manifest protocol or hash")
        self.hash = document["sha256"]
        self.entries = document["entries"]
        self._entries = {e["id"]: e for e in self.entries}
        self.sources = document["sources"]
        if self.sources != _source_ids(self.sources):
            raise ValueError("dataset declared sources are not normalized and unique")
        self.excluded = set(document["excluded_levels"])
        self._levels = {}
        ids = set()
        for e in self.entries:
            if e["id"] in ids or e["source"] != norm(e["source"]) or e["source"] not in self.sources:
                raise ValueError("duplicate or non-normalized dataset source identity")
            ids.add(e["id"])
            if e["id"] != norm(Path(e["source"]) / e["shard"]):
                raise ValueError("dataset shard identity mismatch")
            for agent in e["agents"].values():
                expected = "excluded" if e["level"] in self.excluded else split_of(agent["group"])
                if agent["split"] != expected:
                    raise ValueError("dataset trace split was changed")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), path)

    @classmethod
    def create(cls, dirs, max_shards, excluded=(), parent=None, selected_paths=None,
               max_raw_rows=None, max_seconds=None):
        if max_shards < 1:
            raise ValueError("max_shards must be positive")
        _positive_budget(max_raw_rows, "max_raw_rows")
        _positive_budget(max_seconds, "max_seconds")
        started = time.monotonic()
        sources = _source_ids(dirs)
        if parent is not None and set(parent.excluded) != set(excluded):
            raise ValueError("an extension cannot change excluded levels")
        entries = []
        if parent is not None:
            # Verify the old snapshot; never move its members when extending.
            for _ in parent.iter_shards():
                pass
            entries = list(parent.entries)
            sources = sorted(set(sources) | set(parent.sources))
        candidates = sorted(norm(p) for d in sources for p in Path(d).glob("wave_*.npz"))
        selected = {e["id"] for e in entries}
        todo = [p for p in candidates if p not in selected]
        if selected_paths is not None:
            requested = [norm(p) for p in selected_paths]
            if len(set(requested)) != len(requested):
                raise ValueError("selected_paths contains duplicate identities")
            if set(requested) - set(todo):
                raise ValueError("selected_paths must be new shards in declared sources")
            if len(requested) > max_shards:
                raise ValueError("explicit selection exceeds max_shards")
            todo = sorted(requested)
        raw_rows = 0
        for p in todo[:max_shards]:
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                raise TimeoutError("manifest creation soft time budget reached between shards; no manifest created")
            shard_data = Path(p).read_bytes()
            with zipfile.ZipFile(io.BytesIO(shard_data)) as z:
                row_count = _rows_header(z)
            if max_raw_rows is not None and raw_rows + row_count > max_raw_rows:
                raise ValueError("selected shards exceed max_raw_rows before decoding " + p)
            rows, meta = _read_npz(shard_data)
            raw_rows += len(rows)
            level = meta.get("level")
            if not isinstance(level, str) or Path(level).name != level:
                raise ValueError("invalid level identity in " + p)
            source = norm(Path(p).parent)
            world_path = str(Path(p).with_suffix("")) + "_world.json.gz"
            level_path = str(Path(source) / "levels" / (level + ".npz"))
            world_data, level_data = Path(world_path).read_bytes(), Path(level_path).read_bytes()
            words, level_meta = _level(level_data)
            geom = geometry_id(words, level_meta, level)
            agents = {}
            for a in np.unique(rows["agent"]):
                try:
                    group = trace_id(rows[rows["agent"] == a], geom)
                except ValueError as exc:
                    raise ValueError(f"{p}, agent {int(a)}: {exc}") from exc
                agents[str(int(a))] = {"group": group, "split": "excluded" if level in excluded else split_of(group)}
            entries.append({"id": p, "source": source, "shard": Path(p).name, "level": level,
                            "shard_sha256": digest(shard_data), "world_path": norm(world_path),
                            "world_sha256": digest(world_data), "level_path": norm(level_path),
                            "level_sha256": digest(level_data), "geometry": geom, "agents": agents})
        entries.sort(key=lambda e: e["id"])
        body = {"protocol": PROTOCOL, "sources": sources,
                "excluded_levels": sorted(set(excluded)), "entries": entries,
                "parent_sha256": parent.hash if parent else None,
                "coverage": {"selected": len(entries), "candidates": len(candidates),
                             "complete": len(entries) == len(candidates), "new_shard_cap": max_shards,
                             "scope": "listed sources at creation; no whole-archive claim"}}
        if selected_paths is not None or max_raw_rows is not None or max_seconds is not None:
            body["selection"] = {"mode": "explicit paths" if selected_paths is not None else "lexical cap",
                                 "requested_paths": sorted(requested) if selected_paths is not None else None,
                                 "max_new_raw_rows": max_raw_rows, "new_raw_rows": raw_rows,
                                 "max_seconds": max_seconds, "time_limit": "soft, checked between shards"}
        return cls(dict(body, sha256=digest(canonical(body))))

    @staticmethod
    def catalog(dirs, max_candidates, seed=0, per_source_cap=None):
        """Bounded metadata/header inventory. Uninspected files have unknown levels and trace coverage."""
        _positive_budget(max_candidates, "max_candidates")
        _positive_budget(per_source_cap, "per_source_cap")
        started = time.monotonic()
        sources = _source_ids(dirs)
        candidates, statistics = {}, {}
        for d in sources:
            files = sorted(Path(d).glob("wave_*.npz"))
            n = min(len(files), per_source_cap if per_source_cap is not None else max_candidates)
            rng = np.random.default_rng(int(digest(canonical([seed, d]))[:16], 16))
            bounds = np.linspace(0, len(files), n + 1).astype(int) if n else []
            indices = [int(rng.integers(a, b)) for a, b in zip(bounds[:-1], bounds[1:])]
            candidates[d] = [files[k] for k in indices]
            statistics[d] = {"files_listed": len(files), "metadata_inspected": 0}
        picked = []
        for k in range(max((len(v) for v in candidates.values()), default=0)):
            for d in sources:
                if k < len(candidates[d]) and len(picked) < max_candidates:
                    picked.append((d, candidates[d][k]))
        entries = []
        for d, p in picked:
            with zipfile.ZipFile(p) as z:
                with z.open("meta.npy") as f:
                    meta = json.loads(str(np.lib.format.read_array(f, allow_pickle=False)))
                row_count = _rows_header(z)
            entries.append({"id": norm(p), "source": d, "shard": p.name, "level": meta.get("level"),
                            "raw_rows": row_count, "partial": bool(meta.get("partial", False)),
                            "source_mode": meta.get("source"), "compressed_bytes": p.stat().st_size})
            statistics[d]["metadata_inspected"] += 1
        total = sum(s["files_listed"] for s in statistics.values())
        return {"protocol": "dataset-catalog-v1", "sources": sources, "seed": seed,
                "max_candidates": max_candidates, "per_source_cap": per_source_cap,
                "selection": "one seeded filename-index sample per evenly spaced bin; sources round-robin",
                "source_counts": statistics, "entries": entries, "files_listed": total,
                "metadata_inspected": len(entries), "rows_decoded": 0,
                "complete_listed_metadata": len(entries) == total,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "limitations": "metadata/header sample only; uninspected levels, events and exact traces unknown"}

    def save(self, path):
        # Exclusive create: a manifest is never edited or replaced in place.
        with open(path, "x", encoding="utf-8") as f:
            json.dump(self.document, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        self.path = norm(path)

    def iter_shards(self, source=None, split=None, max_shards=None, max_rows=None, max_seconds=None, budget=None):
        """Verify at the owner boundary, then decode the exact bytes checked (not a later reread)."""
        source = norm(source) if source is not None else None
        for value, name in ((max_shards, "max_shards"), (max_rows, "max_rows"), (max_seconds, "max_seconds")):
            _positive_budget(value, name)
        started = time.monotonic()
        status = budget if budget is not None else {}
        status.update(processed_shards=0, decoded_rows=0, processed_ids=[], stop_reason="complete",
                      next_entry=None, next_rows=None)
        local = {d: {p.name: k for k, p in enumerate(sorted(Path(d).glob("wave_*.npz")))}
                 for d in self.sources if source is None or source == d}
        for index, e in enumerate(self.entries):
            if source is not None and e["source"] != source:
                continue
            if split and not any(a["split"] == split for a in e["agents"].values()):
                continue
            if max_shards is not None and status["processed_shards"] >= max_shards:
                status.update(stop_reason="shard_budget", next_entry=e["id"])
                break
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                status.update(stop_reason="soft_time_budget", next_entry=e["id"])
                break
            shard_data = _checked(e["id"], e["shard_sha256"])
            if max_rows is not None:
                with zipfile.ZipFile(io.BytesIO(shard_data)) as z:
                    row_count = _rows_header(z)
                if status["decoded_rows"] + row_count > max_rows:
                    status.update(stop_reason="row_budget", next_entry=e["id"], next_rows=row_count)
                    break
            rows, meta = _read_npz(shard_data)
            world = json.loads(gzip.decompress(_checked(e["world_path"], e["world_sha256"])))
            level_data = _checked(e["level_path"], e["level_sha256"])
            self._levels[(e["source"], e["level"])] = _level(level_data)
            if meta.get("level") != e["level"] or {str(int(a)) for a in np.unique(rows["agent"])} != set(e["agents"]):
                raise ValueError("dataset metadata/agents no longer match the manifest")
            meta = dict(meta, _dataset_entry=e["id"], _dataset_index=index,
                        _dataset_local_index=local[e["source"]][e["shard"]])
            # This is an internal unique identifier only, never split membership.
            rows["wave"] = index
            status["processed_shards"] += 1
            status["decoded_rows"] += len(rows)
            status["processed_ids"].append(e["id"])
            if split is not None:
                rows = self.rows_for(meta, rows, split)
            yield meta, rows, world
        status["elapsed_seconds"] = round(time.monotonic() - started, 6)

    def entry(self, meta):
        return self._entries[meta["_dataset_entry"]]

    def source_list(self, requested=()):
        if requested and {norm(d) for d in requested} != set(self.sources):
            raise ValueError("data folders must match the manifest snapshot's sources")
        return self.sources

    def rows_for(self, meta, rows, split):
        allowed = [int(a) for a, v in self.entry(meta)["agents"].items() if v["split"] == split]
        return rows[np.isin(rows["agent"], allowed)]

    def validation_mask(self, meta, rows):
        allowed = [int(a) for a, v in self.entry(meta)["agents"].items() if v["split"] == "validation"]
        return np.isin(rows["agent"], allowed)

    def level(self, source, name):
        return self._levels[(norm(source), name)]

    def reference(self):
        return {"protocol": PROTOCOL, "path": self.path, "sha256": self.hash,
                "coverage": self.document["coverage"]}

    def require_model(self, model):
        ref = (getattr(model, "info", {}) or {}).get("dataset", {})
        if ref.get("protocol") != PROTOCOL or ref.get("sha256") != self.hash:
            raise ValueError("checkpoint is legacy or belongs to a different manifest; strict init/evaluation "
                             "requires the same snapshot (extension fine-tuning is not supported)")
        from wm.protocol import prior_protocol, require_feature_protocol
        require_feature_protocol(model.info)
        prior_protocol(model.info)


def add_cli(ap):
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", help="immutable trace-split-v1 dataset manifest")
    mode.add_argument("--legacy-split", action="store_true", help="explicit legacy global-modulo compatibility; "
                      "not clean evaluation")


def cli_dataset(args, model=None):
    if args.manifest:
        dataset = Dataset.load(args.manifest)
        if model is not None:
            dataset.require_model(model)
        print(f"dataset {PROTOCOL} {dataset.hash}; coverage {dataset.document['coverage']}", flush=True)
        return dataset
    if model is not None and (getattr(model, "info", {}) or {}).get("dataset", {}).get("protocol") == PROTOCOL:
        raise ValueError("strict checkpoint cannot be evaluated or fine-tuned using --legacy-split")
    print("LEGACY global-mod5: historical compatibility, NOT clean evaluation", flush=True)
    return None


def sample_validation(data, strict=False):
    val = data["validation"] if strict else data["shard"] % 5 == 0
    if not val.any() or val.all():
        if strict:
            raise ValueError("manifest sampling must leave both training and validation transitions")
        val = data["shard"] == data["shard"].max()
    return val


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--data", nargs="+", required=True)
    create.add_argument("--out", required=True)
    create.add_argument("--max-shards", type=int, required=True, help="maximum NEW shards to fingerprint")
    create.add_argument("--parent", help="extend an immutable snapshot into a new manifest")
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        raise SystemExit("manifest already exists: " + args.out)
    from wm.buffer import eval_levels
    data = Dataset.create(args.data, args.max_shards, excluded=eval_levels(),
                          parent=Dataset.load(args.parent) if args.parent else None)
    data.save(args.out)
    print(json.dumps(data.reference(), indent=2))


if __name__ == "__main__":
    main()
