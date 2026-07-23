from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import hdbscan
import numba
import numpy as np
import scipy
import sklearn
import umap
import yaml
from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm
from hdbscan._hdbscan_linkage import label as make_single_linkage
from hdbscan.hdbscan_ import (
    _hdbscan_boruvka_kdtree,
    _tree_to_labels,
)
from sklearn.neighbors import KDTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    REPOSITORY_ROOT
    / ".research"
    / "native-reference"
    / "test-audio-reference.npz"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / ".research"
    / "native-reference"
    / "clustering-parity"
    / "test-audio"
)
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "senko" / "cluster" / "conf" / "umap_hdbscan.yaml"
)
OFFLINE_CLUSTER_SOURCE = REPOSITORY_ROOT / "senko" / "cluster" / "cluster_cpu.py"

SCHEMA_VERSION = 1
FIXTURE_KIND = "senko-native-clustering-parity"
LITTLE_ENDIAN_DTYPES = {
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "int32": np.dtype("<i4"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate native UMAP/HDBSCAN intermediate fixtures without "
            "modifying or invoking Senko's full offline pipeline."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="NPZ produced by run_reference.py; must contain an embeddings member.",
    )
    parser.add_argument(
        "--source-member",
        default="embeddings",
        help="Name of the embedding matrix inside --source.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Offline Senko UMAP/HDBSCAN YAML configuration.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for seed-<N> and unseeded fixture directories.",
    )
    parser.add_argument(
        "--mode",
        choices=("seeded", "unseeded", "both"),
        default="both",
        help=(
            "seeded is a deterministic single-thread diagnostic oracle; "
            "unseeded mirrors ordinary offline Senko stochastic behavior."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="UMAP random_state used only by seeded diagnostic mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated fixture directory.",
    )
    parser.add_argument(
        "--validate-only",
        type=Path,
        help="Validate one existing fixture directory and do not generate data.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray, dtype: np.dtype[Any]) -> str:
    encoded = np.ascontiguousarray(array, dtype=dtype)
    return hashlib.sha256(memoryview(encoded).cast("B")).hexdigest()


def relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def load_offline_cluster_module() -> ModuleType:
    # Loading the source file directly avoids importing senko.__init__, which
    # initializes the full diarization stack on macOS. The file is not changed.
    spec = importlib.util.spec_from_file_location(
        "senko_native_cluster_cpu_fixture", OFFLINE_CLUSTER_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {OFFLINE_CLUSTER_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_configuration(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    try:
        args = document["cluster"]["args"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid Senko clustering config: {path}") from error

    expected = {
        "cluster_type",
        "mer_cos",
        "n_neighbors",
        "n_components",
        "min_samples",
        "min_cluster_size",
        "metric",
    }
    missing = expected.difference(args)
    if missing:
        raise ValueError(f"Missing clustering config keys: {sorted(missing)}")
    if args["cluster_type"] != "umap_hdbscan":
        raise ValueError(
            f"Expected umap_hdbscan config, found {args['cluster_type']!r}"
        )
    return dict(args)


def read_embeddings(path: Path, member: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if member not in archive:
            raise KeyError(f"{path} has no {member!r} member; found {archive.files}")
        embeddings = np.asarray(archive[member])
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a rank-2 embedding matrix, got {embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Offline fixture embeddings must be float32, got {embeddings.dtype}"
        )
    if embeddings.shape[0] < 3:
        raise ValueError("UMAP requires at least three embeddings")
    if not embeddings.flags.c_contiguous:
        embeddings = np.ascontiguousarray(embeddings)
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains non-finite values")
    return embeddings


def label_summary(labels: np.ndarray) -> dict[str, Any]:
    values, counts = np.unique(labels, return_counts=True)
    return {
        "populationCount": int(values.size),
        "clusterCountExcludingNoise": int(np.count_nonzero(values >= 0)),
        "noiseCount": int(counts[values == -1].sum()) if -1 in values else 0,
        "populations": [
            {"label": int(value), "count": int(count)}
            for value, count in zip(values, counts, strict=True)
        ],
    }


def write_artifact(
    output: Path,
    key: str,
    filename: str,
    array: np.ndarray,
    scalar_type: str,
    *,
    columns: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    dtype = LITTLE_ENDIAN_DTYPES[scalar_type]
    encoded = np.ascontiguousarray(array, dtype=dtype)
    path = output / filename
    encoded.tofile(path)
    metadata: dict[str, Any] = {
        "file": filename,
        "dtype": f"{scalar_type}-le",
        "shape": list(encoded.shape),
        "byteLength": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if columns is not None:
        metadata["columns"] = columns
    return key, metadata


def run_umap(
    embeddings: np.ndarray,
    config: dict[str, Any],
    *,
    random_state: int | None,
) -> tuple[umap.UMAP, np.ndarray, float]:
    components = min(int(config["n_components"]), embeddings.shape[0] - 2)
    # random_state makes UMAP's optimizer deterministic and disables parallel
    # execution. Setting n_jobs=1 explicitly avoids an avoidable warning while
    # preserving the behavior UMAP selects for any non-None random_state.
    n_jobs = 1 if random_state is not None else -1
    model = umap.UMAP(
        n_neighbors=int(config["n_neighbors"]),
        min_dist=0.0,
        n_components=components,
        metric=str(config["metric"]),
        random_state=random_state,
        n_jobs=n_jobs,
    )
    started = time.perf_counter()
    projection = model.fit_transform(embeddings)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if projection.dtype != np.float32:
        raise AssertionError(f"Unexpected UMAP output dtype {projection.dtype}")
    if projection.shape != (embeddings.shape[0], components):
        raise AssertionError(f"Unexpected UMAP output shape {projection.shape}")
    if not np.isfinite(projection).all():
        raise AssertionError("UMAP projection contains non-finite values")
    return model, np.ascontiguousarray(projection), elapsed_ms


def run_native_hdbscan_stages(
    projection: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    min_samples = min(projection.shape[0] - 1, int(config["min_samples"]))
    min_cluster_size = int(config["min_cluster_size"])
    leaf_size = 40
    core_dist_n_jobs = 4

    # This is the exact optimized branch selected by HDBSCAN(algorithm="best")
    # for Senko's 60-dimensional projection. It is expanded here solely so the
    # browser port can compare the native core distances, approximate Boruvka
    # MST, linkage, condensation, and EOM labels independently.
    hdb_input = np.asarray(projection, dtype=np.float64, order="C")
    started = time.perf_counter()
    tree = KDTree(hdb_input, metric="euclidean", leaf_size=leaf_size)
    algorithm = KDTreeBoruvkaAlgorithm(
        tree,
        min_samples,
        metric="euclidean",
        leaf_size=leaf_size // 3,
        approx_min_span_tree=True,
        n_jobs=core_dist_n_jobs,
    )
    core_reduced_distances = np.asarray(
        algorithm.core_distance_arr, dtype=np.float64
    ).copy()
    # KDTreeBoruvkaAlgorithm stores Euclidean core distances in its internal
    # reduced-distance representation (squared distance).
    core_distances = np.sqrt(core_reduced_distances)
    mst = np.asarray(algorithm.spanning_tree(), dtype=np.float64)
    mst = np.ascontiguousarray(mst[np.argsort(mst[:, 2]), :])
    single_linkage = np.ascontiguousarray(make_single_linkage(mst))
    (
        raw_labels,
        probabilities,
        stabilities,
        condensed,
        returned_single_linkage,
    ) = _tree_to_labels(
        projection,
        single_linkage,
        min_cluster_size=min_cluster_size,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        match_reference_implementation=False,
        cluster_selection_epsilon=0.0,
        cluster_selection_persistence=0.0,
        max_cluster_size=0,
        cluster_selection_epsilon_max=float("inf"),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not np.array_equal(returned_single_linkage, single_linkage):
        raise AssertionError("_tree_to_labels changed the single-linkage tree")

    condensed_matrix = np.column_stack(
        (
            condensed["parent"],
            condensed["child"],
            condensed["lambda_val"],
            condensed["child_size"],
        )
    ).astype(np.float64)

    # Verify this expanded capture path is exactly the private wrapper used by
    # HDBSCAN's public `best` algorithm at 60 dimensions.
    wrapper_single_linkage, wrapper_mst = _hdbscan_boruvka_kdtree(
        projection,
        min_samples=min_samples,
        alpha=1.0,
        metric="euclidean",
        p=2,
        leaf_size=leaf_size,
        approx_min_span_tree=True,
        gen_min_span_tree=True,
        core_dist_n_jobs=core_dist_n_jobs,
    )
    wrapper_labels = _tree_to_labels(
        projection,
        wrapper_single_linkage,
        min_cluster_size=min_cluster_size,
    )[0]
    public_labels = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    ).fit_predict(projection)

    validations = {
        "expandedMstEqualsPrivateWrapper": bool(np.array_equal(mst, wrapper_mst)),
        "expandedSingleLinkageEqualsPrivateWrapper": bool(
            np.array_equal(single_linkage, wrapper_single_linkage)
        ),
        "expandedLabelsEqualPrivateWrapper": bool(
            np.array_equal(raw_labels, wrapper_labels)
        ),
        "expandedLabelsEqualPublicHdbscan": bool(
            np.array_equal(raw_labels, public_labels)
        ),
        "mstWeightsSorted": bool(np.all(mst[:-1, 2] <= mst[1:, 2])),
        "allCoreDistancesFinite": bool(np.isfinite(core_distances).all()),
    }
    if not all(validations.values()):
        failed = [name for name, value in validations.items() if not value]
        raise AssertionError(f"Native HDBSCAN stage validation failed: {failed}")

    arrays = {
        "coreDistances": core_distances,
        "coreReducedDistances": core_reduced_distances,
        "mst": mst,
        "singleLinkage": single_linkage,
        "condensedTree": condensed_matrix,
        "rawLabels": np.asarray(raw_labels),
        "probabilities": np.asarray(probabilities),
        "stabilities": np.asarray(stabilities),
    }
    return arrays, validations, elapsed_ms


def postprocess_with_offline_senko(
    embeddings: np.ndarray,
    raw_labels: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    module = load_offline_cluster_module()
    common = module.CommonClustering(**config)
    labels = np.asarray(raw_labels).copy()
    started = time.perf_counter()
    minor_filtered = common.filter_minor_cluster(
        labels, embeddings, int(config["min_cluster_size"])
    ).copy()
    common_labels = minor_filtered.copy()
    if config["mer_cos"] is not None:
        common_labels = common.merge_by_cos(
            common_labels, embeddings, float(config["mer_cos"])
        )

    # This is the next label-only operation in Diarizer._perform_clustering.
    normalized = np.zeros(len(common_labels), dtype=np.int64)
    for normalized_label, native_label in enumerate(np.unique(common_labels)):
        normalized[common_labels == native_label] = normalized_label
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return minor_filtered, common_labels, normalized, elapsed_ms


def package_versions() -> dict[str, str]:
    names = (
        "hdbscan",
        "llvmlite",
        "numba",
        "numpy",
        "pynndescent",
        "scikit-learn",
        "scipy",
        "umap-learn",
    )
    return {name: importlib.metadata.version(name) for name in names}


def git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"commit": commit, "workingTreeDirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "workingTreeDirty": None}


def environment_metadata() -> dict[str, Any]:
    thread_variables = {
        name: os.environ.get(name)
        for name in (
            "NUMBA_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    return {
        "python": platform.python_version(),
        "pythonImplementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logicalCpuCount": os.cpu_count(),
        "numbaThreads": numba.get_num_threads(),
        "threadEnvironment": thread_variables,
        "packages": package_versions(),
        "moduleVersions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikitLearn": sklearn.__version__,
            "umap": umap.__version__,
        },
        "git": git_metadata(),
    }


def write_fixture(
    output: Path,
    source: Path,
    source_member: str,
    config_path: Path,
    config: dict[str, Any],
    embeddings: np.ndarray,
    *,
    random_state: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)

    generated_at = datetime.now(UTC).isoformat()
    total_started = time.perf_counter()
    umap_model, projection, umap_ms = run_umap(
        embeddings, config, random_state=random_state
    )
    hdb_arrays, hdb_validations, hdb_ms = run_native_hdbscan_stages(
        projection, config
    )
    minor_filtered, common_labels, normalized, postprocess_ms = (
        postprocess_with_offline_senko(
            embeddings, hdb_arrays["rawLabels"], config
        )
    )

    artifacts: dict[str, dict[str, Any]] = {}

    def add(
        key: str,
        filename: str,
        array: np.ndarray,
        scalar_type: str,
        *,
        columns: list[str] | None = None,
    ) -> None:
        artifact_key, metadata = write_artifact(
            output,
            key,
            filename,
            array,
            scalar_type,
            columns=columns,
        )
        artifacts[artifact_key] = metadata

    add(
        "umapProjection",
        "umap-projection.f32",
        projection,
        "float32",
        columns=[f"component-{index}" for index in range(projection.shape[1])],
    )
    graph = umap_model.graph_.tocsr()
    add("umapGraphIndptr", "umap-graph-indptr.i32", graph.indptr, "int32")
    add("umapGraphIndices", "umap-graph-indices.i32", graph.indices, "int32")
    add("umapGraphData", "umap-graph-data.f32", graph.data, "float32")
    if hasattr(umap_model, "_knn_indices"):
        add(
            "umapKnnIndices",
            "umap-knn-indices.i32",
            umap_model._knn_indices,
            "int32",
        )
        add(
            "umapKnnDistances",
            "umap-knn-distances.f32",
            umap_model._knn_dists,
            "float32",
        )
    add("umapSigmas", "umap-sigmas.f32", umap_model._sigmas, "float32")
    add("umapRhos", "umap-rhos.f32", umap_model._rhos, "float32")

    add(
        "hdbscanCoreDistances",
        "hdbscan-core-distances.f64",
        hdb_arrays["coreDistances"],
        "float64",
    )
    add(
        "hdbscanCoreReducedDistances",
        "hdbscan-core-reduced-distances.f64",
        hdb_arrays["coreReducedDistances"],
        "float64",
    )
    add(
        "hdbscanMst",
        "hdbscan-mst.f64",
        hdb_arrays["mst"],
        "float64",
        columns=["from", "to", "mutualReachabilityDistance"],
    )
    add(
        "hdbscanSingleLinkage",
        "hdbscan-single-linkage.f64",
        hdb_arrays["singleLinkage"],
        "float64",
        columns=["left", "right", "distance", "size"],
    )
    add(
        "hdbscanCondensedTree",
        "hdbscan-condensed-tree.f64",
        hdb_arrays["condensedTree"],
        "float64",
        columns=["parent", "child", "lambda", "childSize"],
    )
    add(
        "hdbscanRawLabels",
        "hdbscan-raw-labels.i32",
        hdb_arrays["rawLabels"],
        "int32",
    )
    add(
        "hdbscanProbabilities",
        "hdbscan-probabilities.f64",
        hdb_arrays["probabilities"],
        "float64",
    )
    add(
        "hdbscanStabilities",
        "hdbscan-stabilities.f64",
        hdb_arrays["stabilities"],
        "float64",
    )
    add(
        "minorFilteredLabels",
        "minor-filtered-labels.i32",
        minor_filtered,
        "int32",
    )
    add(
        "commonLabels",
        "common-labels.i32",
        common_labels,
        "int32",
    )
    add(
        "normalizedLabels",
        "normalized-labels.i32",
        normalized,
        "int32",
    )

    effective_epochs = 500 if embeddings.shape[0] <= 10_000 else 200
    mode = (
        "unseeded-production-sample"
        if random_state is None
        else "seeded-diagnostic"
    )
    manifest: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "fixtureKind": FIXTURE_KIND,
        "generatedAtUtc": generated_at,
        "mode": mode,
        "source": {
            "path": relative_or_absolute(source),
            "member": source_member,
            "fileSha256": sha256_file(source),
            "embeddings": {
                "dtype": "float32",
                "shape": list(embeddings.shape),
                "byteSha256": sha256_array(
                    embeddings, LITTLE_ENDIAN_DTYPES["float32"]
                ),
            },
        },
        "offlineSources": {
            "config": {
                "path": relative_or_absolute(config_path),
                "sha256": sha256_file(config_path),
            },
            "clusterImplementation": {
                "path": relative_or_absolute(OFFLINE_CLUSTER_SOURCE),
                "sha256": sha256_file(OFFLINE_CLUSTER_SOURCE),
            },
        },
        "configuration": {
            "common": {
                "clusterType": config["cluster_type"],
                "minimumClusterSize": int(config["min_cluster_size"]),
                "centroidMergeCosineThreshold": float(config["mer_cos"]),
            },
            "umap": {
                "neighbors": int(config["n_neighbors"]),
                "componentsConfigured": int(config["n_components"]),
                "componentsEffective": int(projection.shape[1]),
                "minimumDistance": 0.0,
                "metric": str(config["metric"]),
                "initialization": "spectral",
                "epochsConfigured": None,
                "epochsEffective": effective_epochs,
                "randomState": random_state,
                "requestedJobs": 1 if random_state is not None else -1,
                "parallelOptimizer": random_state is None,
                "a": float(umap_model._a),
                "b": float(umap_model._b),
            },
            "hdbscan": {
                "minSamples": int(config["min_samples"]),
                "minClusterSize": int(config["min_cluster_size"]),
                "metric": "euclidean",
                "algorithmConfigured": "best",
                "algorithmEffective": "boruvka_kdtree",
                "leafSize": 40,
                "approximateMinimumSpanningTree": True,
                "coreDistanceJobs": 4,
                "alpha": 1.0,
                "clusterSelectionMethod": "eom",
                "allowSingleCluster": False,
                "matchReferenceImplementation": False,
                "clusterSelectionEpsilon": 0.0,
                "clusterSelectionPersistence": 0.0,
                "maxClusterSize": 0,
                "clusterSelectionEpsilonMax": "Infinity",
            },
        },
        "environment": environment_metadata(),
        "timingsMs": {
            "umap": umap_ms,
            "hdbscanCapture": hdb_ms,
            "commonPostprocess": postprocess_ms,
            "totalBeforeValidation": (time.perf_counter() - total_started) * 1000.0,
            "note": (
                "Fixture-generation timings include intermediate capture and "
                "extra native parity checks; they are not production benchmarks."
            ),
        },
        "labelSummaries": {
            "hdbscanRaw": label_summary(hdb_arrays["rawLabels"]),
            "minorFiltered": label_summary(minor_filtered),
            "common": label_summary(common_labels),
            "normalized": label_summary(normalized),
        },
        "validations": {
            **hdb_validations,
            "projectionFinite": bool(np.isfinite(projection).all()),
            "commonLabelsContainNoMinorPopulations": all(
                population["count"] >= int(config["min_cluster_size"])
                for population in label_summary(common_labels)["populations"]
            ),
        },
        "artifacts": artifacts,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_fixture(output)
    return manifest


def validate_fixture(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported fixture schema {manifest.get('schemaVersion')!r}"
        )
    if manifest.get("fixtureKind") != FIXTURE_KIND:
        raise ValueError(f"Unexpected fixture kind {manifest.get('fixtureKind')!r}")
    for key, artifact in manifest["artifacts"].items():
        path = directory / artifact["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact {key}: {path}")
        if path.stat().st_size != artifact["byteLength"]:
            raise ValueError(f"Byte-length mismatch for {key}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {key}")
        scalar_type = artifact["dtype"].removesuffix("-le")
        dtype = LITTLE_ENDIAN_DTYPES[scalar_type]
        expected_elements = int(np.prod(artifact["shape"], dtype=np.int64))
        if path.stat().st_size != expected_elements * dtype.itemsize:
            raise ValueError(f"Shape/dtype size mismatch for {key}")
    if not all(manifest["validations"].values()):
        failed = [
            name for name, passed in manifest["validations"].items() if not passed
        ]
        raise ValueError(f"Fixture has failed validations: {failed}")
    return manifest


def main() -> None:
    args = parse_args()
    if args.validate_only is not None:
        manifest = validate_fixture(args.validate_only.resolve())
        print(
            json.dumps(
                {
                    "validated": str(args.validate_only),
                    "mode": manifest["mode"],
                    "artifacts": len(manifest["artifacts"]),
                },
                sort_keys=True,
            )
        )
        return

    source = args.source.resolve()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    embeddings = read_embeddings(source, args.source_member)
    config = load_configuration(config_path)

    modes: list[tuple[str, int | None]] = []
    if args.mode in ("unseeded", "both"):
        modes.append(("unseeded", None))
    if args.mode in ("seeded", "both"):
        modes.append((f"seed-{args.seed}", args.seed))

    summaries = []
    for directory_name, random_state in modes:
        output = output_root / directory_name
        manifest = write_fixture(
            output,
            source,
            args.source_member,
            config_path,
            config,
            embeddings,
            random_state=random_state,
            overwrite=args.overwrite,
        )
        summaries.append(
            {
                "output": str(output),
                "mode": manifest["mode"],
                "umapMs": manifest["timingsMs"]["umap"],
                "rawLabels": manifest["labelSummaries"]["hdbscanRaw"],
                "commonLabels": manifest["labelSummaries"]["common"],
            }
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
