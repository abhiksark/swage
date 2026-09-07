# benchmarks/real_traces.py
"""Strict offline loader for frozen real benchmark row-length traces."""

import hashlib
import json
import pathlib
import statistics
from collections.abc import Mapping
from typing import Any, cast

_TRACE_NAME = "soc-epinions1-outdegree-v1"
_TRACE_FILENAME = "soc-Epinions1-outdegree-sample-v1.txt"
_PROVENANCE_FILENAME = (
    "soc-Epinions1-outdegree-sample-v1.provenance.json"
)
_PROVENANCE_SHA256 = (
    "77af425385baf18f87ae66c613651c0fdb693e45a12951ce48344bba83f7d3ee"
)
_TRACE_SHA256 = (
    "97138fdc92d297aa026480edda222827f9c1e84cc2d001d66a52d18e3201ee48"
)
_TRACE_COUNT = 32_768
_MAX_LENGTH = 4096

_PROVENANCE_SCHEMA: dict[str, Any] = {
    "schema_version": int,
    "source": {
        "collection": str,
        "group": str,
        "dataset_name": str,
        "matrix_id": int,
        "landing_url": str,
        "download_url": str,
        "retrieved_at": str,
        "archive": {
            "filename": str,
            "size_bytes": int,
            "sha256": str,
            "md5": str,
            "etag": str,
            "last_modified": str,
        },
        "member": {
            "path": str,
            "size_bytes": int,
            "sha256": str,
        },
    },
    "matrix_market": {
        "banner": str,
        "object": str,
        "format": str,
        "field": str,
        "symmetry": str,
        "index_base": int,
        "rows": int,
        "columns": int,
        "stored_entries": int,
    },
    "graph": {
        "advertised_vertex_count": int,
        "advertised_edge_count": int,
        "actual_vertex_count": int,
        "vertex_universe_rule": str,
        "orientation": str,
        "zero_row_policy": str,
        "duplicate_policy": str,
        "self_loop_policy": str,
        "symmetry_expansion_policy": str,
    },
    "extraction": {
        "version": str,
        "domain_separator_display": str,
        "domain_separator_hex": str,
        "row_id_encoding": str,
        "digest": str,
        "ranking": str,
        "sample_count": int,
        "replacement": bool,
        "output_order": str,
    },
    "derived_trace": {
        "filename": str,
        "encoding": str,
        "sha256": str,
        "count": int,
        "sum": int,
        "min": int,
        "median": float,
        "p95": int,
        "max": int,
        "generator_revision": str,
        "redistributed_fields": str,
    },
    "license": {
        "identifier": str,
        "url": str,
        "collection_policy_url": str,
        "attribution": str,
        "modification_notice": str,
    },
    "citations": [
        {
            "authors": str,
            "title": str,
            "doi": str,
            "url": str,
        }
    ],
}


def _sha256(data: bytes) -> str:
    """Return the lowercase SHA-256 hexadecimal digest for data."""
    return hashlib.sha256(data).hexdigest()


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_schema(value: object, schema: object, path: str) -> None:
    """Require exact keys and JSON value types for a recursive schema."""
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        expected_keys = set(schema)
        actual_keys = set(value)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"{path} schema mismatch: missing={missing}, extra={extra}"
            )
        for key, child_schema in schema.items():
            _validate_schema(value[key], child_schema, f"{path}.{key}")
        return
    if isinstance(schema, list):
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        if not value:
            raise ValueError(f"{path} must not be empty")
        for index, item in enumerate(value):
            _validate_schema(item, schema[0], f"{path}[{index}]")
        return
    if type(value) is not schema:
        expected_name = schema.__name__
        raise ValueError(f"{path} must have type {expected_name}")


def _require_pinned_provenance(provenance: Mapping[str, object]) -> None:
    """Require source, extraction, licensing, and citation pins."""
    source = cast(dict[str, object], provenance["source"])
    matrix = cast(dict[str, object], provenance["matrix_market"])
    graph = cast(dict[str, object], provenance["graph"])
    extraction = cast(dict[str, object], provenance["extraction"])
    derived = cast(dict[str, object], provenance["derived_trace"])
    license_record = cast(dict[str, object], provenance["license"])
    citations = cast(list[dict[str, object]], provenance["citations"])

    expected = {
        "schema_version": 1,
        "source.dataset_name": "soc-Epinions1",
        "source.matrix_id": 2284,
        "source.archive.size_bytes": 1_552_639,
        "source.archive.sha256": (
            "ed75c7f919799a8571aab592e2b89d8e54bc734530b4b8c5ab2071386ecd698e"
        ),
        "source.member.path": "soc-Epinions1/soc-Epinions1.mtx",
        "source.member.size_bytes": 5_158_059,
        "source.member.sha256": (
            "f24ae6890fc53c9e9b153812c4d22cd2e87a3642a1af12eeefd06dae0888f627"
        ),
        "matrix_market.banner": (
            "%%MatrixMarket matrix coordinate pattern general"
        ),
        "matrix_market.rows": 75_888,
        "matrix_market.columns": 75_888,
        "matrix_market.stored_entries": 508_837,
        "graph.actual_vertex_count": 75_879,
        "extraction.version": _TRACE_NAME,
        "extraction.domain_separator_hex": (
            "73776167652f736f632d4570696e696f6e73312f726f772d73616d706c652f"
            "763100"
        ),
        "extraction.row_id_encoding": "uint64_be",
        "extraction.sample_count": _TRACE_COUNT,
        "derived_trace.filename": _TRACE_FILENAME,
        "derived_trace.sha256": _TRACE_SHA256,
        "derived_trace.count": _TRACE_COUNT,
        "license.identifier": "CC-BY-4.0",
        "license.url": "https://creativecommons.org/licenses/by/4.0/",
    }
    archive = cast(dict[str, object], source["archive"])
    member = cast(dict[str, object], source["member"])
    actual = {
        "schema_version": provenance["schema_version"],
        "source.dataset_name": source["dataset_name"],
        "source.matrix_id": source["matrix_id"],
        "source.archive.size_bytes": archive["size_bytes"],
        "source.archive.sha256": archive["sha256"],
        "source.member.path": member["path"],
        "source.member.size_bytes": member["size_bytes"],
        "source.member.sha256": member["sha256"],
        "matrix_market.banner": matrix["banner"],
        "matrix_market.rows": matrix["rows"],
        "matrix_market.columns": matrix["columns"],
        "matrix_market.stored_entries": matrix["stored_entries"],
        "graph.actual_vertex_count": graph["actual_vertex_count"],
        "extraction.version": extraction["version"],
        "extraction.domain_separator_hex": extraction[
            "domain_separator_hex"
        ],
        "extraction.row_id_encoding": extraction["row_id_encoding"],
        "extraction.sample_count": extraction["sample_count"],
        "derived_trace.filename": derived["filename"],
        "derived_trace.sha256": derived["sha256"],
        "derived_trace.count": derived["count"],
        "license.identifier": license_record["identifier"],
        "license.url": license_record["url"],
    }
    if actual != expected:
        changed = sorted(
            key for key in expected if actual[key] != expected[key]
        )
        raise ValueError(f"provenance pin mismatch at {', '.join(changed)}")

    expected_dois = {
        "10.1007/978-3-540-39718-2_23",
        "10.1145/2049662.2049663",
    }
    actual_dois = {citation["doi"] for citation in citations}
    if actual_dois != expected_dois:
        raise ValueError("provenance citation DOI mismatch")
    if not license_record["attribution"]:
        raise ValueError("provenance attribution must not be empty")
    notice = license_record["modification_notice"]
    if "No vertex IDs or edges are redistributed." not in notice:
        raise ValueError("provenance modification notice is incomplete")


def _parse_trace(trace_data: bytes) -> list[int]:
    """Parse the canonical unsigned-decimal trace representation."""
    if not trace_data.endswith(b"\n"):
        raise ValueError("trace must end with LF")
    if b"\r" in trace_data:
        raise ValueError("trace must use LF, not CRLF")
    lines = trace_data.splitlines()
    lengths: list[int] = []
    for line_number, line in enumerate(lines, start=1):
        if not line or not line.isdigit():
            raise ValueError(
                f"trace line {line_number} must be an unsigned decimal integer"
            )
        if len(line) > 1 and line.startswith(b"0"):
            raise ValueError(f"trace line {line_number} is not canonical")
        value = int(line)
        if value > _MAX_LENGTH:
            raise ValueError(
                f"trace line {line_number} exceeds maximum {_MAX_LENGTH}"
            )
        lengths.append(value)
    return lengths


def _summarize(lengths: list[int]) -> dict[str, int | float]:
    """Compute count, sum, extrema, median, and nearest-rank p95."""
    if not lengths:
        raise ValueError("trace must not be empty")
    ordered = sorted(lengths)
    p95_index = (95 * len(ordered) + 99) // 100 - 1
    return {
        "count": len(ordered),
        "sum": sum(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _load_trace_files(
    trace_path: pathlib.Path,
    provenance_path: pathlib.Path,
    *,
    expected_provenance_sha256: str,
) -> tuple[list[int], dict[str, object]]:
    """Load and validate an explicitly located frozen trace pair."""
    try:
        provenance_data = provenance_path.read_bytes()
        trace_data = trace_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read frozen trace files: {error}") from error

    provenance_digest = _sha256(provenance_data)
    if provenance_digest != expected_provenance_sha256:
        raise ValueError(
            "provenance SHA-256 mismatch: expected "
            f"{expected_provenance_sha256}, got {provenance_digest}"
        )
    try:
        provenance = json.loads(
            provenance_data,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid provenance JSON: {error}") from error
    _validate_schema(provenance, _PROVENANCE_SCHEMA, "provenance")
    _require_pinned_provenance(provenance)

    trace_digest = _sha256(trace_data)
    derived = cast(dict[str, object], provenance["derived_trace"])
    if trace_digest != derived["sha256"] or trace_digest != _TRACE_SHA256:
        raise ValueError(
            "trace SHA-256 mismatch: expected "
            f"{derived['sha256']}, got {trace_digest}"
        )
    lengths = _parse_trace(trace_data)
    summary = _summarize(lengths)
    expected_summary = {
        key: derived[key]
        for key in ("count", "sum", "min", "median", "p95", "max")
    }
    if summary != expected_summary:
        raise ValueError(
            f"trace statistics mismatch: expected {expected_summary}, "
            f"got {summary}"
        )
    if len(lengths) != _TRACE_COUNT:
        raise ValueError(
            f"trace count mismatch: expected {_TRACE_COUNT}, got {len(lengths)}"
        )
    return lengths, provenance


def load_real_trace(name: str) -> tuple[list[int], dict[str, object]]:
    """Load a frozen real row-length trace from committed files only.

    Args:
        name: Exact frozen trace name. The only supported value is
            ``soc-epinions1-outdegree-v1``.

    Returns:
        A fresh list of row lengths and a fresh provenance dictionary.

    Raises:
        ValueError: If the name is unsupported or any committed byte, schema,
            source pin, encoding, count, or statistic is invalid.
    """
    if name != _TRACE_NAME:
        raise ValueError(f"unknown real trace {name!r}")
    trace_directory = pathlib.Path(__file__).with_name("traces")
    return _load_trace_files(
        trace_directory / _TRACE_FILENAME,
        trace_directory / _PROVENANCE_FILENAME,
        expected_provenance_sha256=_PROVENANCE_SHA256,
    )
