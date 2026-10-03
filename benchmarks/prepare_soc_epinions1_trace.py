# benchmarks/prepare_soc_epinions1_trace.py
"""Regenerate the frozen soc-Epinions1 outgoing-row-length trace."""

import argparse
import hashlib
import io
import json
import os
import pathlib
import statistics
import tarfile
import tempfile
import urllib.request
from collections.abc import Collection, Mapping

_DATASET_NAME = "soc-Epinions1"
_ARCHIVE_URL = "https://sparse.tamu.edu/MM/SNAP/soc-Epinions1.tar.gz"
_ARCHIVE_FILENAME = "soc-Epinions1.tar.gz"
_ARCHIVE_SIZE = 1_552_639
_ARCHIVE_SHA256 = (
    "ed75c7f919799a8571aab592e2b89d8e54bc734530b4b8c5ab2071386ecd698e"
)
_ARCHIVE_MD5 = "aeea34c0fe3b48be8ae96de54867f418"
_MEMBER_PATH = "soc-Epinions1/soc-Epinions1.mtx"
_MEMBER_SIZE = 5_158_059
_MEMBER_SHA256 = (
    "f24ae6890fc53c9e9b153812c4d22cd2e87a3642a1af12eeefd06dae0888f627"
)
_MATRIX_BANNER = "%%MatrixMarket matrix coordinate pattern general"
_MATRIX_ROWS = 75_888
_MATRIX_COLUMNS = 75_888
_MATRIX_ENTRIES = 508_837
_VERTEX_COUNT = 75_879
_SAMPLE_COUNT = 32_768
_DOMAIN_SEPARATOR = b"swage/soc-Epinions1/row-sample/v1\0"


def _sha256(data: bytes) -> str:
    """Return the lowercase SHA-256 hexadecimal digest for data."""
    return hashlib.sha256(data).hexdigest()


def _require_blob(
    data: bytes, *, expected_size: int, expected_sha256: str, label: str
) -> None:
    """Require the exact size and SHA-256 of a pinned input blob."""
    if len(data) != expected_size:
        raise ValueError(
            f"{label} size mismatch: expected {expected_size}, got {len(data)}"
        )
    digest = _sha256(data)
    if digest != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {digest}"
        )


def _extract_member(
    archive_data: bytes,
    *,
    member_path: str,
    expected_size: int,
    expected_sha256: str,
) -> bytes:
    """Extract and verify exactly one regular member from a tar archive."""
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as tar:
            matches = [member for member in tar if member.name == member_path]
            if len(matches) != 1:
                raise ValueError(
                    f"archive must contain exactly one {member_path!r} member"
                )
            member = matches[0]
            if not member.isfile():
                raise ValueError(
                    f"archive member {member_path!r} is not a file"
                )
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read archive member {member_path!r}")
            member_data = stream.read()
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f"invalid tar archive: {error}") from error

    _require_blob(
        member_data,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        label=f"archive member {member_path!r}",
    )
    return member_data


def _decimal(token: bytes, *, label: str) -> int:
    """Parse one canonical unsigned decimal token."""
    if not token or not token.isdigit():
        raise ValueError(f"{label} must be an unsigned decimal integer")
    if len(token) > 1 and token.startswith(b"0"):
        raise ValueError(f"{label} is not canonically encoded")
    return int(token)


def _parse_matrix_market(
    matrix_data: bytes,
    *,
    expected_rows: int,
    expected_columns: int,
    expected_entries: int,
) -> tuple[set[int], dict[int, int]]:
    """Parse a general pattern-coordinate matrix and count outgoing rows."""
    lines = matrix_data.splitlines()
    if not lines or lines[0] != _MATRIX_BANNER.encode("ascii"):
        raise ValueError(
            f"Matrix Market banner must be exactly {_MATRIX_BANNER!r}"
        )

    dimensions_index = 1
    while (
        dimensions_index < len(lines)
        and lines[dimensions_index].startswith(b"%")
    ):
        dimensions_index += 1
    if dimensions_index == len(lines):
        raise ValueError("Matrix Market dimensions are missing")

    dimensions = lines[dimensions_index].split()
    if len(dimensions) != 3:
        raise ValueError("Matrix Market dimensions must contain three integers")
    rows, columns, entries = (
        _decimal(token, label="Matrix Market dimension")
        for token in dimensions
    )
    actual_shape = (rows, columns, entries)
    expected_shape = (
        expected_rows,
        expected_columns,
        expected_entries,
    )
    if actual_shape != expected_shape:
        raise ValueError(
            "Matrix Market schema mismatch: expected "
            f"{expected_shape}, got {actual_shape}"
        )

    records = lines[dimensions_index + 1 :]
    if len(records) != expected_entries:
        raise ValueError(
            "Matrix Market entry-count mismatch: expected "
            f"{expected_entries}, got {len(records)}"
        )

    vertices: set[int] = set()
    row_counts: dict[int, int] = {}
    coordinates: set[tuple[int, int]] = set()
    for record_number, record in enumerate(records, start=1):
        fields = record.split()
        if len(fields) != 2:
            raise ValueError(
                f"Matrix Market entry {record_number} must have two indices"
            )
        row = _decimal(fields[0], label=f"entry {record_number} row")
        column = _decimal(fields[1], label=f"entry {record_number} column")
        if not 1 <= row <= expected_rows:
            raise ValueError(f"entry {record_number} row is out of bounds")
        if not 1 <= column <= expected_columns:
            raise ValueError(f"entry {record_number} column is out of bounds")
        coordinate = (row, column)
        if coordinate in coordinates:
            raise ValueError(
                f"Matrix Market entry {record_number} duplicates {coordinate}"
            )
        coordinates.add(coordinate)
        vertices.add(row)
        vertices.add(column)
        row_counts[row] = row_counts.get(row, 0) + 1

    return vertices, row_counts


def _select_vertices(
    vertices: Collection[int],
    *,
    sample_count: int,
    domain_separator: bytes,
) -> list[int]:
    """Select vertex IDs by the versioned domain-separated hash ranking."""
    if not 0 <= sample_count <= len(vertices):
        raise ValueError("sample count must be within the vertex universe")

    def rank(row_id: int) -> tuple[bytes, int]:
        if not 0 <= row_id < 1 << 64:
            raise ValueError("row IDs must fit uint64")
        payload = domain_separator + row_id.to_bytes(8, "big")
        return hashlib.sha256(payload).digest(), row_id

    return sorted(vertices, key=rank)[:sample_count]


def _select_row_lengths(
    vertices: Collection[int],
    row_counts: Mapping[int, int],
    *,
    sample_count: int,
    domain_separator: bytes,
) -> list[int]:
    """Return outgoing row lengths in deterministic hash-ranked order."""
    selected = _select_vertices(
        vertices,
        sample_count=sample_count,
        domain_separator=domain_separator,
    )
    return [row_counts.get(row_id, 0) for row_id in selected]


def _summarize(lengths: list[int]) -> dict[str, int | float]:
    """Compute the frozen nearest-rank summary for a nonempty trace."""
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


def _encode_trace(lengths: list[int]) -> bytes:
    """Encode unsigned lengths as one canonical decimal integer per LF."""
    invalid = any(
        type(value) is not int or value < 0 for value in lengths
    )
    if not lengths or invalid:
        raise ValueError("trace lengths must be nonnegative integers")
    return b"".join(f"{value}\n".encode("ascii") for value in lengths)


def _provenance(trace_filename: str, trace_data: bytes) -> dict[str, object]:
    """Build the machine-readable provenance record for a derived trace."""
    lengths = [int(line) for line in trace_data.splitlines()]
    return {
        "schema_version": 1,
        "source": {
            "collection": "SuiteSparse Matrix Collection",
            "group": "SNAP",
            "dataset_name": _DATASET_NAME,
            "matrix_id": 2284,
            "landing_url": "https://sparse.tamu.edu/SNAP/soc-Epinions1",
            "download_url": _ARCHIVE_URL,
            "retrieved_at": "2026-09-04",
            "archive": {
                "filename": _ARCHIVE_FILENAME,
                "size_bytes": _ARCHIVE_SIZE,
                "sha256": _ARCHIVE_SHA256,
                "md5": _ARCHIVE_MD5,
                "etag": _ARCHIVE_MD5,
                "last_modified": "2020-05-01 19:19:17 GMT",
            },
            "member": {
                "path": _MEMBER_PATH,
                "size_bytes": _MEMBER_SIZE,
                "sha256": _MEMBER_SHA256,
            },
        },
        "matrix_market": {
            "banner": _MATRIX_BANNER,
            "object": "matrix",
            "format": "coordinate",
            "field": "pattern",
            "symmetry": "general",
            "index_base": 1,
            "rows": _MATRIX_ROWS,
            "columns": _MATRIX_COLUMNS,
            "stored_entries": _MATRIX_ENTRIES,
        },
        "graph": {
            "advertised_vertex_count": _VERTEX_COUNT,
            "advertised_edge_count": _MATRIX_ENTRIES,
            "actual_vertex_count": _VERTEX_COUNT,
            "vertex_universe_rule": (
                "sorted union of one-based row and column IDs occurring in "
                "the coordinate records"
            ),
            "orientation": "matrix row = source; row nnz = outgoing degree",
            "zero_row_policy": "retain sink-only vertices with length zero",
            "duplicate_policy": "reject duplicate stored coordinates",
            "self_loop_policy": "retain stored self-loops",
            "symmetry_expansion_policy": (
                "none; general matrices are not symmetrized"
            ),
        },
        "extraction": {
            "version": "soc-epinions1-outdegree-v1",
            "domain_separator_display": (
                "swage/soc-Epinions1/row-sample/v1\\0"
            ),
            "domain_separator_hex": _DOMAIN_SEPARATOR.hex(),
            "row_id_encoding": "uint64_be",
            "digest": "SHA-256",
            "ranking": (
                "ascending lexicographic (full 32-byte digest, row_id)"
            ),
            "sample_count": _SAMPLE_COUNT,
            "replacement": False,
            "output_order": "hash-ranking order",
        },
        "derived_trace": {
            "filename": trace_filename,
            "encoding": "one unsigned decimal integer plus LF per row",
            "sha256": _sha256(trace_data),
            **_summarize(lengths),
            "generator_revision": (
                "benchmarks/prepare_soc_epinions1_trace.py:v1"
            ),
            "redistributed_fields": "outgoing row lengths only",
        },
        "license": {
            "identifier": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "collection_policy_url": "https://sparse.tamu.edu/about",
            "attribution": (
                "SuiteSparse Matrix Collection, SNAP/soc-Epinions1; "
                "original graph by Matthew Richardson, Rakesh Agrawal, "
                "and Pedro Domingos."
            ),
            "modification_notice": (
                "Modified by the Swage project: counted outgoing stored "
                "coordinates over the actual endpoint-ID union, retained "
                "sink-only zero rows, and selected 32,768 rows with the "
                "documented hash ranking. No vertex IDs or edges are "
                "redistributed."
            ),
        },
        "citations": [
            {
                "authors": (
                    "Matthew Richardson; Rakesh Agrawal; Pedro Domingos"
                ),
                "title": "Trust Management for the Semantic Web",
                "doi": "10.1007/978-3-540-39718-2_23",
                "url": "https://doi.org/10.1007/978-3-540-39718-2_23",
            },
            {
                "authors": "Timothy A. Davis; Yifan Hu",
                "title": (
                    "The University of Florida Sparse Matrix Collection"
                ),
                "doi": "10.1145/2049662.2049663",
                "url": "https://doi.org/10.1145/2049662.2049663",
            },
        ],
    }


def _regenerate(
    archive_data: bytes, trace_filename: str
) -> tuple[bytes, bytes]:
    """Regenerate canonical trace and provenance bytes from a pinned archive."""
    _require_blob(
        archive_data,
        expected_size=_ARCHIVE_SIZE,
        expected_sha256=_ARCHIVE_SHA256,
        label="archive",
    )
    matrix_data = _extract_member(
        archive_data,
        member_path=_MEMBER_PATH,
        expected_size=_MEMBER_SIZE,
        expected_sha256=_MEMBER_SHA256,
    )
    vertices, row_counts = _parse_matrix_market(
        matrix_data,
        expected_rows=_MATRIX_ROWS,
        expected_columns=_MATRIX_COLUMNS,
        expected_entries=_MATRIX_ENTRIES,
    )
    if len(vertices) != _VERTEX_COUNT:
        raise ValueError(
            "actual vertex-count mismatch: expected "
            f"{_VERTEX_COUNT}, got {len(vertices)}"
        )
    lengths = _select_row_lengths(
        vertices,
        row_counts,
        sample_count=_SAMPLE_COUNT,
        domain_separator=_DOMAIN_SEPARATOR,
    )
    trace_data = _encode_trace(lengths)
    provenance = _provenance(trace_filename, trace_data)
    provenance_data = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return trace_data, provenance_data


def _write_file(path: pathlib.Path, data: bytes, *, overwrite: bool) -> None:
    """Atomically write one output, rejecting an existing path by default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {path}"
        )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        stream.write(data)
    try:
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _download_archive() -> bytes:
    """Download the official archive; callers still verify its pinned digest."""
    request = urllib.request.Request(
        _ARCHIVE_URL,
        headers={"User-Agent": "swage-trace-regenerator/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    """Regenerate the trace and provenance at explicit output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=pathlib.Path,
        help="local official archive; omit to download the pinned URL",
    )
    parser.add_argument("--trace-output", required=True, type=pathlib.Path)
    parser.add_argument("--provenance-output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output files",
    )
    args = parser.parse_args()

    outputs = (args.trace_output, args.provenance_output)
    if args.trace_output == args.provenance_output:
        parser.error("trace and provenance outputs must be different paths")
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            parser.error(
                "refusing to overwrite existing output: " + ", ".join(existing)
            )

    archive_data = (
        args.archive.read_bytes() if args.archive else _download_archive()
    )
    trace_data, provenance_data = _regenerate(
        archive_data, args.trace_output.name
    )
    _write_file(args.trace_output, trace_data, overwrite=args.overwrite)
    _write_file(
        args.provenance_output,
        provenance_data,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
