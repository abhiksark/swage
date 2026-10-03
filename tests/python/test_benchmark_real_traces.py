# tests/python/test_benchmark_real_traces.py
"""Tests for the frozen real benchmark row-length trace."""

import hashlib
import io
import json
import pathlib
import tarfile

import pytest

from benchmarks.prepare_soc_epinions1_trace import (
    _extract_member,
    _parse_matrix_market,
    _require_blob,
    _select_row_lengths,
    _select_vertices,
)
from benchmarks.real_traces import (
    _PROVENANCE_SHA256,
    _load_trace_files,
    _parse_trace,
    load_real_trace,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TRACE_PATH = (
    _ROOT / "benchmarks/traces/soc-Epinions1-outdegree-sample-v1.txt"
)
_PROVENANCE_PATH = _TRACE_PATH.with_suffix(".provenance.json")
_BANNER = b"%%MatrixMarket matrix coordinate pattern general\n"


def test_committed_trace_matches_frozen_provenance():
    """Load the exact offline trace and preserve its declared statistics."""
    lengths, provenance = load_real_trace("soc-epinions1-outdegree-v1")
    derived = provenance["derived_trace"]

    assert derived == {
        "count": 32_768,
        "encoding": "one unsigned decimal integer plus LF per row",
        "filename": "soc-Epinions1-outdegree-sample-v1.txt",
        "generator_revision": (
            "benchmarks/prepare_soc_epinions1_trace.py:v1"
        ),
        "max": 1669,
        "median": 1.0,
        "min": 0,
        "p95": 30,
        "redistributed_fields": "outgoing row lengths only",
        "sha256": (
            "97138fdc92d297aa026480edda222827f9c1e84cc2d001d66a52d18e3201ee48"
        ),
        "sum": 222_306,
    }
    assert len(lengths) == 32_768
    assert sum(lengths) == 222_306
    assert min(lengths) == 0
    assert max(lengths) == 1669
    assert max(lengths) <= 4096


def test_loader_returns_fresh_trace_and_provenance_objects():
    """Prevent callers from mutating data returned to later callers."""
    first_lengths, first_provenance = load_real_trace(
        "soc-epinions1-outdegree-v1"
    )
    second_lengths, second_provenance = load_real_trace(
        "soc-epinions1-outdegree-v1"
    )

    assert first_lengths == second_lengths
    assert first_provenance == second_provenance
    assert first_lengths is not second_lengths
    assert first_provenance is not second_provenance


def test_provenance_preserves_license_attribution_and_citations():
    """Keep CC BY attribution, modification notice, and source DOIs explicit."""
    _, provenance = load_real_trace("soc-epinions1-outdegree-v1")
    license_record = provenance["license"]

    assert license_record["identifier"] == "CC-BY-4.0"
    assert license_record["url"] == (
        "https://creativecommons.org/licenses/by/4.0/"
    )
    assert "SuiteSparse Matrix Collection" in license_record["attribution"]
    assert "No vertex IDs or edges are redistributed." in (
        license_record["modification_notice"]
    )
    assert {citation["doi"] for citation in provenance["citations"]} == {
        "10.1007/978-3-540-39718-2_23",
        "10.1145/2049662.2049663",
    }


def test_loader_rejects_unsupported_name():
    """Fail closed instead of substituting a similarly named trace."""
    with pytest.raises(ValueError, match="unknown real trace"):
        load_real_trace("soc-epinions1")


def test_loader_rejects_changed_trace(tmp_path):
    """Detect a changed committed trace before accepting its values."""
    trace_path = tmp_path / _TRACE_PATH.name
    provenance_path = tmp_path / _PROVENANCE_PATH.name
    trace_data = _TRACE_PATH.read_bytes()
    trace_path.write_bytes(b"9" + trace_data[1:])
    provenance_path.write_bytes(_PROVENANCE_PATH.read_bytes())

    with pytest.raises(ValueError, match="trace SHA-256 mismatch"):
        _load_trace_files(
            trace_path,
            provenance_path,
            expected_provenance_sha256=_PROVENANCE_SHA256,
        )


def test_loader_rejects_changed_provenance(tmp_path):
    """Detect any changed committed provenance byte."""
    trace_path = tmp_path / _TRACE_PATH.name
    provenance_path = tmp_path / _PROVENANCE_PATH.name
    trace_path.write_bytes(_TRACE_PATH.read_bytes())
    provenance_path.write_bytes(_PROVENANCE_PATH.read_bytes() + b" ")

    with pytest.raises(ValueError, match="provenance SHA-256 mismatch"):
        _load_trace_files(
            trace_path,
            provenance_path,
            expected_provenance_sha256=_PROVENANCE_SHA256,
        )


def test_loader_rejects_provenance_schema_mismatch(tmp_path):
    """Reject unknown provenance fields even behind an updated checksum."""
    trace_path = tmp_path / _TRACE_PATH.name
    provenance_path = tmp_path / _PROVENANCE_PATH.name
    provenance = json.loads(_PROVENANCE_PATH.read_text())
    provenance["unexpected"] = True
    provenance_data = json.dumps(provenance).encode()
    trace_path.write_bytes(_TRACE_PATH.read_bytes())
    provenance_path.write_bytes(provenance_data)

    with pytest.raises(ValueError, match="schema mismatch"):
        _load_trace_files(
            trace_path,
            provenance_path,
            expected_provenance_sha256=(
                hashlib.sha256(provenance_data).hexdigest()
            ),
        )


def test_loader_rejects_changed_provenance_statistics(tmp_path):
    """Recompute and compare every declared derived-trace statistic."""
    trace_path = tmp_path / _TRACE_PATH.name
    provenance_path = tmp_path / _PROVENANCE_PATH.name
    provenance = json.loads(_PROVENANCE_PATH.read_text())
    provenance["derived_trace"]["p95"] += 1
    provenance_data = json.dumps(provenance).encode()
    trace_path.write_bytes(_TRACE_PATH.read_bytes())
    provenance_path.write_bytes(provenance_data)

    with pytest.raises(ValueError, match="trace statistics mismatch"):
        _load_trace_files(
            trace_path,
            provenance_path,
            expected_provenance_sha256=(
                hashlib.sha256(provenance_data).hexdigest()
            ),
        )


def test_trace_parser_rejects_noncanonical_and_oversized_lengths():
    """Reject encodings and row lengths outside the benchmark contract."""
    with pytest.raises(ValueError, match="canonical"):
        _parse_trace(b"01\n")
    with pytest.raises(ValueError, match="exceeds maximum"):
        _parse_trace(b"4097\n")
    with pytest.raises(ValueError, match="end with LF"):
        _parse_trace(b"1")


def test_matrix_parser_retains_sink_only_zero_rows():
    """Form the vertex universe from both endpoints and count source rows."""
    matrix_data = _BANNER + b"4 4 3\n1 2\n1 3\n4 2\n"

    vertices, row_counts = _parse_matrix_market(
        matrix_data,
        expected_rows=4,
        expected_columns=4,
        expected_entries=3,
    )

    assert vertices == {1, 2, 3, 4}
    assert row_counts == {1: 2, 4: 1}
    observed = sorted(
        row_counts.get(vertex, 0) for vertex in vertices
    )
    assert observed == [0, 0, 1, 2]


@pytest.mark.parametrize(
    "matrix_data",
    [
        (
            b"%%MatrixMarket matrix coordinate integer general\n"
            b"2 2 1\n1 2\n"
        ),
        _BANNER + b"3 2 1\n1 2\n",
        _BANNER + b"2 2 2\n1 2\n",
    ],
)
def test_matrix_parser_rejects_banner_shape_and_count(matrix_data):
    """Fail closed on every fixed Matrix Market schema assumption."""
    with pytest.raises(ValueError):
        _parse_matrix_market(
            matrix_data,
            expected_rows=2,
            expected_columns=2,
            expected_entries=1,
        )


def test_matrix_parser_rejects_duplicate_coordinates():
    """Reject repeated stored coordinates instead of counting them twice."""
    with pytest.raises(ValueError, match="duplicates"):
        _parse_matrix_market(
            _BANNER + b"2 2 2\n1 2\n1 2\n",
            expected_rows=2,
            expected_columns=2,
            expected_entries=2,
        )


def test_hash_selection_is_domain_separated_and_order_independent():
    """Rank full SHA-256 digests with uint64 big-endian vertex IDs."""
    domain = b"test-domain\0"
    vertices = [9, 2, 7, 4]
    expected = sorted(
        vertices,
        key=lambda row_id: (
            hashlib.sha256(domain + row_id.to_bytes(8, "big")).digest(),
            row_id,
        ),
    )[:3]

    assert _select_vertices(
        vertices, sample_count=3, domain_separator=domain
    ) == expected
    assert _select_vertices(
        list(reversed(vertices)),
        sample_count=3,
        domain_separator=domain,
    ) == expected
    expected_lengths = {
        2: 20,
        4: 40,
        7: 70,
        9: 90,
    }
    assert _select_row_lengths(
        vertices,
        expected_lengths,
        sample_count=3,
        domain_separator=domain,
    ) == [expected_lengths[row_id] for row_id in expected]


def test_archive_and_member_hash_mismatches_fail_closed():
    """Reject changed archives and changed selected member bytes."""
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        _require_blob(
            b"archive",
            expected_size=7,
            expected_sha256="0" * 64,
            label="archive",
        )

    archive_stream = io.BytesIO()
    member_data = b"member contents"
    with tarfile.open(fileobj=archive_stream, mode="w:gz") as archive:
        member = tarfile.TarInfo("graph/matrix.mtx")
        member.size = len(member_data)
        archive.addfile(member, io.BytesIO(member_data))

    with pytest.raises(ValueError, match="archive member.*SHA-256 mismatch"):
        _extract_member(
            archive_stream.getvalue(),
            member_path="graph/matrix.mtx",
            expected_size=len(member_data),
            expected_sha256="0" * 64,
        )
