"""Report shader populations separately; do not classify or execute programs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import mmap
from pathlib import Path
import struct
import zlib

from resource_inventory import scene_records


def cache_index(data, code_base):
    """Count the decoded PS/9 cache index profile reported in issue #594.

    The code-area base must be established independently. Spans are allocations,
    not instruction lengths. Exact aliases count as entries but share a span.
    """
    if len(data) < 128 or data[:4] != b"PS\x00\x09":
        raise ValueError("Expected a decoded PS/9 cache with a 128-byte header")
    if type(code_base) is not int or not 128 <= code_base <= len(data):
        raise ValueError("Code-area base lies outside the decoded cache")
    first, second, third, count = struct.unpack_from("<4I", data, 80)
    start = 128 + first * 0x70 + second * 0x18 + third * 8
    end = start + count * 8
    if end > code_base or count > 1 << 20:
        raise ValueError("Cache tables exceed the code-area base or index entry limit")
    spans = set()
    for index in range(count):
        offset, size = struct.unpack_from("<2I", data, start + index * 8)
        if offset % 16 or size == 0 or offset + size > len(data) - code_base:
            raise ValueError(f"Invalid cache allocation at index {index}")
        spans.add((offset, size))
    ordered = sorted(spans)
    if any(a + size > b for (a, size), (b, _) in zip(ordered, ordered[1:])):
        raise ValueError("Distinct cache allocations overlap")
    return dict(count=count, distinct_spans=len(spans), aliased_entries=count - len(spans),
                source="decoded PS/9 cache index", key="(code_offset, allocation_size)",
                code_base=code_base, index_offset=start,
                program_validation="not performed")


def census(archive, *, cache_blob=None, code_base=None):
    if (cache_blob is None) != (code_base is None):
        raise ValueError("Supply both decoded cache bytes and their code-area base")
    header_digests, container_hashes = set(), set()

    def observe(stage, blob, sections):
        header_digests.add(blob[4:20])
        container_hashes.add(hashlib.sha256(blob).digest())

    # Share the profile audit's existing record acceptance and scan semantics.
    records = scene_records(archive, observe=observe)
    return dict(
        archive_records=dict(count=records["records"], source="scene_records accepted occurrences",
                             key="accepted record occurrence", stages=records["stages"],
                             coverage_denominator=True),
        dxbc_header_digests=dict(count=len(header_digests), source="accepted archive containers",
                                 key="DXBC header bytes [4:20]", integrity_verified=False),
        classification_cache=dict(count=len(container_hashes), source="accepted archive containers",
                                  key="SHA-256 of complete decoded DXBC container"),
        cache_index=None if cache_blob is None else cache_index(cache_blob, code_base),
        candidate_rejections=records["candidate_rejections"],
    )


def _map_input(stack: ExitStack, path: Path):
    if path.is_dir():
        raise ValueError(f"Expected a file, not a directory: {path}")
    stream = stack.enter_context(path.open("rb"))
    if stream.seek(0, 2) == 0:
        return b""
    return stack.enter_context(mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--cache-blob", type=Path, help="Already decoded PS/9 cache from the same build")
    parser.add_argument("--code-base", type=lambda value: int(value, 0),
                        help="Independently established code-area base in that cache (decimal or hex)")
    args = parser.parse_args(argv)
    if (args.cache_blob is None) != (args.code_base is None):
        parser.error("--cache-blob and --code-base must be supplied together")
    if args.code_base is not None and args.code_base < 0:
        parser.error("--code-base must be non-negative")
    try:
        with ExitStack() as stack:
            archive = _map_input(stack, args.archive)
            blob = _map_input(stack, args.cache_blob) if args.cache_blob is not None else None
            result = census(archive, cache_blob=blob, code_base=args.code_base)
    except (OSError, ValueError, struct.error, zlib.error) as error:
        parser.exit(1, f"Shader census: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
