"""Pure-Python CLI, boundary, error-guard, and mutation suite for tools/shader_census.py."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "shader_census.py"
sys.path.insert(0, str(ROOT / "tools"))

import shader_census as sc  # noqa: E402


def _make_dxbc(digest_byte: int, word: int, stage: int = 0) -> bytes:
    program = struct.pack("<II4sIII", (stage << 16) | 0x60, 7, b"DXIL", 0x100, 16, 4) + struct.pack("<I", word)
    return (
        b"DXBC"
        + bytes([digest_byte]) * 16
        + struct.pack("<4I", 1, 44 + len(program), 1, 36)
        + b"DXIL"
        + struct.pack("<I", len(program))
        + program
    )


def _make_record(blob: bytes, name: str = "owned_shader") -> bytes:
    encoder = zlib.compressobj(wbits=-15)
    payload = encoder.compress(blob) + encoder.flush()
    return (
        name.encode("ascii")
        + b"\0\0"
        + struct.pack("<4I", len(payload), 0, len(payload), len(blob))
        + payload
    )


def _make_archive() -> bytes:
    a = _make_dxbc(1, 17, stage=0)
    b = _make_dxbc(1, 18, stage=0)
    c = _make_dxbc(2, 19, stage=5)
    return _make_record(a) + _make_record(a) + _make_record(b) + _make_record(c)


def _make_ps9_cache(entries: list[tuple[int, int]] | None = None) -> tuple[bytes, int]:
    entries = entries if entries is not None else [(0, 16), (16, 16), (0, 16), (32, 32)]
    index_start = 128 + 0x70 + 2 * 0x18 + 3 * 8
    code_base = index_start + len(entries) * 8
    data = bytearray(code_base + 96)
    data[:4] = b"PS\x00\x09"
    struct.pack_into("<4I", data, 80, 1, 2, 3, len(entries))
    for i, (offset, size) in enumerate(entries):
        struct.pack_into("<2I", data, index_start + i * 8, offset, size)
    return bytes(data), code_base


class ShaderCensusCliTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with redirect_stdout(out), redirect_stderr(err):
            try:
                ret = sc.main(argv)
                code = 0 if ret is None else ret
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
        return code, out.getvalue(), err.getvalue()

    def _assert_main_fails(self, argv: list[str], *, expected_code: int, fragment: str | tuple[str, ...]) -> str:
        code, stdout, stderr = self._run_main(argv)
        self.assertEqual(code, expected_code, f"argv={argv!r} stderr={stderr!r}")
        self.assertEqual(stdout, "")
        self.assertNotIn("Traceback", stderr)
        fragments = (fragment,) if isinstance(fragment, str) else fragment
        self.assertTrue(
            any(f in stderr for f in fragments),
            f"none of {fragments!r} found in {stderr!r}",
        )
        return stderr

    def test_main_argv_forwarding_and_json_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "owned.archive"
            cache = root / "decoded.cache"
            archive.write_bytes(_make_archive())
            cache_bytes, base = _make_ps9_cache()
            cache.write_bytes(cache_bytes)

            # Mutating sys.argv ensures main(argv) forwards its explicit argv parameter.
            with mock.patch.object(sys, "argv", ["shader_census.py", "--bogus-global-flag"]):
                for base_arg in (str(base), hex(base)):
                    out, err = io.StringIO(), io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        rc = sc.main([str(archive), "--cache-blob", str(cache), "--code-base", base_arg])
                    self.assertEqual(rc, 0)
                    self.assertEqual(err.getvalue(), "")
                    doc = json.loads(out.getvalue())
                    self.assertEqual(
                        set(doc),
                        {
                            "archive_records",
                            "dxbc_header_digests",
                            "classification_cache",
                            "cache_index",
                            "candidate_rejections",
                        },
                    )
                    self.assertEqual(doc["archive_records"]["count"], 4)
                    self.assertEqual(doc["archive_records"]["stages"], {"pixel": 3, "compute": 1})
                    self.assertTrue(doc["archive_records"]["coverage_denominator"])
                    self.assertEqual(doc["dxbc_header_digests"]["count"], 2)
                    self.assertFalse(doc["dxbc_header_digests"]["integrity_verified"])
                    self.assertEqual(doc["classification_cache"]["count"], 3)
                    self.assertEqual(doc["cache_index"]["count"], 4)
                    self.assertEqual(doc["cache_index"]["distinct_spans"], 3)
                    self.assertEqual(doc["cache_index"]["aliased_entries"], 1)
                    self.assertEqual(doc["cache_index"]["code_base"], base)
                    self.assertEqual(doc["cache_index"]["index_offset"], 312)
                    self.assertEqual(doc["cache_index"]["program_validation"], "not performed")

            # Archive-only subprocess CLI check.
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(archive)],
                capture_output=True,
                text=True,
                check=True,
            )
            sub_doc = json.loads(proc.stdout)
            self.assertEqual(sub_doc["archive_records"]["count"], 4)
            self.assertIsNone(sub_doc["cache_index"])

    def test_zero_byte_archive_and_cache_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty_archive = root / "empty.archive"
            empty_cache = root / "empty.cache"
            empty_archive.write_bytes(b"")
            empty_cache.write_bytes(b"")

            # 0-byte archive succeeds cleanly without mmap ValueError.
            code, out, err = self._run_main([str(empty_archive)])
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            doc = json.loads(out)
            self.assertEqual(doc["archive_records"]["count"], 0)
            self.assertIsNone(doc["cache_index"])

            # 0-byte --cache-blob exits 1 with 'Shader census:' (never mmap ValueError traceback).
            stderr = self._assert_main_fails(
                [str(empty_archive), "--cache-blob", str(empty_cache), "--code-base", "128"],
                expected_code=1,
                fragment="Shader census:",
            )
            self.assertIn("128-byte header", stderr)

    def test_directory_and_missing_path_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "owned.archive"
            archive.write_bytes(_make_archive())
            subdir = root / "a_dir"
            subdir.mkdir()

            # Directory passed as archive -> exit 1 with 'Shader census:' and 'directory'.
            err_arch = self._assert_main_fails([str(subdir)], expected_code=1, fragment="Shader census:")
            self.assertIn("directory", err_arch.lower())

            # Directory passed as --cache-blob -> exit 1 with 'Shader census:' and 'directory'.
            err_blob = self._assert_main_fails(
                [str(archive), "--cache-blob", str(subdir), "--code-base", "128"],
                expected_code=1,
                fragment="Shader census:",
            )
            self.assertIn("directory", err_blob.lower())

            # Missing archive and missing --cache-blob -> exit 1 with 'Shader census:'.
            self._assert_main_fails([str(root / "missing.archive")], expected_code=1, fragment="Shader census:")
            self._assert_main_fails(
                [str(archive), "--cache-blob", str(root / "missing.cache"), "--code-base", "128"],
                expected_code=1,
                fragment="Shader census:",
            )

    def test_mismatched_and_negative_code_base_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "owned.archive"
            cache = root / "decoded.cache"
            archive.write_bytes(_make_archive())
            cache_bytes, base = _make_ps9_cache()
            cache.write_bytes(cache_bytes)

            # Mismatched --cache-blob / --code-base -> argparse exit 2.
            self._assert_main_fails([str(archive), "--cache-blob", str(cache)], expected_code=2, fragment="supplied together")
            self._assert_main_fails([str(archive), "--code-base", hex(base)], expected_code=2, fragment="supplied together")

            # Negative --code-base (decimal and hex) -> parser.error exit 2.
            for args_extra in (["--code-base", "-1"], ["--code-base=-1"], ["--code-base=-0x10"]):
                self._assert_main_fails(
                    [str(archive), "--cache-blob", str(cache), *args_extra],
                    expected_code=2,
                    fragment="non-negative",
                )
            self._assert_main_fails(
                [str(archive), "--cache-blob", str(cache), "--code-base", "-0x10"],
                expected_code=2,
                fragment=("expected one argument", "non-negative"),
            )

            # Non-integer --code-base -> exit 2.
            self._assert_main_fails(
                [str(archive), "--cache-blob", str(cache), "--code-base", "not_an_int"],
                expected_code=2,
                fragment="invalid",
            )

    def test_malformed_truncated_inputs_and_struct_zlib_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "owned.archive"
            cache = root / "decoded.cache"
            archive.write_bytes(_make_archive())

            # Truncated PS/9 cache (<128 bytes) -> exit 1 with 'Shader census:'.
            cache.write_bytes(b"PS\x00\x09" + b"\x00" * 123)
            self._assert_main_fails(
                [str(archive), "--cache-blob", str(cache), "--code-base", "128"],
                expected_code=1,
                fragment="Shader census:",
            )

            # Out-of-bounds code_base or overlapping allocations -> exit 1 with 'Shader census:'.
            bad_cache, bad_base = _make_ps9_cache([(0, 32), (16, 16)])
            cache.write_bytes(bad_cache)
            self._assert_main_fails(
                [str(archive), "--cache-blob", str(cache), "--code-base", str(bad_base)],
                expected_code=1,
                fragment="Shader census:",
            )
            good_cache, good_base = _make_ps9_cache()
            cache.write_bytes(good_cache)
            self._assert_main_fails(
                [str(archive), "--cache-blob", str(cache), "--code-base", str(len(good_cache) + 1)],
                expected_code=1,
                fragment="Shader census:",
            )

            # Corrupt deflate payload and truncated span in archive -> cleanly counted in candidate_rejections.
            corrupt_deflate = b"bad_rec\0\0" + struct.pack("<4I", 20, 0, 20, 64) + b"\xff" * 20
            truncated_span = b"trunc_rec\0\0" + struct.pack("<4I", 64, 0, 64, 64) + b"\x00" * 8
            archive.write_bytes(_make_archive() + corrupt_deflate + truncated_span)
            code, out, err = self._run_main([str(archive)])
            self.assertEqual(code, 0, err)
            doc = json.loads(out)
            self.assertEqual(doc["archive_records"]["count"], 4)
            self.assertEqual(doc["candidate_rejections"]["deflate_error"], 1)
            self.assertEqual(doc["candidate_rejections"]["truncated_span"], 1)

            # Direct struct.error and zlib.error propagation -> guarded to exit 1 with 'Shader census:'.
            archive.write_bytes(_make_archive())
            for err_obj in (struct.error("truncated header"), zlib.error("corrupt deflate stream")):
                with mock.patch.object(sc, "scene_records", side_effect=err_obj):
                    self._assert_main_fails(
                        [str(archive)],
                        expected_code=1,
                        fragment=f"Shader census: {err_obj}",
                    )
                with mock.patch.object(sc, "cache_index", side_effect=err_obj):
                    self._assert_main_fails(
                        [str(archive), "--cache-blob", str(cache), "--code-base", str(good_base)],
                        expected_code=1,
                        fragment=f"Shader census: {err_obj}",
                    )

            # Subprocess end-to-end negative check verifying zero raw traceback on stderr.
            sub_fail = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(archive), "--cache-blob", str(cache), "--code-base=-4"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(sub_fail.returncode, 2)
            self.assertNotIn("Traceback", sub_fail.stderr)
            self.assertIn("non-negative", sub_fail.stderr)


if __name__ == "__main__":
    unittest.main()
