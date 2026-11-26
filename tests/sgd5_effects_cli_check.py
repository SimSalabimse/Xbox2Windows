"""CLI, path-guard, short-read, error-handling, and mutation suite for tools/sgd5_effects.py."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "sgd5_effects.py"
sys.path.insert(0, str(ROOT / "tools"))

import sgd5_effects  # noqa: E402


def _word(value: int) -> bytes:
    return struct.pack("<I", value)


def _len_name(text: str) -> bytes:
    encoded = text.encode("ascii") + b"\0"
    return bytes([len(encoded)]) + encoded


def _deflate_raw(blob: bytes, mode: int = zlib.Z_FINISH) -> bytes:
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(blob) + compressor.flush(mode)


def _program_record(label: str, kind: int, identity: int, *, pack=_deflate_raw) -> tuple[bytes, str]:
    dxil = struct.pack("<HHI4sIII4sI", 0x60, kind, 8, b"DXIL", 0x100, 16, 8, b"BC\xc0\xde", identity)
    blob = b"DXBC" + bytes(16) + struct.pack("<4I", 1, 44 + len(dxil), 1, 36) + b"DXIL" + _word(len(dxil)) + dxil
    packed = pack(blob)
    size = 12 + len(packed)
    wrapped = struct.pack("<3I", 0xddce7, size, len(blob)) + packed
    reflection = _word(9) + struct.pack("<4H", 0, 0, 0, 9) + b"\0"
    record = _len_name(label) + b"\x20" + _word(size) + wrapped + bytes(152) + reflection
    return record, hashlib.sha256(blob).hexdigest()


def _effect_record(label: str, vs: list[int], ps: list[int]) -> bytes:
    lists = b"".join(_word(len(group)) + b"".join(map(_word, group)) for group in (vs, ps, [], [], [], []))
    strings = b"\0entry\0"
    return _len_name(label) + _word(0) + lists + b"\xff\xff" + _word(len(strings)) + strings


def _build_archive(prefix_len: int = 0) -> tuple[bytes, int, int, int, int, int, str, str]:
    vs0, vs_digest = _program_record("vs_main", 1, 101)
    ps0, ps_digest = _program_record("ps_main", 0, 202)
    arrays = _word(1) + vs0 + _word(1) + ps0 + _word(0) * 4
    first = _effect_record("idle", [], [])
    second = _effect_record("draw", [0], [0])
    header = bytes(prefix_len) + b"SGD5" + bytes(28)
    arrays_start = len(header)
    first_start = arrays_start + len(arrays) + 4
    second_start = first_start + len(first)
    archive = header + arrays + _word(2) + first + second
    return archive, prefix_len, len(archive), arrays_start, second_start, len(archive), vs_digest, ps_digest


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = sgd5_effects.main(argv)
        except SystemExit as exc:
            raw = exc.code
            code = 0 if raw is None else (int(raw) if isinstance(raw, int) else 2)
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class Sgd5EffectsCliPositiveTests(unittest.TestCase):
    def test_cli_zero_and_hex_offsets_exact_json_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, prefix_len, fmt_offset in (
                ("zero_group_start", 0, str),
                ("hex_offsets", 16, hex),
            ):
                with self.subTest(label=label):
                    data, grp_start, grp_end, arrays, rec_start, rec_end, vs_digest, ps_digest = _build_archive(
                        prefix_len
                    )
                    archive = root / f"{label}.bin"
                    archive.write_bytes(data)
                    argv = [
                        str(archive),
                        "draw",
                        "--group",
                        fmt_offset(grp_start),
                        fmt_offset(grp_end),
                        "--arrays",
                        fmt_offset(arrays),
                        "--record",
                        fmt_offset(rec_start),
                        fmt_offset(rec_end),
                    ]
                    expected = {
                        "effect": "draw",
                        "candidates": [
                            {
                                "group": "VS",
                                "index": 0,
                                "name": "vs_main",
                                "dxil_kind": 1,
                                "container_sha256": vs_digest,
                            },
                            {
                                "group": "PS",
                                "index": 0,
                                "name": "ps_main",
                                "dxil_kind": 0,
                                "container_sha256": ps_digest,
                            },
                        ],
                    }

                    code, out, err = _run_main(argv)
                    self.assertIs(type(code), int)
                    self.assertEqual(code, 0, err)
                    self.assertEqual(err, "")
                    doc = json.loads(out)
                    self.assertEqual(set(doc), {"effect", "candidates"})
                    self.assertEqual(doc, expected)
                    for item in doc["candidates"]:
                        self.assertEqual(set(item), {"group", "index", "name", "dxil_kind", "container_sha256"})

                    proc = _run_subprocess(*argv)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertEqual(proc.stderr, "")
                    self.assertEqual(json.loads(proc.stdout), expected)

            self.assertEqual(sorted(p.name for p in root.iterdir()), ["hex_offsets.bin", "zero_group_start.bin"])

    def test_main_sys_argv_and_script_entrypoint(self) -> None:
        with TemporaryDirectory() as tmp:
            data, grp_start, grp_end, arrays, rec_start, rec_end, _, _ = _build_archive(0)
            archive = Path(tmp) / "entry.bin"
            archive.write_bytes(data)
            cli_argv = [
                str(TOOL),
                str(archive),
                "draw",
                "--group",
                str(grp_start),
                str(grp_end),
                "--arrays",
                str(arrays),
                "--record",
                str(rec_start),
                str(rec_end),
            ]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(len(json.loads(out)["candidates"]), 2)

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertIs(type(exited.exception.code), int)
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(len(json.loads(stdout.getvalue())["candidates"]), 2)


class Sgd5EffectsCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int = 1,
        expected_fragment: str,
        check_subprocess: bool = True,
    ) -> None:
        code, out, err = _run_main(argv)
        self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn(expected_fragment.lower(), err.lower())

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(expected_fragment.lower(), proc.stderr.lower())

    def test_py314_safe_argparse_errors_exit_nonzero_without_traceback(self) -> None:
        with TemporaryDirectory() as tmp:
            data, grp_start, grp_end, arrays, rec_start, rec_end, _, _ = _build_archive(0)
            archive = Path(tmp) / "valid.bin"
            archive.write_bytes(data)
            base_argv = [
                str(archive),
                "draw",
                "--group",
                str(grp_start),
                str(grp_end),
                "--arrays",
                str(arrays),
                "--record",
                str(rec_start),
                str(rec_end),
            ]
            for label, argv, frag in (
                ("missing_all_args", [], "usage:"),
                ("missing_required_flags", [str(archive), "draw"], "required"),
                ("non_integer_arrays", [str(archive), "draw", "--group", "0", "1", "--arrays", "bad", "--record", "0", "1"], "invalid"),
                ("unknown_flag", [*base_argv, "--unknown-flag"], "unrecognized"),
            ):
                with self.subTest(case=label):
                    with self.assertRaises(SystemExit) as exited:
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            sgd5_effects.main(argv)
                    self.assertNotEqual(exited.exception.code, 0)
                    self._assert_clean_failure(argv, expected_code=2, expected_fragment=frag)

    def test_negative_offsets_rejected_by_offset_parser_before_opening_archive(self) -> None:
        self.assertEqual(sgd5_effects.offset("0"), 0)
        self.assertEqual(sgd5_effects.offset("0x20"), 32)
        for raw in ("-1", "-0x1", "-16"):
            with self.subTest(raw=raw):
                with self.assertRaises(argparse.ArgumentTypeError):
                    sgd5_effects.offset(raw)

        with TemporaryDirectory() as tmp:
            data, grp_start, grp_end, arrays, rec_start, rec_end, _, _ = _build_archive(0)
            existing = Path(tmp) / "existing.bin"
            existing.write_bytes(data)
            missing = Path(tmp) / "missing.bin"

            for archive in (existing, missing):
                for label, bad_argv in (
                    ("neg_group_start", [str(archive), "draw", "--group", "-1", str(grp_end), "--arrays", str(arrays), "--record", str(rec_start), str(rec_end)]),
                    ("neg_group_end", [str(archive), "draw", "--group", str(grp_start), "-1", "--arrays", str(arrays), "--record", str(rec_start), str(rec_end)]),
                    ("neg_arrays", [str(archive), "draw", "--group", str(grp_start), str(grp_end), "--arrays=-0x1", "--record", str(rec_start), str(rec_end)]),
                    ("neg_record_start", [str(archive), "draw", "--group", str(grp_start), str(grp_end), "--arrays", str(arrays), "--record", "-1", str(rec_end)]),
                    # Older argparse versions treat a separate negative hex token as an option, not as an offset.
                    ("neg_record_end", [str(archive), "draw", "--group", str(grp_start), str(grp_end), "--arrays", str(arrays), "--record", str(rec_start), "-16"]),
                ):
                    with self.subTest(archive=archive.name, case=label):
                        with mock.patch.object(
                            sgd5_effects,
                            "candidates",
                            side_effect=AssertionError("candidates called for negative offset"),
                        ):
                            self._assert_clean_failure(
                                bad_argv,
                                expected_code=2,
                                expected_fragment="non-negative",
                            )

    def test_non_file_and_missing_archive_rejected_upfront_before_candidates(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dir_path = root / "archive_dir"
            dir_path.mkdir()
            missing_path = root / "missing.bin"

            for label, bad_archive in (
                ("directory", str(dir_path)),
                ("empty_cwd_dir", ""),
                ("missing_file", str(missing_path)),
            ):
                with self.subTest(case=label):
                    argv = [
                        bad_archive,
                        "draw",
                        "--group",
                        "0",
                        "64",
                        "--arrays",
                        "32",
                        "--record",
                        "48",
                        "64",
                    ]
                    with mock.patch.object(
                        sgd5_effects,
                        "candidates",
                        side_effect=AssertionError(f"candidates called for invalid archive {label}"),
                    ):
                        self._assert_clean_failure(
                            argv,
                            expected_code=1,
                            expected_fragment="regular file",
                            check_subprocess=bool(bad_archive),
                        )

    def test_cursor_short_and_overlong_reads_and_negative_skip_rejected(self) -> None:
        class _FakeView:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def read(self, offset: int, length: int) -> bytes:
                return self._payload

        for label, payload, request in (
            ("empty_short_read", b"", 4),
            ("partial_short_read", b"abc", 4),
            ("overlong_read", b"abcde", 4),
        ):
            with self.subTest(case=label):
                cursor = sgd5_effects.Cursor(_FakeView(payload), 0, 16)
                with self.assertRaisesRegex(ValueError, "Truncated"):
                    cursor.read(request)

        neg_cursor = sgd5_effects.Cursor(_FakeView(b"abcd"), 4, 16)
        with self.assertRaisesRegex(ValueError, "Negative"):
            neg_cursor.skip(-1)
        self.assertEqual(neg_cursor.pos, 4)

        with TemporaryDirectory() as tmp:
            data, grp_start, grp_end, arrays, rec_start, rec_end, _, _ = _build_archive(0)
            archive = Path(tmp) / "short_read.bin"
            archive.write_bytes(data)
            argv = [
                str(archive),
                "draw",
                "--group",
                str(grp_start),
                str(grp_end),
                "--arrays",
                str(arrays),
                "--record",
                str(rec_start),
                str(rec_end),
            ]
            with mock.patch.object(sgd5_effects.PagedResource, "read", side_effect=[b"SGD5", b"SG"]):
                self._assert_clean_failure(
                    argv,
                    expected_code=1,
                    expected_fragment="Truncated",
                    check_subprocess=False,
                )

    def test_os_value_struct_and_zlib_errors_caught_without_traceback(self) -> None:
        with TemporaryDirectory() as tmp:
            data, grp_start, grp_end, arrays, rec_start, rec_end, _, _ = _build_archive(0)
            archive = Path(tmp) / "probe.bin"
            archive.write_bytes(data)
            argv = [
                str(archive),
                "draw",
                "--group",
                str(grp_start),
                str(grp_end),
                "--arrays",
                str(arrays),
                "--record",
                str(rec_start),
                str(rec_end),
            ]

            # Real corrupt archive checks via both in-process main() and subprocess.
            bad_magic = bytearray(data)
            bad_magic[0:4] = b"BAD!"
            bad_magic_path = Path(tmp) / "bad_magic.bin"
            bad_magic_path.write_bytes(bad_magic)
            self._assert_clean_failure(
                [str(bad_magic_path), *argv[1:]],
                expected_code=1,
                expected_fragment="Missing SGD5 group header",
            )

            vs_bad, _ = _program_record("vs_main", 1, 101, pack=lambda blob: b"\xff" * len(_deflate_raw(blob)))
            ps0, _ = _program_record("ps_main", 0, 202)
            bad_arrays = _word(1) + vs_bad + _word(1) + ps0 + _word(0) * 4
            draw_effect = _effect_record("draw", [0], [0])
            bad_zlib_bytes = (
                b"SGD5"
                + bytes(28)
                + bad_arrays
                + _word(2)
                + _effect_record("idle", [], [])
                + draw_effect
            )
            bad_zlib_path = Path(tmp) / "bad_zlib.bin"
            bad_zlib_path.write_bytes(bad_zlib_bytes)
            self._assert_clean_failure(
                [
                    str(bad_zlib_path),
                    "draw",
                    "--group",
                    "0",
                    str(len(bad_zlib_bytes)),
                    "--arrays",
                    "32",
                    "--record",
                    str(len(bad_zlib_bytes) - len(draw_effect)),
                    str(len(bad_zlib_bytes)),
                ],
                expected_code=1,
                expected_fragment="Invalid compressed SGD5 candidate",
            )

            # Direct error-tuple mutation controls for (OSError, ValueError, struct.error, zlib.error).
            for exc in (
                OSError("simulated read error"),
                ValueError("simulated validation error"),
                struct.error("unpack requires a buffer of 12 bytes"),
                zlib.error("Error -3 while decompressing data"),
            ):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(sgd5_effects, "candidates", side_effect=exc):
                        self._assert_clean_failure(
                            argv,
                            expected_code=1,
                            expected_fragment=str(exc),
                            check_subprocess=False,
                        )

            self.assertEqual(
                sorted(p.name for p in Path(tmp).iterdir()),
                ["bad_magic.bin", "bad_zlib.bin", "probe.bin"],
            )


if __name__ == "__main__":
    unittest.main()
