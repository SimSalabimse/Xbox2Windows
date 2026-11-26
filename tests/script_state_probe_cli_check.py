"""Pure-Python CLI and synthetic RPF9 validation suite for tools/script_state_probe.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import mmap
import os
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
TOOL = ROOT / "tools" / "script_state_probe.py"
sys.path.insert(0, str(ROOT / "tools"))

import script_state_probe as probe  # noqa: E402


def _attr(name: str, spelling: str, raw: int, kind: int) -> bytes:
    return struct.pack("<IIIBBH", probe.joaat(name), probe.joaat(spelling), raw, kind, 1, 0)


def _camera_rows(priority: int) -> list[bytes]:
    return [
        _attr("priority", str(priority), priority, 3),
        _attr("blendduration", "25", 25, 3),
        _attr("orbitdistance", "2.500000", 0x40200000, 2),
        _attr("pitchlimits", "-0.500000,0.500000,0.000000", 0, 1),
        _attr("headinglimits", "-1.500000,1.500000,0.000000", 1, 1),
    ]


def _synth_record(
    *,
    priorities: tuple[int, ...] = (7,),
    state_label: str = "cli_synthetic_state",
    truncate_schema_by: int = 0,
    corrupt_zlib: bool = False,
    truncate_zlib: bool = False,
) -> bytes:
    """Build one zlib-compressed PSIN/LSCH/LENM/LSNS/LGID/LRSC/CHKS record payload."""
    attrs = bytearray()
    nodes = bytearray()

    def add_node(name: str, parent: int, row_list: list[bytes]) -> int:
        idx = len(nodes) // 16
        nodes.extend(
            struct.pack(
                "<I5H2B",
                probe.joaat(name),
                parent,
                0,
                0,
                0,
                len(attrs) // 16,
                len(row_list),
                1,
            )
        )
        attrs.extend(b"".join(row_list))
        return idx

    root_parent = 65535
    if len(priorities) > 1:
        root_parent = add_node("root", root_parent, [])
    for prio in priorities:
        state_idx = add_node("state", root_parent, [_attr("id", state_label, 0, 5)])
        add_node("camera", state_idx, _camera_rows(prio))

    vecs = struct.pack("<6f", -0.5, 0.5, 0.0, -1.5, 1.5, 0.0)
    arrays = ((len(attrs) // 16, 2), (0, 0), (2, 4), (0, 0), (len(nodes) // 16, 3))
    root_blob = b"".join(struct.pack("<IIQ", n, n, ref) for n, ref in arrays)
    blocks = (root_blob, bytes(attrs), bytes(nodes), vecs)

    instances = bytearray()
    for kind, blob in enumerate(blocks, 1):
        instances.extend(struct.pack("<IIQ", kind, len(blob), 0xAAAAAAAAAAAAAAAA))
    for blob in blocks:
        instances.extend(bytes(-(len(instances) + 8) % 16))
        instances.extend(blob)

    definitions = (
        (80, (("attributes", 0, 0x5C),)),
        (16, (("name", 0, 0x4A), ("valueHash", 4, 0x4A))),
        (16, (("name", 0, 0x4A), ("parentIndex", 4, 0x13), ("attributeCount", 14, 0x11))),
        (12, tuple((axis, i * 4, 0x14) for i, axis in enumerate("xyz"))),
    )
    schema = bytearray()
    for kind, (size, fields) in enumerate(definitions, 1):
        schema.extend(struct.pack("<IIIIQIHH", kind, 0, 0, 0, 0, size, 0, len(fields)))
    for _, fields in definitions:
        for field_name, off, code in fields:
            schema.extend(struct.pack("<IIBBHI", probe.joaat(field_name), off, code, 0, 0, 0))
    if truncate_schema_by:
        del schema[-truncate_schema_by:]

    header = bytearray(80)
    header[:5] = b"0DRPz"
    struct.pack_into("<I", header, 16, 1)
    struct.pack_into("<3H", header, 56, 4, 0, 4)

    body = bytearray()
    for tag, part in (
        (b"PSIN", instances),
        (b"LSCH", schema),
        (b"LENM", b""),
        (b"LSNS", b""),
        (b"LGID", b""),
        (b"LRSC", header),
    ):
        body.extend(struct.pack(">4sI", tag, len(part) + 8))
        body.extend(part)

    chks_size = 20
    total_len = len(body) + chks_size
    body.extend(struct.pack(">4sIIQ", b"CHKS", chks_size, total_len, 0))

    compressed = zlib.compress(bytes(body))
    if corrupt_zlib:
        return compressed[:4] + b"\xff\x00\xff\x00" + compressed[8:]
    if truncate_zlib:
        return compressed[:6]
    return compressed


def _build_rpf9(*payloads: bytes) -> bytes:
    archive = bytearray(b"9FPR")
    for payload in payloads:
        archive.extend(bytes(-len(archive) % 16))
        archive.extend(payload)
    return bytes(archive)


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            rc = probe.main(argv)
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 2
    return rc, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONUTF8="1")
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=30,
    )


class ScriptStateProbeCliPositiveTests(unittest.TestCase):
    def test_direct_main_default_and_state_schema(self):
        single_bytes = _build_rpf9(_synth_record(priorities=(7,), state_label="orbit_one"))
        ambig_bytes = _build_rpf9(_synth_record(priorities=(7, 11), state_label="orbit_dup"))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            single = root / "single.rpf"
            single.write_bytes(single_bytes)
            ambig = root / "ambig.rpf"
            ambig.write_bytes(ambig_bytes)

            # Default mode without --state.
            rc, out, err = _run_main([str(single)])
            self.assertEqual(rc, 0, err)
            self.assertEqual(err, "")
            default_json = json.loads(out)
            self.assertEqual(
                set(default_json),
                {"counts", "requested_state", "first_verified_example"},
            )
            self.assertIsNone(default_json["requested_state"])
            self.assertIsNone(default_json["first_verified_example"])
            self.assertEqual(
                default_json["counts"],
                {
                    "section_records": 1,
                    "node_trees": 1,
                    "state_nodes": 1,
                    "camera_parameter_sets": 1,
                    "all_five_literals_verified": 1,
                },
            )
            self.assertEqual(single.read_bytes(), single_bytes)

            # Matching --state mode.
            rc, out, err = _run_main([str(single), "--state", "orbit_one"])
            self.assertEqual(rc, 0, err)
            self.assertEqual(err, "")
            matched_json = json.loads(out)
            self.assertEqual(matched_json["requested_state"], "orbit_one")
            self.assertEqual(
                matched_json["first_verified_example"],
                {
                    "priority": "7",
                    "blendduration": "25",
                    "orbitdistance": "2.500000",
                    "pitchlimits": "-0.500000,0.500000,0.000000",
                    "headinglimits": "-1.500000,1.500000,0.000000",
                },
            )
            self.assertNotIn("ambiguous_state_candidates", matched_json)

            # Unmatched --state mode.
            rc, out, err = _run_main([str(single), "--state", "absent_state"])
            self.assertEqual(rc, 0, err)
            self.assertEqual(err, "")
            missed_json = json.loads(out)
            self.assertEqual(missed_json["requested_state"], "absent_state")
            self.assertIsNone(missed_json["first_verified_example"])

            # Ambiguous --state mode.
            rc, out, err = _run_main([str(ambig), "--state", "orbit_dup"])
            self.assertEqual(rc, 0, err)
            self.assertEqual(err, "")
            ambig_json = json.loads(out)
            self.assertIsNone(ambig_json["first_verified_example"])
            self.assertEqual(
                [c["node_path"] for c in ambig_json["ambiguous_state_candidates"]],
                [[0, 1], [0, 3]],
            )

            # Subprocess parity and sys.argv/runpy entrypoint.
            proc = _run_subprocess(str(single), "--state", "orbit_one")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(json.loads(proc.stdout), matched_json)

            with mock.patch.object(sys, "argv", [str(TOOL), str(single), "--state", "orbit_one"]):
                rc_none, out_none, err_none = _run_main(None)
            self.assertEqual(rc_none, 0, err_none)
            self.assertEqual(json.loads(out_none), matched_json)

            out_runpy, err_runpy = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(single)]),
                redirect_stdout(out_runpy),
                redirect_stderr(err_runpy),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(err_runpy.getvalue(), "")
            self.assertEqual(json.loads(out_runpy.getvalue()), default_json)

    def test_corrupt_or_truncated_zlib_yields_zero_node_trees_without_crash(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, kw in (
                ("corrupt_zlib", {"corrupt_zlib": True}),
                ("truncated_zlib", {"truncate_zlib": True}),
            ):
                with self.subTest(case=label):
                    archive = root / f"{label}.rpf"
                    archive.write_bytes(_build_rpf9(_synth_record(**kw)))
                    rc, out, err = _run_main([str(archive)])
                    self.assertEqual(rc, 0, err)
                    self.assertEqual(err, "")
                    payload = json.loads(out)
                    self.assertEqual(payload["counts"].get("node_trees", 0), 0)
                    self.assertEqual(payload["counts"], {})


class ScriptStateProbeCliNegativeTests(unittest.TestCase):
    def _assert_cli_failure(
        self,
        argv: list[str],
        *,
        expected_code: int,
        fragment: str,
        check_subprocess: bool = True,
    ) -> None:
        rc, out, err = _run_main(argv)
        self.assertEqual(rc, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn(fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(fragment, proc.stderr)

    def test_upfront_is_file_and_archive_validation_controls(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.rpf"
            folder = root / "dir_archive"
            folder.mkdir()
            empty = root / "empty.rpf"
            empty.write_bytes(b"")
            short_header = root / "short.rpf"
            short_header.write_bytes(b"9FP")
            bad_magic = root / "bad_magic.rpf"
            bad_magic.write_bytes(b"NOPE" + b"\x00" * 16)
            truncated_schema = root / "truncated_schema.rpf"
            truncated_schema.write_bytes(_build_rpf9(_synth_record(truncate_schema_by=8)))

            # Missing and directory must reject upfront before open/mmap.
            with (
                mock.patch.object(Path, "open", side_effect=AssertionError("open must not run")),
                mock.patch.object(mmap, "mmap", side_effect=AssertionError("mmap must not run")),
            ):
                with self.assertRaises(FileNotFoundError):
                    probe.inspect(missing)
                with self.assertRaisesRegex(ValueError, "directory"):
                    probe.inspect(folder)

            self._assert_cli_failure([str(missing)], expected_code=1, fragment="not found")
            self._assert_cli_failure([str(folder)], expected_code=1, fragment="directory")

            # 0-byte archive must fail cleanly before mmap.mmap.
            with mock.patch.object(mmap, "mmap", side_effect=AssertionError("mmap must not run")):
                with self.assertRaisesRegex(ValueError, "Empty archive"):
                    probe.inspect(empty)
            self._assert_cli_failure([str(empty)], expected_code=1, fragment="Empty archive")

            # Truncated header, bad magic, and truncated inner schema.
            self._assert_cli_failure([str(short_header)], expected_code=1, fragment="Expected an RPF9 archive")
            self._assert_cli_failure([str(bad_magic)], expected_code=1, fragment="Expected an RPF9 archive")
            self._assert_cli_failure([str(truncated_schema)], expected_code=1, fragment="Truncated schema members")

    def test_empty_and_non_ascii_state_validation(self):
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "valid.rpf"
            archive.write_bytes(_build_rpf9(_synth_record()))

            # Upfront state validation rejects before open/mmap.
            with (
                mock.patch.object(Path, "open", side_effect=AssertionError("open must not run")),
                mock.patch.object(mmap, "mmap", side_effect=AssertionError("mmap must not run")),
            ):
                with self.assertRaisesRegex(ValueError, "non-empty ASCII"):
                    probe.inspect(archive, "")
                with self.assertRaisesRegex(ValueError, "ASCII"):
                    probe.inspect(archive, "café")
                with self.assertRaisesRegex(ValueError, "non-empty ASCII"):
                    probe.joaat("", allow_empty=False)

            # Empty and non-ASCII --state via main(argv) and subprocess.
            self._assert_cli_failure([str(archive), "--state", ""], expected_code=1, fragment="non-empty ASCII")
            self._assert_cli_failure([str(archive), "--state", "café"], expected_code=1, fragment="ASCII")

    def test_argparse_errors_and_help_py314_compatible(self):
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "valid.rpf"
            archive.write_bytes(_build_rpf9(_synth_record()))

            for argv, fragment in (
                ([], "usage:"),
                ([str(archive), "--unknown-flag"], "unrecognized"),
                ([str(archive), "--state"], "--state"),
            ):
                with self.subTest(argv=argv):
                    self._assert_cli_failure(argv, expected_code=2, fragment=fragment)

            rc, out, err = _run_main(["--help"])
            self.assertEqual(rc, 0)
            self.assertIn("usage:", out)
            self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
