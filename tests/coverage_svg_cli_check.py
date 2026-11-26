#!/usr/bin/env python3
"""CLI, validation, collision, atomic-write, and mutation suite for tools/coverage_svg.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "coverage_svg.py"
sys.path.insert(0, str(ROOT / "tools"))

import coverage_svg as cs  # noqa: E402


def _sample_coverage():
    return {
        "x64_zen2": {
            "covered": 4,
            "total": 1007,
            "blocked_by_skip": ["SSE movss"],
            "groups": {
                "Integer": {"covered": 3, "total": 1000},
                "SSE": {"covered": 1, "total": 6},
                "Tiny": {"covered": 0, "total": 1},
                "Zero": {"covered": 0, "total": 0},
            },
        },
        "gfx1013": {
            "covered": 0,
            "total": 0,
            "blocked_by_skip": [],
            "groups": {
                "ZeroOnly": {"covered": 0, "total": 0},
            },
        },
        "gfx1030": {
            "covered": 2,
            "total": 5,
            "blocked_by_skip": ["VOP3 v_add3_u32"],
            "groups": {
                "SOP1": {"covered": 2, "total": 3},
                "VOP3": {"covered": 0, "total": 2},
            },
        },
    }


def _run_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = cs.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args):
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class CoverageSvgCliPositiveTests(unittest.TestCase):
    def test_direct_main_writes_svg_creates_parents_and_leaves_zero_tmp(self):
        data = _sample_coverage()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cov = root / "coverage.json"
            cov.write_text(json.dumps(data), encoding="utf-8")
            out_svg = root / "nested" / "sub" / "dashboard.svg"
            self.assertFalse(out_svg.parent.exists())

            code, out, err = _run_main([str(cov), str(out_svg), "--source", "test-commit, test-os, warp"])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertTrue(out_svg.is_file())
            self.assertEqual(
                out_svg.read_text(encoding="utf-8"),
                cs.render(data, "test-commit, test-os, warp"),
            )
            self.assertEqual([p.name for p in out_svg.parent.iterdir()], ["dashboard.svg"])

    def test_sys_argv_and_script_entrypoint(self):
        data = _sample_coverage()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cov = root / "coverage.json"
            cov.write_text(json.dumps(data), encoding="utf-8")
            out_svg = root / "out" / "entry.svg"
            cli_argv = [str(TOOL), str(cov), str(out_svg), "--source", "entry-banner"]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertIn("Measured on entry-banner.", out_svg.read_text(encoding="utf-8"))

            out_svg.unlink()
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue(out_svg.is_file())

    def test_subprocess_end_to_end_positive(self):
        data = _sample_coverage()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cov = root / "coverage.json"
            cov.write_text(json.dumps(data), encoding="utf-8")
            out_svg = root / "sub.svg"

            proc = _run_subprocess(str(cov), str(out_svg), "--source", "sub-banner")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertEqual(proc.stderr, "")
            self.assertEqual(out_svg.read_text(encoding="utf-8"), cs.render(data, "sub-banner"))
            self.assertFalse(any(p.name.endswith(".tmp") for p in root.iterdir()))


class CoverageSvgCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(self, argv, *, expected_fragment, expected_code=1, check_subprocess=True):
        code, out, err = _run_main(argv)
        self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        if expected_code == 1:
            self.assertIn("coverage_svg:", err)
        else:
            self.assertIn("usage:", err.lower())
        if expected_fragment:
            self.assertIn(expected_fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            if expected_code == 1:
                self.assertIn("coverage_svg:", proc.stderr)
            else:
                self.assertIn("usage:", proc.stderr.lower())
            if expected_fragment:
                self.assertIn(expected_fragment, proc.stderr)

    def test_empty_and_whitespace_source_rejected_upfront_and_in_render(self):
        data = _sample_coverage()
        for bad_source in ("", "   ", "\t\n", None):
            with self.subTest(bad_source=bad_source):
                with self.assertRaisesRegex(ValueError, "source"):
                    cs.render(data, bad_source)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cov = root / "coverage.json"
            cov.write_text(json.dumps(data), encoding="utf-8")
            out_svg = root / "never_created" / "out.svg"

            for bad_source in ("", "   ", "\t\n"):
                with self.subTest(cli_source=bad_source):
                    with (
                        mock.patch.object(cs, "render") as render_mock,
                        mock.patch.object(cs, "_write_atomic") as write_mock,
                    ):
                        self._assert_clean_failure(
                            [str(cov), str(out_svg), "--source", bad_source],
                            expected_fragment="--source must not be empty",
                        )
                    render_mock.assert_not_called()
                    write_mock.assert_not_called()
                    self.assertFalse(out_svg.parent.exists())

    def test_render_rejects_bool_negative_and_covered_gt_total_counts(self):
        def _mutated(mutator):
            sample = _sample_coverage()
            mutator(sample)
            return sample

        cases = (
            ("area_bool_covered", lambda d: d["x64_zen2"].__setitem__("covered", True), TypeError, "integers"),
            ("area_bool_total", lambda d: d["x64_zen2"].__setitem__("total", False), TypeError, "integers"),
            ("area_float_covered", lambda d: d["x64_zen2"].__setitem__("covered", 1.5), TypeError, "integers"),
            ("area_neg_covered", lambda d: d["x64_zen2"].__setitem__("covered", -1), ValueError, "non-negative"),
            ("area_neg_total", lambda d: d["x64_zen2"].__setitem__("total", -1), ValueError, "non-negative"),
            (
                "area_covered_gt_total",
                lambda d: d["x64_zen2"].__setitem__("covered", d["x64_zen2"]["total"] + 1),
                ValueError,
                "exceed total",
            ),
            ("group_bool_covered", lambda d: d["x64_zen2"]["groups"]["Integer"].__setitem__("covered", True), TypeError, "integers"),
            ("group_bool_total", lambda d: d["x64_zen2"]["groups"]["Integer"].__setitem__("total", True), TypeError, "integers"),
            ("group_float_total", lambda d: d["x64_zen2"]["groups"]["Integer"].__setitem__("total", 2.5), TypeError, "integers"),
            ("group_neg_covered", lambda d: d["x64_zen2"]["groups"]["Integer"].__setitem__("covered", -1), ValueError, "non-negative"),
            ("group_neg_total", lambda d: d["x64_zen2"]["groups"]["Integer"].__setitem__("total", -1), ValueError, "non-negative"),
            (
                "group_covered_gt_total",
                lambda d: d["x64_zen2"]["groups"]["SSE"].__setitem__("covered", 7),
                ValueError,
                "exceed total",
            ),
        )
        for name, mutator, exc_type, fragment in cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(exc_type, fragment):
                    cs.render(_mutated(mutator), "banner")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_svg = root / "never" / "out.svg"
            for name, mutator, _, fragment in cases:
                with self.subTest(cli_case=name):
                    cov = root / f"{name}.json"
                    cov.write_text(json.dumps(_mutated(mutator)), encoding="utf-8")
                    self._assert_clean_failure(
                        [str(cov), str(out_svg), "--source", "banner"],
                        expected_fragment=fragment,
                    )
                    self.assertFalse(out_svg.parent.exists())
                    self.assertFalse(any(p.name.endswith(".tmp") for p in root.iterdir()))

    def test_coverage_input_missing_dir_and_non_regular_rejected_upfront(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.json"
            dir_in = root / "dir_in"
            dir_in.mkdir()
            reg_in = root / "coverage.json"
            reg_in.write_text(json.dumps(_sample_coverage()), encoding="utf-8")
            out_svg = root / "never" / "out.svg"

            for path, frag in (
                (missing, "coverage file not found"),
                (dir_in, "coverage path is a directory"),
            ):
                with self.subTest(path=path):
                    with (
                        mock.patch.object(cs, "render") as render_mock,
                        mock.patch.object(cs, "_write_atomic") as write_mock,
                    ):
                        self._assert_clean_failure(
                            [str(path), str(out_svg), "--source", "banner"],
                            expected_fragment=frag,
                        )
                    render_mock.assert_not_called()
                    write_mock.assert_not_called()
                    self.assertFalse(out_svg.parent.exists())

            with (
                mock.patch.object(Path, "is_file", return_value=False),
                mock.patch.object(cs, "render") as render_mock,
                mock.patch.object(cs, "_write_atomic") as write_mock,
            ):
                self._assert_clean_failure(
                    [str(reg_in), str(out_svg), "--source", "banner"],
                    expected_fragment="regular file",
                    check_subprocess=False,
                )
            render_mock.assert_not_called()
            write_mock.assert_not_called()
            self.assertFalse(out_svg.parent.exists())

    def test_output_dir_symlink_file_parent_and_samefile_rejected_upfront(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            cov = root / "coverage.json"
            cov.write_text(json.dumps(_sample_coverage()), encoding="utf-8")
            cov_bytes = cov.read_bytes()

            # 1) Existing directory output rejected upfront.
            out_dir = root / "existing_dir"
            out_dir.mkdir()
            with (
                mock.patch.object(cs, "render") as render_mock,
                mock.patch.object(cs, "_write_atomic") as write_mock,
            ):
                self._assert_clean_failure(
                    [str(cov), str(out_dir), "--source", "banner"],
                    expected_fragment="existing directory",
                )
            render_mock.assert_not_called()
            write_mock.assert_not_called()
            self.assertEqual(list(out_dir.iterdir()), [])

            # 2) Output symlink rejected upfront (real symlink if permitted, else mocked).
            target = root / "keep_target.svg"
            target.write_text("keep-target", encoding="utf-8")
            sym_out = root / "link.svg"
            try:
                sym_out.symlink_to(target)
                created_real = True
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                created_real = False

            if created_real:
                with (
                    mock.patch.object(cs, "render") as render_mock,
                    mock.patch.object(cs, "_write_atomic") as write_mock,
                ):
                    self._assert_clean_failure(
                        [str(cov), str(sym_out), "--source", "banner"],
                        expected_fragment="symbolic link",
                    )
                render_mock.assert_not_called()
                write_mock.assert_not_called()
                sym_out.unlink()
            else:
                with (
                    mock.patch.object(
                        Path,
                        "is_symlink",
                        autospec=True,
                        side_effect=lambda p, expected=sym_out: p.resolve() == expected.resolve(),
                    ),
                    mock.patch.object(cs, "render") as render_mock,
                    mock.patch.object(cs, "_write_atomic") as write_mock,
                ):
                    self._assert_clean_failure(
                        [str(cov), str(sym_out), "--source", "banner"],
                        expected_fragment="symbolic link",
                        check_subprocess=False,
                    )
                render_mock.assert_not_called()
                write_mock.assert_not_called()
            self.assertEqual(target.read_text(encoding="utf-8"), "keep-target")

            # 3) File-parent output (direct and nested under regular file) rejected upfront.
            file_parent = root / "regular_parent"
            file_parent.write_text("keep-parent", encoding="utf-8")
            for bad_out in (file_parent / "out.svg", file_parent / "nested" / "out.svg"):
                with self.subTest(file_parent_out=bad_out):
                    with (
                        mock.patch.object(cs, "render") as render_mock,
                        mock.patch.object(cs, "_write_atomic") as write_mock,
                    ):
                        self._assert_clean_failure(
                            [str(cov), str(bad_out), "--source", "banner"],
                            expected_fragment="parent path is not a directory",
                        )
                    render_mock.assert_not_called()
                    write_mock.assert_not_called()
            self.assertEqual(file_parent.read_text(encoding="utf-8"), "keep-parent")

            # 4) Output colliding with input via exact path, resolve() alias, and hardlink samefile().
            alias_out = sub / ".." / "coverage.json"
            hardlink_out = sub / "hardlink.svg"
            os.link(cov, hardlink_out)
            self.assertNotEqual(cov.resolve(), hardlink_out.resolve())
            self.assertTrue(cov.samefile(hardlink_out))

            for label, bad_out in (("exact", cov), ("alias", alias_out), ("samefile", hardlink_out)):
                with self.subTest(collision=label):
                    with (
                        mock.patch.object(cs, "render") as render_mock,
                        mock.patch.object(cs, "_write_atomic") as write_mock,
                    ):
                        self._assert_clean_failure(
                            [str(cov), str(bad_out), "--source", "banner"],
                            expected_fragment="must not match input",
                        )
                    render_mock.assert_not_called()
                    write_mock.assert_not_called()
                    self.assertEqual(cov.read_bytes(), cov_bytes)
            self.assertFalse(any(p.name.endswith(".tmp") for p in root.iterdir()))
            self.assertFalse(any(p.name.endswith(".tmp") for p in sub.iterdir()))

    def test_bad_json_and_atomic_write_failure_leave_zero_tmp_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_json = root / "bad.json"
            bad_json.write_text("{not json", encoding="utf-8")
            never_out = root / "never" / "out.svg"
            self._assert_clean_failure(
                [str(bad_json), str(never_out), "--source", "banner"],
                expected_fragment="Expecting",
            )
            self.assertFalse(never_out.parent.exists())
            self.assertFalse(any(p.name.endswith(".tmp") for p in root.iterdir()))

            cov = root / "coverage.json"
            cov.write_text(json.dumps(_sample_coverage()), encoding="utf-8")
            sentinel_dir = root / "sentinel_dir"
            sentinel_dir.mkdir()
            sentinel = sentinel_dir / "keep.svg"
            sentinel.write_text("prior-svg", encoding="utf-8")

            with mock.patch.object(Path, "replace", side_effect=OSError("simulated replace failure")):
                self._assert_clean_failure(
                    [str(cov), str(sentinel), "--source", "banner"],
                    expected_fragment="simulated replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "prior-svg")
            self.assertEqual([p.name for p in sentinel_dir.iterdir()], ["keep.svg"])

    def test_argparse_errors_py314_compatible_exit_2_without_traceback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cov = root / "coverage.json"
            cov.write_text(json.dumps(_sample_coverage()), encoding="utf-8")
            out_svg = root / "out.svg"

            for bad_argv in (
                [],
                [str(cov)],
                [str(cov), str(out_svg)],
                [str(cov), str(out_svg), "--source"],
                [str(cov), str(out_svg), "--source", "ok", "--unknown-flag"],
            ):
                with self.subTest(argv=bad_argv):
                    self._assert_clean_failure(
                        bad_argv,
                        expected_fragment="",
                        expected_code=2,
                    )
                    self.assertFalse(out_svg.exists())


if __name__ == "__main__":
    unittest.main()
