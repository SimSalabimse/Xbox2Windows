"""Pure-Python CLI, atomic-write, path-collision, error-guard, and mutation suite for tools/rdna_cvt_pk_u8.py."""

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
TOOL = ROOT / "tools" / "rdna_cvt_pk_u8.py"
MEASURED_TSV = ROOT / "tests" / "data" / "rdna_cvt_pk_u8_measured.tsv"
sys.path.insert(0, str(ROOT / "tools"))

import rdna_cvt_pk_u8  # noqa: E402


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = rdna_cvt_pk_u8.main(argv)
        except SystemExit as exc:
            return exc.code, stdout.getvalue(), stderr.getvalue()
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
        timeout=30,
    )


class _FakeHip:
    device = 0

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def count(self) -> int:
        return 2

    def arch(self, device: int | None = None) -> str:
        idx = self.device if device is None else device
        return "gfx1030:sramecc-:xnack-" if idx == 0 else "gfx1036"

    def run_kernel(self, hsaco: bytes, kname: str, words: list[tuple[int, int]]) -> list[int]:
        self.calls.append((kname, len(words)))
        model = "trunc_sat_nan0" if kname.endswith("_rtz") else "rne_sat_nan0"
        return [rdna_cvt_pk_u8.model_word(model, b, s) for b, s in words]


def _fake_build(arch: str, variant: str, workdir: str, args) -> tuple[bytes, str]:
    out = Path(workdir) / f"{variant}_{arch}.hsaco"
    out.write_bytes(b"HSACO_" + variant.encode("ascii"))
    return out.read_bytes(), str(out)


class RdnaCvtPkU8CliPositiveTests(unittest.TestCase):
    def test_self_test_direct_argv_and_subprocess(self) -> None:
        code, out, err = _run_main(["--self-test"])
        self.assertIs(type(code), int)
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        self.assertIn("self-test OK", out)

        with mock.patch.object(sys, "argv", [str(TOOL), "--self-test"]):
            code_default, out_default, err_default = _run_main(None)
        self.assertIs(type(code_default), int)
        self.assertEqual(code_default, 0, err_default)
        self.assertIn("self-test OK", out_default)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(sys, "argv", [str(TOOL), "--self-test"]),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exited,
        ):
            runpy.run_path(str(TOOL), run_name="__main__")
        self.assertIs(type(exited.exception.code), int)
        self.assertEqual(exited.exception.code, 0)
        self.assertIn("self-test OK", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

        proc = _run_subprocess("--self-test")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertIn("self-test OK", proc.stdout)

    def test_mocked_hip_writes_tsv_and_json_atomically_and_matches_measured(self) -> None:
        expected_rows = [
            line.rstrip("\n")
            for line in MEASURED_TSV.read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")
        ]
        self.assertEqual(len(expected_rows), 5432)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_tsv = root / "tsv_dir" / "rdna_cvt_pk_u8.tsv"
            out_json = root / "json_dir" / "rdna_cvt_pk_u8.json"
            seen_replaces: list[tuple[Path, Path]] = []
            real_replace = os.replace
            created_workdirs: list[Path] = []
            real_mkdtemp = rdna_cvt_pk_u8.tempfile.mkdtemp

            def tracking_replace(src, dst):
                src_p, dst_p = Path(src), Path(dst)
                seen_replaces.append((src_p, dst_p))
                self.assertEqual(src_p.parent.resolve(), dst_p.parent.resolve())
                return real_replace(src, dst)

            def tracking_mkdtemp(**kwargs):
                d = Path(real_mkdtemp(**kwargs))
                created_workdirs.append(d)
                return str(d)

            with (
                mock.patch.object(rdna_cvt_pk_u8, "Hip", _FakeHip),
                mock.patch.object(rdna_cvt_pk_u8, "build", side_effect=_fake_build),
                mock.patch.object(rdna_cvt_pk_u8.os, "replace", side_effect=tracking_replace),
                mock.patch.object(rdna_cvt_pk_u8.tempfile, "mkdtemp", side_effect=tracking_mkdtemp),
            ):
                code, out, err = _run_main([
                    "--out", str(out_tsv),
                    "--json", str(out_json),
                    "--show", "0",
                ])

            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn("HIP device 0: gfx1030:sramecc-:xnack-  <- used", out)
            self.assertIn("HIP device 1: gfx1036", out)
            self.assertIn("rows identical to", out)
            self.assertIn("results written to", out)
            self.assertIn("verbose JSON written to", out)

            self.assertEqual(len(created_workdirs), 1)
            self.assertFalse(created_workdirs[0].exists())

            self.assertEqual(len(seen_replaces), 2)
            self.assertEqual(
                {dst.resolve() for _, dst in seen_replaces},
                {out_tsv.resolve(), out_json.resolve()},
            )
            self.assertEqual([p.name for p in out_tsv.parent.iterdir()], [out_tsv.name])
            self.assertEqual([p.name for p in out_json.parent.iterdir()], [out_json.name])

            tsv_lines = out_tsv.read_text(encoding="utf-8").splitlines()
            self.assertIn("# device gfx1030", tsv_lines)
            tsv_rows = [line for line in tsv_lines if not line.startswith("#")]
            self.assertEqual(tsv_rows, expected_rows)

            doc = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(doc["arch"], "gfx1030")
            self.assertEqual(doc["sentinel"], rdna_cvt_pk_u8.SENTINEL)
            self.assertEqual(len(doc["rows"]), 5432)
            self.assertEqual(doc["summary"]["default"]["matching_models"], ["rne_sat_nan0"])
            self.assertEqual(doc["summary"]["rtz"]["matching_models"], ["trunc_sat_nan0"])
            self.assertEqual(doc["summary"]["flush"]["matching_models"], ["rne_sat_nan0"])

    def test_asm_only_with_keep_and_compare_diff_report(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep_dir = root / "keep_artifacts"
            with mock.patch.object(rdna_cvt_pk_u8, "build", side_effect=_fake_build):
                code, out, err = _run_main(["--asm-only", "--arch", "gfx1030", "--keep", str(keep_dir)])
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn("built default", out)
            self.assertTrue((keep_dir / "default_gfx1030.hsaco").is_file())

            # Compare with a modified TSV triggers the DIFFER summary line without failing.
            ref_lines = MEASURED_TSV.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(ref_lines):
                if not line.startswith("#"):
                    parts = line.split("\t")
                    parts[3] = "00000000"
                    ref_lines[idx] = "\t".join(parts)
                    break
            diff_ref = root / "diff_ref.tsv"
            diff_ref.write_text("\n".join(ref_lines) + "\n", encoding="utf-8")
            out_tsv = root / "out.tsv"

            with (
                mock.patch.object(rdna_cvt_pk_u8, "Hip", _FakeHip),
                mock.patch.object(rdna_cvt_pk_u8, "build", side_effect=_fake_build),
            ):
                code2, out2, err2 = _run_main([
                    "--out", str(out_tsv),
                    "--compare", str(diff_ref),
                ])
            self.assertEqual(code2, 0, err2)
            self.assertEqual(err2, "")
            self.assertIn("rows DIFFER from", out2)
            self.assertIn("(1 of 5432 rows)", out2)


class RdnaCvtPkU8CliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int = 1,
        expected_fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
        allow_stdout: bool = False,
        cwd: Path | None = None,
    ) -> str:
        code, out, err = _run_main(argv)
        self.assertIs(type(code), int)
        self.assertEqual(code, expected_code, err)
        if not allow_stdout:
            self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip(), f"expected stderr output for {argv!r}")
        for fragment in expected_fragments:
            self.assertIn(fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv, cwd=cwd)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            if not allow_stdout:
                self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip(), f"expected subprocess stderr output for {argv!r}")
            for fragment in expected_fragments:
                self.assertIn(fragment, proc.stderr)
        return err

    def test_asm_only_needs_arch_and_hip_runtime_errors(self) -> None:
        # --asm-only without --arch -> stderr + return 2 (not SystemExit).
        with mock.patch.object(rdna_cvt_pk_u8, "Hip", side_effect=AssertionError("Hip() must not run")):
            self._assert_clean_failure(
                ["--asm-only"],
                expected_code=2,
                expected_fragments=("error: --asm-only needs --arch",),
            )

        with TemporaryDirectory() as tmp:
            out_tsv = Path(tmp) / "out.tsv"
            with mock.patch.object(rdna_cvt_pk_u8, "Hip", side_effect=RuntimeError("no HIP dll")):
                self._assert_clean_failure(
                    ["--out", str(out_tsv)],
                    expected_code=2,
                    expected_fragments=("error: no HIP dll",),
                    check_subprocess=False,
                )

            class _FailingKernelHip(_FakeHip):
                def run_kernel(self, hsaco, kname, words):
                    raise RuntimeError("kernel launch failed")

            with (
                mock.patch.object(rdna_cvt_pk_u8, "Hip", _FailingKernelHip),
                mock.patch.object(rdna_cvt_pk_u8, "build", side_effect=_fake_build),
            ):
                self._assert_clean_failure(
                    ["--out", str(out_tsv)],
                    expected_code=3,
                    expected_fragments=("error: kernel launch failed",),
                    check_subprocess=False,
                    allow_stdout=True,
                )
            self.assertFalse(out_tsv.exists())

    def test_flag_and_path_type_validation_guards(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg_file = root / "regular.txt"
            reg_file.write_text("keep-me", encoding="utf-8")
            existing_dir = root / "existing_dir"
            existing_dir.mkdir()
            valid_out = root / "ok.tsv"

            cases = [
                (["--show", "-1", "--out", str(valid_out)], "--show must be >= 0", True),
                (["--keep", "", "--out", str(valid_out)], "--keep path must not be empty", True),
                (["--keep", str(reg_file), "--out", str(valid_out)], "--keep path is not a directory", True),
                (["--keep", str(reg_file / "sub"), "--out", str(valid_out)], "--keep parent path is not a directory", True),
                (["--compare", str(existing_dir), "--out", str(valid_out)], "--compare path is a directory", True),
                (["--out", ""], "--out path must not be empty", True),
                (["--out", str(existing_dir) + os.sep], "--out path is a directory path", True),
                (["--out", str(existing_dir)], "--out path is a directory", True),
                (["--out", str(reg_file / "child.tsv")], "--out parent path is not a directory", True),
                (["--out", str(reg_file / "sub" / "child.tsv")], "--out parent path is not a directory", True),
                (["--out", str(valid_out), "--json", ""], "--json path must not be empty", True),
                (["--out", str(valid_out), "--json", str(existing_dir) + "/"], "--json path is a directory path", True),
                (["--out", str(valid_out), "--json", str(existing_dir)], "--json path is a directory", True),
                (["--out", str(valid_out), "--json", str(reg_file / "child.json")], "--json parent path is not a directory", True),
            ]
            for argv, fragment, subproc in cases:
                with self.subTest(argv=argv):
                    with (
                        mock.patch.object(rdna_cvt_pk_u8, "Hip", side_effect=AssertionError("Hip() must not run")),
                        mock.patch.object(rdna_cvt_pk_u8, "build", side_effect=AssertionError("build() must not run")),
                    ):
                        self._assert_clean_failure(
                            argv,
                            expected_code=1,
                            expected_fragments=("error:", fragment),
                            check_subprocess=subproc,
                        )
            self.assertEqual(reg_file.read_text(encoding="utf-8"), "keep-me")
            self.assertEqual(list(existing_dir.iterdir()), [])

    def test_symlink_output_guards_kill_mutations(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.txt"
            target.write_text("do-not-clobber", encoding="utf-8")
            valid_out = root / "valid.tsv"

            for flag, name in (("--out", "out_link.tsv"), ("--json", "json_link.json")):
                link = root / name
                try:
                    link.symlink_to(target)
                    created_real = True
                except OSError as exc:
                    if getattr(exc, "winerror", None) != 1314:
                        raise
                    created_real = False

                argv = [flag, str(link)] if flag == "--out" else ["--out", str(valid_out), flag, str(link)]
                if created_real:
                    self._assert_clean_failure(
                        argv,
                        expected_code=1,
                        expected_fragments=("error:", f"{flag} path must not be a symlink"),
                    )
                    link.unlink()
                else:
                    with (
                        mock.patch.object(
                            Path,
                            "is_symlink",
                            autospec=True,
                            side_effect=lambda p, expected=link: os.path.normcase(str(p.resolve()))
                            == os.path.normcase(str(expected.resolve())),
                        ),
                        mock.patch.object(rdna_cvt_pk_u8, "Hip", side_effect=AssertionError("Hip() must not run")),
                    ):
                        self._assert_clean_failure(
                            argv,
                            expected_code=1,
                            expected_fragments=("error:", f"{flag} path must not be a symlink"),
                            check_subprocess=False,
                        )
            self.assertEqual(target.read_text(encoding="utf-8"), "do-not-clobber")

    def test_collision_guards_alias_nonexistent_and_hardlink(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            ref = root / "ref.tsv"
            ref.write_text("# ref\n", encoding="utf-8")
            ref_hardlink = root / "ref_hardlink.tsv"
            os.link(ref, ref_hardlink)
            self.assertNotEqual(ref.resolve(), ref_hardlink.resolve())
            self.assertTrue(ref.samefile(ref_hardlink))

            ghost_a = root / "ghost.txt"
            ghost_b = sub / ".." / "ghost.txt"
            other_out = root / "other_out.tsv"

            collision_cases = [
                (["--out", str(ref), "--compare", str(ref)], "--out path collides with --compare"),
                (["--out", str(sub / ".." / "ref.tsv"), "--compare", str(ref)], "--out path collides with --compare"),
                (["--out", str(ref_hardlink), "--compare", str(ref)], "--out path collides with --compare"),
                (["--out", str(ghost_a), "--compare", str(ghost_b)], "--out path collides with --compare"),
                (["--out", str(other_out), "--json", str(ref), "--compare", str(ref)], "--json path collides with --compare"),
                (["--out", str(other_out), "--json", str(ref_hardlink), "--compare", str(ref)], "--json path collides with --compare"),
                (["--out", str(ghost_a), "--json", str(ghost_b)], "--out and --json paths must not collide"),
                (["--out", str(ref), "--json", str(ref_hardlink)], "--out and --json paths must not collide"),
                (["--keep", str(ghost_a), "--out", str(ghost_b)], "--keep and --out paths must not collide"),
            ]
            for argv, fragment in collision_cases:
                with self.subTest(argv=argv):
                    with mock.patch.object(rdna_cvt_pk_u8, "Hip", side_effect=AssertionError("Hip() must not run")):
                        self._assert_clean_failure(
                            argv,
                            expected_code=1,
                            expected_fragments=("error:", fragment),
                        )
            self.assertEqual(ref.read_text(encoding="utf-8"), "# ref\n")

    def test_atomic_write_and_compare_failures_leave_zero_temps(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_tsv = root / "out_dir" / "res.tsv"
            out_json = root / "json_dir" / "res.json"
            out_tsv.parent.mkdir()
            out_json.parent.mkdir()
            out_tsv.write_text("PRIOR_TSV", encoding="utf-8")
            out_json.write_text("PRIOR_JSON", encoding="utf-8")

            # 1) os.replace fails on --out: prior files preserved, zero sibling .tmp leftovers.
            with (
                mock.patch.object(rdna_cvt_pk_u8, "Hip", _FakeHip),
                mock.patch.object(rdna_cvt_pk_u8, "build", side_effect=_fake_build),
                mock.patch.object(rdna_cvt_pk_u8.os, "replace", side_effect=OSError("simulated tsv replace fail")),
            ):
                self._assert_clean_failure(
                    ["--out", str(out_tsv), "--json", str(out_json)],
                    expected_code=1,
                    expected_fragments=("error: simulated tsv replace fail",),
                    check_subprocess=False,
                    allow_stdout=True,
                )
            self.assertEqual(out_tsv.read_text(encoding="utf-8"), "PRIOR_TSV")
            self.assertEqual(out_json.read_text(encoding="utf-8"), "PRIOR_JSON")
            self.assertEqual([p.name for p in out_tsv.parent.iterdir()], ["res.tsv"])
            self.assertEqual([p.name for p in out_json.parent.iterdir()], ["res.json"])

            # 2) os.replace fails on --json after --out: prior --json preserved, zero sibling .tmp leftovers.
            real_replace = os.replace
            calls = []

            def fail_second_replace(src, dst):
                calls.append(Path(dst))
                if len(calls) == 2:
                    raise OSError("simulated json replace fail")
                return real_replace(src, dst)

            with (
                mock.patch.object(rdna_cvt_pk_u8, "Hip", _FakeHip),
                mock.patch.object(rdna_cvt_pk_u8, "build", side_effect=_fake_build),
                mock.patch.object(rdna_cvt_pk_u8.os, "replace", side_effect=fail_second_replace),
            ):
                self._assert_clean_failure(
                    ["--out", str(out_tsv), "--json", str(out_json)],
                    expected_code=1,
                    expected_fragments=("error: simulated json replace fail",),
                    check_subprocess=False,
                    allow_stdout=True,
                )
            self.assertEqual(out_json.read_text(encoding="utf-8"), "PRIOR_JSON")
            self.assertEqual([p.name for p in out_tsv.parent.iterdir()], ["res.tsv"])
            self.assertEqual([p.name for p in out_json.parent.iterdir()], ["res.json"])

            # 3) Corrupt UTF-8 --compare file -> ValueError/UnicodeDecodeError caught -> clean exit 1.
            bad_cmp = root / "bad_utf8.tsv"
            bad_cmp.write_bytes(b"\xff\xfe\xfd")
            with (
                mock.patch.object(rdna_cvt_pk_u8, "Hip", _FakeHip),
                mock.patch.object(rdna_cvt_pk_u8, "build", side_effect=_fake_build),
            ):
                self._assert_clean_failure(
                    ["--out", str(out_tsv), "--compare", str(bad_cmp)],
                    expected_code=1,
                    expected_fragments=("error:",),
                    check_subprocess=False,
                    allow_stdout=True,
                )

            # 4) json.JSONDecodeError caught cleanly if raised during main flow.
            with (
                mock.patch.object(rdna_cvt_pk_u8, "Hip", _FakeHip),
                mock.patch.object(rdna_cvt_pk_u8, "build", side_effect=_fake_build),
                mock.patch.object(
                    rdna_cvt_pk_u8.json,
                    "dumps",
                    side_effect=json.JSONDecodeError("bad json", "x", 0),
                ),
            ):
                self._assert_clean_failure(
                    ["--out", str(out_tsv), "--json", str(out_json)],
                    expected_code=1,
                    expected_fragments=("error:", "bad json"),
                    check_subprocess=False,
                    allow_stdout=True,
                )

    def test_argparse_usage_errors_compatible_with_py314(self) -> None:
        for bad_argv, fragment in (
            (["--unknown-flag"], "--unknown-flag"),
            (["--arch"], "--arch"),
            (["--show", "not-an-int"], "--show"),
            (["--cov", "9"], "--cov"),
            (["--build", "bogus"], "--build"),
        ):
            with self.subTest(argv=bad_argv):
                self._assert_clean_failure(
                    bad_argv,
                    expected_code=2,
                    expected_fragments=("usage:", fragment),
                )


if __name__ == "__main__":
    unittest.main()
