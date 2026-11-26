#!/usr/bin/env python3
"""Pure-Python CLI, atomic-write, path-guard, and error-handling suite for tools/bink_heap_aot.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "bink_heap_aot.py"
sys.path.insert(0, str(TOOLS_DIR))
import bink_heap_aot  # noqa: E402


SYNTHETIC_CPP = (
    '#include <cstdint>\n'
    'using word = uint64_t;\n'
    'extern "C" __declspec(dllexport) void* heap_request(word bytes) { return nullptr; }\n'
)


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = bink_heap_aot.main(argv)
        except SystemExit as exc:
            code = exc.code
            if not isinstance(code, int):
                if code is not None:
                    stderr.write(f"{code}\n")
                code = 0 if code is None else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class BinkHeapAotCliPositiveTests(unittest.TestCase):
    def test_atomic_cpp_output_in_nested_new_dir_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "synthetic.dll"
            image.write_bytes(b"synthetic-image-placeholder")
            nested_out = base / "nested" / "sub" / "heap_aot.cpp"

            # 1) Create missing parent directories and write atomically with 0 .tmp files.
            with mock.patch.object(bink_heap_aot, "generate", return_value=SYNTHETIC_CPP) as gen:
                code, out, err = _run_main([str(image), str(nested_out)])
            gen.assert_called_once_with(image, None, None)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(nested_out.read_text(encoding="ascii"), SYNTHETIC_CPP)
            self.assertEqual(sorted(p.name for p in nested_out.parent.iterdir()), ["heap_aot.cpp"])

            # 2) Forward --entries and --group-entry, and atomically overwrite existing file with 0 .tmp files.
            nested_out.write_text("stale content", encoding="ascii")
            entries_cpp = SYNTHETIC_CPP + "// entries-mode\n"
            with mock.patch.object(bink_heap_aot, "generate", return_value=entries_cpp) as gen_entries:
                code_ent, out_ent, err_ent = _run_main([
                    str(image),
                    str(nested_out),
                    "--entries",
                    "0x1000",
                    "0x2000",
                    "--group-entry",
                    "heap_group",
                ])
            gen_entries.assert_called_once_with(image, [0x1000, 0x2000], "heap_group")
            self.assertEqual(code_ent, 0, err_ent)
            self.assertEqual(out_ent, "")
            self.assertEqual(err_ent, "")
            self.assertEqual(nested_out.read_text(encoding="ascii"), entries_cpp)
            self.assertEqual(sorted(p.name for p in nested_out.parent.iterdir()), ["heap_aot.cpp"])

            # 3) Verify sys.argv forwarding when argv=None.
            cli_argv = [str(TOOL), str(image), str(nested_out)]
            with (
                mock.patch.object(sys, "argv", cli_argv),
                mock.patch.object(bink_heap_aot, "generate", return_value=SYNTHETIC_CPP),
            ):
                code_none, out_none, err_none = _run_main(None)
            self.assertEqual(code_none, 0, err_none)
            self.assertEqual(out_none, "")
            self.assertEqual(err_none, "")
            self.assertEqual(nested_out.read_text(encoding="ascii"), SYNTHETIC_CPP)

            # 4) Verify __main__ entrypoint propagates SystemExit cleanly on invalid PE.
            bad_pe = base / "bad_main.dll"
            bad_pe.write_bytes(b"not a PE")
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(bad_pe), str(base / "unused.cpp")]),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertTrue(stderr.getvalue().strip())


class BinkHeapAotCliNegativeTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None,
        fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> str:
        code, out, err = _run_main(argv)
        if expected_code is None:
            self.assertNotEqual(code, 0)
        else:
            self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip())
        for fragment in fragments:
            self.assertIn(fragment.lower(), err.lower())

        if check_subprocess:
            proc = _run_subprocess(*argv)
            if expected_code is None:
                self.assertNotEqual(proc.returncode, 0)
            else:
                self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            for fragment in fragments:
                self.assertIn(fragment.lower(), proc.stderr.lower())
        return err

    def test_missing_and_directory_image_rejected_before_generate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dir_image = base / "dir_image"
            dir_image.mkdir()
            absent_out = base / "should_not_exist" / "heap_aot.cpp"

            with (
                mock.patch.object(
                    bink_heap_aot,
                    "generate",
                    side_effect=AssertionError("generate called for invalid image path"),
                ),
                mock.patch.object(
                    bink_heap_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for invalid image path"),
                ),
            ):
                self._assert_clean_failure(
                    [str(base / "missing.dll"), str(absent_out)],
                    expected_code=1,
                    fragments=("image file not found",),
                )
                self._assert_clean_failure(
                    [str(dir_image), str(absent_out)],
                    expected_code=1,
                    fragments=("image path is a directory",),
                )

            self.assertFalse(absent_out.parent.exists())

    def test_invalid_pe_bytes_exit_1_without_creating_output_or_temps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            bad_pe = base / "bad.dll"
            bad_pe.write_bytes(b"not a valid PE32+ image")
            absent_out = base / "absent_dir" / "heap_aot.cpp"

            self._assert_clean_failure(
                [str(bad_pe), str(absent_out)],
                expected_code=1,
            )
            self.assertFalse(absent_out.parent.exists())

    def test_negative_entries_and_empty_group_entry_rejected_before_generate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"synthetic-image-bytes")
            out_cpp = base / "absent_dir" / "heap_aot.cpp"

            with (
                mock.patch.object(
                    bink_heap_aot,
                    "generate",
                    side_effect=AssertionError("generate called for invalid entries/group-entry"),
                ),
                mock.patch.object(
                    bink_heap_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for invalid entries/group-entry"),
                ),
            ):
                for label, argv, frags in (
                    ("neg_alloc_rva", [str(image), str(out_cpp), "--entries", "-1", "0x2000"], ("non-negative",)),
                    ("neg_free_rva", [str(image), str(out_cpp), "--entries", "0x1000", "-4"], ("non-negative",)),
                    ("both_neg_rva", [str(image), str(out_cpp), "--entries", "-1", "-4"], ("non-negative",)),
                    ("empty_group_entry", [str(image), str(out_cpp), "--group-entry", ""], ("--group-entry",)),
                    ("blank_group_entry", [str(image), str(out_cpp), "--group-entry", "   "], ("--group-entry",)),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, expected_code=1, fragments=frags)

            self.assertFalse(out_cpp.parent.exists())

    def test_output_guards_reject_before_generate_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"synthetic-image-bytes")
            same_cpp = base / "same.cpp"
            same_cpp.write_bytes(b"synthetic-cpp-bytes")
            hardlink_cpp = base / "hardlink.cpp"
            hardlink_cpp.hardlink_to(image)
            out_dir = base / "existing_dir.cpp"
            out_dir.mkdir()
            file_parent = base / "regular_parent.txt"
            file_parent.write_text("keep-me", encoding="ascii")

            with (
                mock.patch.object(
                    bink_heap_aot,
                    "generate",
                    side_effect=AssertionError("generate called for invalid output path"),
                ),
                mock.patch.object(
                    bink_heap_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for invalid output path"),
                ),
            ):
                for label, argv, code, frags in (
                    ("non_cpp_suffix", [str(image), str(base / "out.txt")], 2, ("distinct .cpp output",)),
                    ("empty_output", [str(image), ""], 2, ("distinct .cpp output",)),
                    ("same_path", [str(same_cpp), str(same_cpp)], 2, ("distinct .cpp output",)),
                    ("hardlink_samefile", [str(image), str(hardlink_cpp)], 2, ("distinct .cpp output",)),
                    ("output_is_dir", [str(image), str(out_dir)], 1, ("directory",)),
                    ("parent_is_file", [str(image), str(file_parent / "heap_aot.cpp")], 1, ("not a directory",)),
                    ("ancestor_is_file", [str(image), str(file_parent / "sub" / "heap_aot.cpp")], 1, ("not a directory",)),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, expected_code=code, fragments=frags)

            self.assertEqual(same_cpp.read_bytes(), b"synthetic-cpp-bytes")
            self.assertEqual(image.read_bytes(), b"synthetic-image-bytes")
            self.assertEqual(list(out_dir.iterdir()), [])
            self.assertEqual(file_parent.read_text(encoding="ascii"), "keep-me")
            self.assertFalse((base / "out.txt").exists())

    def test_output_symlink_rejected_before_generate_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"synthetic-image-bytes")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            real_target = out_dir / "real.cpp"
            real_target.write_text("untouched", encoding="ascii")
            live_link = out_dir / "live_link.cpp"
            dangling_link = out_dir / "dangling_link.cpp"
            try:
                live_link.symlink_to(real_target)
                dangling_link.symlink_to(out_dir / "missing.cpp")
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                self.skipTest("this Windows account cannot create symlinks (WinError 1314)")

            with (
                mock.patch.object(
                    bink_heap_aot,
                    "generate",
                    side_effect=AssertionError("generate called for symlink output"),
                ),
                mock.patch.object(
                    bink_heap_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for symlink output"),
                ),
            ):
                for label, link in (("live_symlink", live_link), ("dangling_symlink", dangling_link)):
                    with self.subTest(case=label):
                        self._assert_clean_failure(
                            [str(image), str(link)],
                            expected_code=1,
                            fragments=("symlink",),
                        )

            self.assertEqual(real_target.read_text(encoding="ascii"), "untouched")
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                ["dangling_link.cpp", "live_link.cpp", "real.cpp"],
            )

    def test_atomic_write_failure_leaves_zero_tmp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"synthetic-image-bytes")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            out_cpp = out_dir / "heap_aot.cpp"
            out_cpp.write_text("original", encoding="ascii")

            with (
                mock.patch.object(bink_heap_aot, "generate", return_value=SYNTHETIC_CPP),
                mock.patch.object(Path, "replace", side_effect=OSError("simulated replace failure")),
            ):
                self._assert_clean_failure(
                    [str(image), str(out_cpp)],
                    expected_code=1,
                    fragments=("simulated replace failure",),
                    check_subprocess=False,
                )

            self.assertEqual(out_cpp.read_text(encoding="ascii"), "original")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["heap_aot.cpp"])

    def test_argparse_errors_py314_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"synthetic")
            out_cpp = base / "heap_aot.cpp"

            for argv, frags in (
                ([], ("required",)),
                ([str(image)], ("required",)),
                ([str(image), str(out_cpp), "--entries", "0x1000"], ("expected 2 arguments",)),
                ([str(image), str(out_cpp), "--entries", "bad", "0x2000"], ("invalid",)),
                ([str(image), str(out_cpp), "--bogus-flag"], ("unrecognized",)),
            ):
                with self.subTest(argv=argv):
                    self._assert_clean_failure(
                        argv,
                        expected_code=None,
                        fragments=frags,
                    )


if __name__ == "__main__":
    unittest.main()
