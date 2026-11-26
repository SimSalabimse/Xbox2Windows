#!/usr/bin/env python3
"""Synthetic PE32+ CLI, deterministic scan, and error-handling suite for tools/pe_package_preflight.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "pe_package_preflight.py"
sys.path.insert(0, str(TOOLS_DIR))
import pe_package_preflight  # noqa: E402


IMAGE_BASE = 0x180000000
RAW_OFFSET = 0x200
SECTION_RVA = 0x1000
SECTION_SIZE = 0x1000


def _build_pe32_plus(
    *,
    imports: tuple[tuple[str, str, str | int], ...] = (),
    exports: tuple[tuple[str | None, str | None], ...] = (),
    machine: int = 0x8664,
) -> bytes:
    """Build a minimal synthetic PE32+ image with optional exports and normal/delay imports."""
    rdata = bytearray(SECTION_SIZE)
    cursor = 0

    def alloc(size: int, align: int = 4) -> int:
        nonlocal cursor
        cursor = (cursor + align - 1) & ~(align - 1)
        rva = SECTION_RVA + cursor
        if cursor + size > len(rdata):
            raise AssertionError("synthetic .rdata overflow")
        cursor += size
        return rva

    def put(rva: int, blob: bytes) -> None:
        offset = rva - SECTION_RVA
        rdata[offset:offset + len(blob)] = blob

    export_rva = export_size = 0
    if exports:
        code_rva = alloc(4, 4)
        put(code_rva, b"\xC3\x90\x90\x90")
        functions = []
        named_entries = []
        for idx, (symbol, forwarder) in enumerate(exports):
            if forwarder is not None:
                fwd_blob = forwarder.encode("ascii") + b"\0"
                fwd_rva = alloc(len(fwd_blob), 2)
                put(fwd_rva, fwd_blob)
                functions.append(fwd_rva)
            else:
                functions.append(code_rva)
            if symbol is not None:
                name_blob = symbol.encode("ascii") + b"\0"
                name_rva = alloc(len(name_blob), 2)
                put(name_rva, name_blob)
                named_entries.append((symbol, idx, name_rva))
        named_entries.sort(key=lambda item: item[0])

        eat_rva = alloc(4 * len(functions), 4)
        put(eat_rva, b"".join(struct.pack("<I", rva) for rva in functions))
        names_rva = alloc(max(4, 4 * len(named_entries)), 4)
        ordinals_rva = alloc(max(2, 2 * len(named_entries)), 2)
        if named_entries:
            put(names_rva, b"".join(struct.pack("<I", item[2]) for item in named_entries))
            put(ordinals_rva, b"".join(struct.pack("<H", item[1]) for item in named_entries))

        dll_name_rva = alloc(16, 2)
        put(dll_name_rva, b"synthetic.dll\0")
        export_rva = alloc(40, 4)
        put(
            export_rva,
            struct.pack(
                "<2I2H7I",
                0, 0, 0, 0, dll_name_rva, 1,
                len(functions), len(named_entries),
                eat_rva, names_rva, ordinals_rva,
            ),
        )
        export_size = 40

    module_rvas = {}
    for module in sorted({module for _, module, _ in imports}):
        raw = module.encode("ascii") + b"\0"
        rva = alloc(len(raw), 2)
        put(rva, raw)
        module_rvas[module] = rva

    hint_rvas = {}
    for kind, module, symbol in imports:
        if isinstance(symbol, int):
            continue
        raw = struct.pack("<H", 0) + symbol.encode("ascii") + b"\0"
        rva = alloc(len(raw), 2)
        put(rva, raw)
        hint_rvas[(kind, module, symbol)] = rva

    groups: dict[tuple[str, str], list[str | int]] = {}
    for kind, module, symbol in imports:
        groups.setdefault((kind, module), []).append(symbol)

    tables = {}
    for key in sorted(groups):
        values = []
        for sym in groups[key]:
            if isinstance(sym, int):
                values.append((1 << 63) | sym)
            else:
                values.append(hint_rvas[(key[0], key[1], sym)])
        blob = b"".join(struct.pack("<Q", v) for v in [*values, 0])
        lookup = alloc(len(blob), 8)
        put(lookup, blob)
        iat = alloc(len(blob), 8)
        put(iat, blob)
        tables[key] = (lookup, iat)

    normal_groups = sorted(k for k in groups if k[0] == "normal")
    delay_groups = sorted(k for k in groups if k[0] == "delay")
    normal_rva = normal_size = delay_rva = delay_size = 0

    if normal_groups:
        normal_size = 20 * (len(normal_groups) + 1)
        normal_rva = alloc(normal_size, 4)
        for idx, key in enumerate(normal_groups):
            lookup, iat = tables[key]
            put(normal_rva + idx * 20, struct.pack("<5I", lookup, 0, 0, module_rvas[key[1]], iat))

    if delay_groups:
        delay_size = 32 * (len(delay_groups) + 1)
        delay_rva = alloc(delay_size, 4)
        for idx, key in enumerate(delay_groups):
            lookup, iat = tables[key]
            put(delay_rva + idx * 32, struct.pack("<8I", 1, module_rvas[key[1]], 0, iat, lookup, 0, 0, 0))

    pe_off = 0x80
    buf = bytearray(RAW_OFFSET + SECTION_SIZE)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 60, pe_off)
    buf[pe_off:pe_off + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_off + 4, machine, 1, 0, 0, 0, 0xF0, 0x2022)
    opt = pe_off + 24
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        buf,
        opt,
        0x20B, 14, 0, 0, SECTION_SIZE, 0, 0, SECTION_RVA, IMAGE_BASE,
        0x1000, 0x200, 6, 0, 0, 0, 6, 0, 0, 0x2000, RAW_OFFSET,
        0, 3, 0, 0x100000, 0x1000, 0x100000, 0x1000, 0, 16,
    )
    dd = opt + 112
    if export_rva:
        struct.pack_into("<II", buf, dd + 0 * 8, export_rva, export_size)
    if normal_groups:
        struct.pack_into("<II", buf, dd + 1 * 8, normal_rva, normal_size)
    if delay_groups:
        struct.pack_into("<II", buf, dd + 13 * 8, delay_rva, delay_size)

    sec_off = opt + 0xF0
    struct.pack_into(
        "<8sIIIIIIHHI",
        buf,
        sec_off,
        b".rdata\0\0", SECTION_SIZE, SECTION_RVA, SECTION_SIZE, RAW_OFFSET, 0, 0, 0, 0, 0x40000040,
    )
    buf[RAW_OFFSET:RAW_OFFSET + SECTION_SIZE] = rdata
    return bytes(buf)


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = pe_package_preflight.main(argv)
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


def _make_synthetic_package(pkg: Path) -> None:
    sub = pkg / "sub"
    sub.mkdir()
    (pkg / "app.exe").write_bytes(
        _build_pe32_plus(
            imports=(
                ("normal", "helper.dll", "HelperInit"),
                ("delay", "helper.dll", 2),
                ("normal", "kernel32.dll", "ExitProcess"),
            )
        )
    )
    (sub / "helper.dll").write_bytes(
        _build_pe32_plus(
            imports=(("normal", "core.dll", "CoreTick"),),
            exports=(("HelperInit", None), (None, None)),
        )
    )
    (sub / "core.dll").write_bytes(
        _build_pe32_plus(exports=(("CoreTick", None),))
    )


class PePackagePreflightCliPositiveTests(unittest.TestCase):
    def test_stdout_and_atomic_output_with_zero_leftover_temps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg = base / "package"
            pkg.mkdir()
            _make_synthetic_package(pkg)

            # 1) Stdout mode (in-process + subprocess).
            code, out, err = _run_main([str(pkg), "app.exe"])
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            report = json.loads(out)
            self.assertEqual(report["schema"], 1)
            self.assertEqual(report["entry"], "app.exe")
            self.assertEqual(report["summary"]["images_scanned"], 3)
            self.assertEqual(report["summary"]["images_reachable"], 3)
            self.assertEqual(report["summary"]["static_imports"], 4)
            self.assertEqual(report["summary"]["bundled_failures"], 0)
            self.assertEqual(report["summary"]["external_unverified"], 1)
            self.assertEqual(report["ignored_images"], [])
            self.assertFalse(report["reachable"]["sub/helper.dll"]["conditional"])
            self.assertEqual(
                report["reachable"]["sub/core.dll"]["path"],
                ["app.exe", "sub/helper.dll", "sub/core.dll"],
            )

            proc = _run_subprocess(str(pkg), "app.exe")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(json.loads(proc.stdout), report)

            # 2) Atomic --output creation into nested non-existent directory, 0 leftover temps.
            nested_out = base / "reports" / "nested" / "preflight.json"
            proc_out = _run_subprocess(str(pkg), "app.exe", "--output", str(nested_out))
            self.assertEqual(proc_out.returncode, 0, proc_out.stderr)
            self.assertEqual(proc_out.stdout, "")
            self.assertEqual(proc_out.stderr, "")
            self.assertEqual(json.loads(nested_out.read_text(encoding="utf-8")), report)
            self.assertEqual(
                sorted(p.name for p in nested_out.parent.iterdir()),
                ["preflight.json"],
            )

            # 3) Atomic replacement of pre-existing output file, 0 leftover temps.
            nested_out.write_text("stale content", encoding="utf-8")
            code_rep, out_rep, err_rep = _run_main([str(pkg), "app.exe", "--output", str(nested_out)])
            self.assertEqual(code_rep, 0, err_rep)
            self.assertEqual(out_rep, "")
            self.assertEqual(err_rep, "")
            self.assertEqual(json.loads(nested_out.read_text(encoding="utf-8")), report)
            self.assertEqual(
                sorted(p.name for p in nested_out.parent.iterdir()),
                ["preflight.json"],
            )

            # 4) sys.argv forwarding when argv=None and __main__ entrypoint.
            cli_argv = [str(TOOL), str(pkg), "app.exe"]
            with mock.patch.object(sys, "argv", cli_argv):
                code_none, out_none, err_none = _run_main(None)
            self.assertEqual(code_none, 0, err_none)
            self.assertEqual(json.loads(out_none), report)

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(json.loads(stdout.getvalue()), report)

    def test_unsorted_rglob_deterministic_order_and_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a_dir = root / "a_dir"
            b_dir = root / "B_dir"
            a_dir.mkdir()
            b_dir.mkdir()

            entry = a_dir / "first.exe"
            second = a_dir / "second.dll"
            huge = b_dir / "huge.dll"
            entry.write_bytes(b"a" * 100)
            second.write_bytes(b"b" * 100)
            huge.write_bytes(b"z" * 500)

            # Unsorted rglob puts B_dir/huge.dll first (and ASCII sort puts 'B' < 'a');
            # deterministic casefold sort visits a_dir/first.exe, a_dir/second.dll before B_dir/huge.dll.
            unsorted_paths = [huge, second, entry]
            with (
                mock.patch.object(Path, "rglob", return_value=iter(unsorted_paths)),
                mock.patch.object(pe_package_preflight, "MAX_IMAGE_BYTES", 300),
                mock.patch.object(pe_package_preflight, "MAX_IMAGES", 1),
            ):
                with self.assertRaisesRegex(ValueError, "1024 PE image inspection limit"):
                    pe_package_preflight._paths(root.resolve())

            with (
                mock.patch.object(Path, "rglob", return_value=iter(unsorted_paths)),
                mock.patch.object(pe_package_preflight, "MAX_IMAGE_BYTES", 300),
                mock.patch.object(pe_package_preflight, "MAX_TOTAL_BYTES", 150),
            ):
                with self.assertRaisesRegex(ValueError, "2 GiB PE inspection limit"):
                    pe_package_preflight._paths(root.resolve())

        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            _make_synthetic_package(pkg)
            forward = [pkg / "app.exe", pkg / "sub" / "core.dll", pkg / "sub" / "helper.dll"]
            reversed_order = list(reversed(forward))
            with mock.patch.object(Path, "rglob", return_value=iter(forward)):
                code_a, out_a, err_a = _run_main([str(pkg), "app.exe"])
            with mock.patch.object(Path, "rglob", return_value=iter(reversed_order)):
                code_b, out_b, err_b = _run_main([str(pkg), "app.exe"])
            self.assertEqual(code_a, 0, err_a)
            self.assertEqual(code_b, 0, err_b)
            self.assertEqual(out_a, out_b)


class PePackagePreflightCliNegativeTests(unittest.TestCase):
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

    def test_root_is_dir_and_non_empty_entry_validated_before_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg = base / "package"
            pkg.mkdir()
            _make_synthetic_package(pkg)

            not_a_dir = base / "regular.txt"
            not_a_dir.write_text("not a directory", encoding="utf-8")

            with mock.patch.object(
                pe_package_preflight,
                "inspect",
                side_effect=AssertionError("inspect called for invalid package/entry"),
            ):
                self._assert_clean_failure(
                    [str(base / "missing_pkg"), "app.exe"],
                    expected_code=1,
                    fragments=("pe_package_preflight:",),
                )
                self._assert_clean_failure(
                    [str(not_a_dir), "app.exe"],
                    expected_code=1,
                    fragments=("pe_package_preflight:", "Package root must be a directory"),
                )
                for blank_entry in ("", "   "):
                    with self.subTest(entry=blank_entry):
                        self._assert_clean_failure(
                            [str(pkg), blank_entry],
                            expected_code=1,
                            fragments=("pe_package_preflight:", "Entry must be"),
                        )

            for blank_entry in ("", "   "):
                with self.subTest(direct_entry=blank_entry):
                    with self.assertRaisesRegex(ValueError, "Entry must be"):
                        pe_package_preflight.inspect(pkg, blank_entry)

    def test_output_collision_dir_inside_pkg_rejected_before_temp_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg = base / "package"
            pkg.mkdir()
            _make_synthetic_package(pkg)

            file_parent = base / "collision.txt"
            file_parent.write_text("keep-me", encoding="utf-8")
            out_dir = base / "existing_dir"
            out_dir.mkdir()

            cases = (
                ("parent_regular_file", file_parent / "report.json", ("not a directory",)),
                ("ancestor_regular_file", file_parent / "nested" / "report.json", ("not a directory",)),
                ("output_is_dir", out_dir, ("directory",)),
                ("output_empty_string", Path(""), ("directory",)),
                ("output_inside_pkg_root", pkg, ("outside the inspected package",)),
                ("output_inside_pkg_file", pkg / "report.json", ("outside the inspected package",)),
                ("output_inside_pkg_nested", pkg / "new_sub" / "report.json", ("outside the inspected package",)),
            )
            for label, target, frags in cases:
                with self.subTest(case=label):
                    with (
                        mock.patch.object(
                            pe_package_preflight,
                            "inspect",
                            side_effect=AssertionError(f"inspect called for {label}"),
                        ),
                        mock.patch.object(
                            pe_package_preflight.tempfile,
                            "TemporaryDirectory",
                            side_effect=AssertionError(f"temp staging called for {label}"),
                        ),
                    ):
                        argv = [str(pkg), "app.exe", "--output", "" if label == "output_empty_string" else str(target)]
                        self._assert_clean_failure(
                            argv,
                            expected_code=1,
                            fragments=("pe_package_preflight:", *frags),
                        )

            self.assertEqual(file_parent.read_text(encoding="utf-8"), "keep-me")
            self.assertEqual(list(out_dir.iterdir()), [])
            self.assertFalse((pkg / "report.json").exists())
            self.assertFalse((pkg / "new_sub").exists())

    def test_output_symlink_rejected_before_temp_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg = base / "package"
            pkg.mkdir()
            _make_synthetic_package(pkg)

            out_dir = base / "out_dir"
            out_dir.mkdir()
            real_target = out_dir / "real.json"
            real_target.write_text("untouched", encoding="utf-8")
            live_link = out_dir / "live_link.json"
            dangling_link = out_dir / "dangling_link.json"
            try:
                live_link.symlink_to(real_target)
                dangling_link.symlink_to(out_dir / "nonexistent.json")
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                self.skipTest("this Windows account cannot create symlinks (WinError 1314)")

            for label, target in (("live_symlink", live_link), ("dangling_symlink", dangling_link)):
                with self.subTest(case=label):
                    with (
                        mock.patch.object(
                            pe_package_preflight,
                            "inspect",
                            side_effect=AssertionError(f"inspect called for {label}"),
                        ),
                        mock.patch.object(
                            pe_package_preflight.tempfile,
                            "TemporaryDirectory",
                            side_effect=AssertionError(f"temp staging called for {label}"),
                        ),
                    ):
                        self._assert_clean_failure(
                            [str(pkg), "app.exe", "--output", str(target)],
                            expected_code=1,
                            fragments=("pe_package_preflight:", "symlink"),
                        )

            self.assertEqual(real_target.read_text(encoding="utf-8"), "untouched")
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                ["dangling_link.json", "live_link.json", "real.json"],
            )

    def test_temp_staging_replace_failure_leaves_zero_leftover_temps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg = base / "package"
            pkg.mkdir()
            _make_synthetic_package(pkg)

            out_dir = base / "out_dir"
            out_dir.mkdir()
            out_file = out_dir / "report.json"
            out_file.write_text("original", encoding="utf-8")

            with mock.patch.object(Path, "replace", side_effect=OSError("simulated replace failure")):
                self._assert_clean_failure(
                    [str(pkg), "app.exe", "--output", str(out_file)],
                    expected_code=1,
                    fragments=("pe_package_preflight:", "simulated replace failure"),
                    check_subprocess=False,
                )

            self.assertEqual(out_file.read_text(encoding="utf-8"), "original")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["report.json"])

    def test_invalid_pe_and_missing_entry_do_not_create_output_dir_or_temps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg = base / "package"
            pkg.mkdir()
            (pkg / "bad.exe").write_bytes(b"not a valid PE")
            absent_out = base / "should_not_exist" / "report.json"

            self._assert_clean_failure(
                [str(pkg), "bad.exe", "--output", str(absent_out)],
                expected_code=1,
                fragments=("pe_package_preflight:",),
            )
            self.assertFalse(absent_out.parent.exists())

            _make_synthetic_package(pkg)
            self._assert_clean_failure(
                [str(pkg), "missing.exe", "--output", str(absent_out)],
                expected_code=1,
                fragments=("pe_package_preflight:",),
            )
            self.assertFalse(absent_out.parent.exists())

    def test_argparse_errors_py314_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            _make_synthetic_package(pkg)
            cases = (
                ([], ("required",)),
                ([str(pkg)], ("required",)),
                ([str(pkg), "app.exe", "--bogus-flag"], ("unrecognized",)),
                ([str(pkg), "app.exe", "--output"], ("expected one argument",)),
            )
            for argv, frags in cases:
                with self.subTest(argv=argv):
                    self._assert_clean_failure(
                        argv,
                        expected_code=None,
                        fragments=frags,
                    )


if __name__ == "__main__":
    unittest.main()
