"""Translate the counter updates following two upscaler working-set queries."""
import argparse
from pathlib import Path
import struct
import tempfile

import pefile

from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, UPSCALER, checked_image, lower, region


def generate(image: Path, package: bool) -> str:
    source = [INTEGER_HEADER]
    with (checked_image(image, UPSCALER) if package else pefile.PE(str(image), fast_load=True)) as pe:
        for name, start, end, sample_offset in (("create_delta", 0x74d8, 0x74f8, 0x78),
                                                ("release_delta", 0x75b7, 0x75d7, 0x28)):
            if package:
                initial = ["rsi = load<word>(input);", "rax = load<uint32_t>(input + 16);",
                           f"store<word>(rsp + {sample_offset}, load<word>(input + 8));",
                           "rbx = counters - 0x40;"]
                if name == "create_delta":
                    initial.append("store<uint32_t>(rsp + 0x30, load<uint32_t>(input + 20));")
                else:
                    initial.extend(["r14 = rsp;", "store<uint8_t>(r14 + 0x18, load<uint8_t>(input + 20));"])
                body = lower(region(pe, start, end), initial, {}, {}, {end: "return 0;"})
            else:
                (bounds,) = function_ranges(pe, export_rva(pe, name))
                body = lower(region(pe, *bounds), ["rcx = input; rdx = counters;"], {}, {}, {})
            source.append(f'extern "C" __declspec(dllexport) word {name}(word input, word counters) {{\n'
                          + body + "\n}\n")
    return "\n".join(source)


def _validate_paths(image: Path, output: Path, parser: argparse.ArgumentParser) -> None:
    if image.is_dir():
        raise IsADirectoryError(f"image path is a directory: {image}")
    if not image.is_file():
        raise FileNotFoundError(f"image file not found: {image}")
    if output.suffix.lower() != ".cpp":
        parser.error("choose a distinct .cpp output")
    if output.is_symlink():
        raise ValueError(f"output path must not be a symlink: {output}")
    if output.is_dir():
        raise IsADirectoryError(f"output path is a directory: {output}")
    for parent in output.parents:
        if parent.is_file() or ((parent.exists() or parent.is_symlink()) and not parent.is_dir()):
            raise NotADirectoryError(f"output parent is not a directory: {parent}")
    if output.resolve() == image.resolve() or (output.exists() and image.samefile(output)):
        parser.error("choose a distinct .cpp output")


def _write_atomic(path: Path, content: str) -> None:
    """Write ASCII content to a sibling temp file and atomically replace path, leaving 0 temp files on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
        tmp_path.replace(path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args(argv)
    try:
        _validate_paths(args.image, args.output, parser)
        _write_atomic(args.output, generate(args.image, args.package))
    except (OSError, ValueError, struct.error, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
