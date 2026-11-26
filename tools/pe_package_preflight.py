"""Inspect source-built PE package imports offline, without a Windows host or code execution.

This deliberately answers *only* which statically imported symbols are fulfilled
by DLLs within the supplied directory. Host DLLs and dynamic loads are unknown.
"""

import argparse
from collections import Counter, defaultdict
import heapq
import json
from pathlib import Path
import struct
import sys
import tempfile

import pefile

from pe_inventory import _export_table, parse_image, resolve_desktop


MAX_IMAGES = 1024
MAX_IMAGE_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
X86_MACHINES = {"IMAGE_FILE_MACHINE_I386", "IMAGE_FILE_MACHINE_AMD64"}


def _relative(root, path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or path.is_symlink():
        raise ValueError("PE path escapes package root or uses a symlink")
    if not path.is_file():
        raise ValueError("PE input is not a regular file")
    return resolved.relative_to(root).as_posix()


def _paths(root):
    result = {}
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda p: (p.as_posix().casefold(), p.as_posix())):
        if path.suffix.casefold() not in (".dll", ".exe"):
            continue
        if path.is_symlink():
            raise ValueError("A PE candidate is a symlink")
        if not path.is_file():
            continue
        name = _relative(root, path)
        size = path.stat().st_size
        if size > MAX_IMAGE_BYTES:
            raise ValueError("PE image exceeds 512 MiB inspection limit")
        total_bytes += size
        if total_bytes > MAX_TOTAL_BYTES:
            raise ValueError("Package exceeds 2 GiB PE inspection limit")
        result[name] = path
        if len(result) > MAX_IMAGES:
            raise ValueError("Package exceeds 1024 PE image inspection limit")
    return result


def _classify(resolution):
    chain, status = resolution["chain"], resolution["status"]
    if not chain[0].get("package"):
        return "external_unverified"
    if status == "export_present":
        return "bundled_export_present"
    if status == "provider_not_found" and not chain[-1].get("package"):
        return "forwarded_external_unverified"
    if status == "ambiguous_package_basename":
        return "ambiguous_package_basename"
    return "bundled_" + status


def _check_x86_machines(resolution, importer_machine, images):
    """A matching export cannot bind an I386 image to an AMD64 DLL or vice versa.

    Other machine combinations are outside this check: hybrid ARM64EC/ARM64X
    images cannot be classified by an equality comparison of COFF machine tags.
    """
    if importer_machine not in X86_MACHINES:
        return resolution
    for index, step in enumerate(resolution["chain"]):
        if not step.get("package") or step["provider"] not in images:
            continue
        machine = images[step["provider"]][0]["machine"]
        if machine in X86_MACHINES and machine != importer_machine:
            chain = resolution["chain"][:index] + [dict(step,
                importer_machine=importer_machine, provider_machine=machine)]
            return dict(status="machine_mismatch", chain=chain)
    return resolution


def summarize(images, entry):
    """Resolve packaged imports using the existing strict PE export/forwarder reader.

    images maps relative paths to (image metadata, (kind, DLL, symbol) imports).
    This pure function enables synthetic tests without a Windows loader.
    """
    if entry not in images:
        raise ValueError("Entry is not an image in the package")
    by_name = defaultdict(list)
    exports = {}
    for name, (info, _) in sorted(images.items()):
        by_name[Path(name).name.casefold()].append(name)
        exports[name] = _export_table(info["exports"])
    def packaged(module):
        paths = by_name[module.casefold()]
        return paths, exports[paths[0]] if len(paths) == 1 else None

    per_image = defaultdict(list)
    for path, (_, rows) in sorted(images.items()):
        for kind, dll, symbol in sorted(set(rows)):
            resolution = resolve_desktop(dll.casefold(), symbol, Path(path).name.casefold(),
                                         {}, lambda _dll: None, packaged)
            resolution = _check_x86_machines(resolution, images[path][0]["machine"], images)
            per_image[path].append(dict(image=path, kind=kind, dll=dll, symbol=symbol,
                                        status=_classify(resolution),
                                        resolution_status=resolution["status"],
                                        chain=resolution["chain"]))

    # Prefer an unconditional path over a delay-import path, even if longer.
    queue = [(False, 0, (entry,), ())]
    reached = {}
    active_rows = []
    while queue:
        conditional, depth, path, reasons = heapq.heappop(queue)
        here = path[-1]
        if here in reached:
            continue
        reached[here] = dict(path=list(path), conditional=conditional, reasons=list(reasons))
        for row in per_image[here]:
            active_rows.append(row)
            provider_path, edge_reasons = path, reasons
            kind = row["kind"]
            for step in row["chain"]:
                provider = step["provider"]
                if step.get("package") and provider and provider not in provider_path:
                    edge = kind if provider_path == path else "forwarder"
                    provider_path += (provider,)
                    edge_reasons += (edge,)
                    distance = depth + len(provider_path) - len(path)
                    heapq.heappush(queue, (conditional or kind == "delay", distance,
                                           provider_path, edge_reasons))

    counts = Counter(row["status"] for row in active_rows)
    fatal = {"ambiguous_package_basename", "bundled_export_not_found", "bundled_forwarder_cycle",
             "bundled_invalid_forwarder", "bundled_provider_not_found"}
    local_failures = sum(count for status, count in counts.items()
                         if status in fatal or (status.startswith("bundled_") and status != "bundled_export_present"))
    return dict(schema=1, entry=entry,
                scope="Static package import/export evidence only. External imports, Windows API sets, "
                      "KnownDLLs, runtime ABI behavior, dynamic loads, and executable startup are unverified. "
                      "I386/AMD64 machine mismatches are checked; other machine combinations are unverified.",
                images={name: dict(machine=info["machine"], sha256=info["sha256"],
                                   exports=len(info["exports"]), warnings=len(info["parse_warnings"]))
                        for name, (info, _) in sorted(images.items())},
                reachable=reached, imports=active_rows,
                summary=dict(images_scanned=len(images), images_reachable=len(reached),
                             static_imports=len(active_rows), normal=sum(x["kind"] == "normal" for x in active_rows),
                             delay=sum(x["kind"] == "delay" for x in active_rows),
                             status_counts=dict(sorted(counts.items())), bundled_failures=local_failures,
                             external_unverified=sum(v for k, v in counts.items() if "external_unverified" in k)),
                ignored_images=sorted(set(images) - set(reached)))


def _validate_entry(entry):
    if not isinstance(entry, str) or not entry.strip():
        raise ValueError("Entry must be a non-empty relative path to an .exe inside the package")


def _validate_output(output, root):
    if output.is_symlink():
        raise ValueError(f"Output path must not be a symlink: {output}")
    for parent in output.parents:
        if parent.is_file() or ((parent.exists() or parent.is_symlink()) and not parent.is_dir()):
            raise NotADirectoryError(f"Output parent is not a directory: {parent}")
    if output.resolve().is_relative_to(root):
        raise ValueError("Write reports outside the inspected package")
    if output.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {output}")


def inspect(root, entry):
    _validate_entry(entry)
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Package root must be a directory")
    entry_path = root / entry
    entry_name = _relative(root, entry_path)
    if not entry_name.casefold().endswith(".exe"):
        raise ValueError("Entry must be an .exe inside the package")
    paths = _paths(root)
    if entry_name not in paths:
        raise ValueError("Entry is not among scanned PE files")
    images = {name: parse_image(path) for name, path in sorted(paths.items())}
    return summarize(images, entry_name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Directory containing EXE and DLL files")
    parser.add_argument("entry", help="Relative path to the source-built EXE")
    parser.add_argument("--output", type=Path, help="Optional JSON output path outside the inspected package")
    args = parser.parse_args(argv)
    try:
        root = args.package.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Package root must be a directory")
        _validate_entry(args.entry)
        if args.output is not None:
            _validate_output(args.output, root)
        report = inspect(root, args.entry)
        output = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Replace the directory entry instead of writing through a possible hard link.
            with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
                report_path = Path(directory) / "report.json"
                report_path.write_text(output, encoding="utf-8")
                report_path.replace(args.output)
        else:
            print(output, end="")
    except (OSError, ValueError, KeyError, struct.error, pefile.PEFormatError) as exc:
        parser.exit(1, f"pe_package_preflight: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
