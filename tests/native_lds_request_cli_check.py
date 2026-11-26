"""compile_lds_request.main() writes a replay for a source-built compute request.

The program is s_endpgm, the root signature is an empty RTS0 container, and state block A
asks for four user SGPRs and a 1x1x1 group. No GPU and no DXC. tests/native_compute_clear_check.py
write_request hashes the 14cs revision as one blob and enables TG_SIZE_EN, so the slot 67 loader
refuses it; this check names the files with gfx2hlsl.compute_key instead.

A child interpreter stands in for the compiler: sitecustomize.py writes the bytecode and exits.
"""
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "startup"))
import compile_lds_request as compile_lds
import gfx2hlsl as gfx

S_ENDPGM = 0xBF810000
MISMATCH = "compute request files do not match the key in their name"


def empty_root():
    """DXBC container with one RTS0 part and no root parameters."""
    rts0 = struct.pack("<6I", 1, 0, 24, 0, 24, 0)
    part = b"RTS0" + struct.pack("<I", len(rts0)) + rts0
    return b"DXBC" + bytes(16) + struct.pack("<III", 1, 32 + 4 + len(part), 1) + struct.pack("<I", 36) + part


def write_request(folder, key=None):
    """Write req_<key>.c67 and its siblings. key defaults to compute_key of the bytes."""
    code = struct.pack("<I", S_ENDPGM)
    data = b""
    root = empty_root()
    # USER_SGPR count is bits 5:1. Four user SGPRs cover s0-s3, which the compute entry reserves.
    sa = struct.pack("<7I", 0, 0, 0, 1 | (1 << 16), 1, 0, 4 << 1)
    sb = b""
    desc = struct.pack("<12Q", 0, 0, len(sa), 0, len(sb), 0, len(code), 0, len(data), 0, 0, 0)
    actual = gfx.compute_key(code, data, root, desc, sa, sb)
    marker = folder / f"req_{(actual if key is None else key):016x}.c67"
    parts = {".c67": code, ".data": data, ".root": root, ".desc": desc, ".sa": sa, ".sb": sb}
    for suffix, blob in parts.items():
        marker.with_suffix(suffix).write_bytes(blob)
    return marker, actual


def install_fake_compiler(folder):
    """Return (compiler, pythonpath) for a child that writes DXBC to its bytecode argument.

    The tool runs [compiler, sdk, cache, cs_6_6, hlsl, bytecode]. The interpreter drops its own
    path, so the stand-in sees [sdk, cache, cs_6_6, hlsl, bytecode]. SystemExit while site is
    importing is fatal, so the stand-in leaves with os._exit.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sitecustomize.py").write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "if len(sys.argv) == 5 and sys.argv[2] == 'cs_6_6':\n"
        "    Path(sys.argv[4]).write_bytes(b'DXBC')\n"
        "    os._exit(0)\n",
        encoding="utf-8",
    )
    return Path(sys.executable), str(folder)


def invoke(marker, output, compiler, wave, pythonpath):
    output.mkdir(parents=True, exist_ok=True)
    (output / "sdk").mkdir(exist_ok=True)
    (output / "cache").mkdir(exist_ok=True)
    argv = [
        "compile_lds_request.py",
        str(marker),
        str(output),
        "--compiler",
        str(compiler),
        "--sdk",
        str(output / "sdk"),
        "--cache",
        str(output / "cache"),
        "--rw-image",
        "0",
        "0",
        "--wave-size",
        str(wave),
    ]
    previous_argv = sys.argv[:]
    previous_path = os.environ.get("PYTHONPATH")
    sys.argv = argv
    if pythonpath is not None:
        os.environ["PYTHONPATH"] = pythonpath if previous_path is None else pythonpath + os.pathsep + previous_path
    stdout = io.StringIO()
    try:
        with redirect_stdout(stdout):
            compile_lds.main()
    finally:
        sys.argv = previous_argv
        if pythonpath is not None:
            if previous_path is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_path
    lines = [line for line in stdout.getvalue().splitlines() if line]
    if len(lines) != 1:
        raise AssertionError(f"expected one replay path, got {lines!r}")
    return Path(lines[0])


def assert_replay(dest, key, wave):
    replay_path = dest / "replay.json"
    manifest_path = dest / f"{key:016x}.txt"
    hlsl_path = dest / f"{key:016x}.cs.hlsl"
    bytecode_path = dest / f"{key:016x}.cs.cso"
    assert replay_path.is_file(), replay_path
    assert manifest_path.is_file(), manifest_path
    replay_bytes = replay_path.read_bytes()
    assert dest.name == f"{key:016x}-{hashlib.sha256(replay_bytes).hexdigest()}"
    metadata = json.loads(replay_bytes)
    assert metadata["schema"] == 1
    assert metadata["profile"] == "cs_6_6"
    assert metadata["wave_size"] == wave
    assert metadata["rw_image"] == [0, 0]
    assert metadata["request"] == f"request/req_{key:016x}.c67"
    recorded = metadata["sha256"]
    for relative in (f"{key:016x}.txt", f"{key:016x}.cs.hlsl", f"{key:016x}.cs.cso", f"request/req_{key:016x}.c67"):
        assert relative in recorded, relative
    for relative, digest in recorded.items():
        assert hashlib.sha256((dest / relative).read_bytes()).hexdigest() == digest, relative
    hlsl = hlsl_path.read_text(encoding="utf-8")
    assert f"[WaveSize({wave})]\n[numthreads(" in hlsl
    assert f"[WaveSize({64 if wave == 32 else 32})]" not in hlsl
    assert manifest_path.read_text(encoding="utf-8").startswith("x2w-manifest 2\nstage cs\n")
    assert bytecode_path.read_bytes() == b"DXBC"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        compiler, pythonpath = install_fake_compiler(root / "compiler")
        source = root / "source"
        source.mkdir()
        marker, key = write_request(source)
        bad = source / "bad"
        bad.mkdir()
        mismatched, _ = write_request(bad, key=key ^ 1)
        try:
            invoke(mismatched, root / "refused", compiler, 32, pythonpath)
        except gfx.Unsupported as error:
            assert str(error) == MISMATCH, error
        else:
            raise AssertionError("a request whose key does not match its files was accepted")
        assert not any((root / "refused").rglob("replay.json"))
        print("mismatched key refused:", MISMATCH)

        for wave in (32, 64):
            dest = invoke(marker, root / f"out{wave}", compiler, wave, pythonpath)
            assert_replay(dest, key, wave)
            print(f"wave {wave}: replay, manifest, [WaveSize({wave})] and SHA-256 matched")

        folder = marker.parent
        base = gfx.load_compute_request(str(folder), key)
        assert type(base) is gfx.ComputePipeline
        built = gfx.load_compute_request(
            str(folder), key, pipeline_type=partial(compile_lds.ComputePipeline, feedback=(("table", 0), 0)))
        assert type(built) is compile_lds.ComputePipeline
        assert built.feedback == (("table", 0), 0)
        print("default pipeline is ComputePipeline; the LDS subclass received feedback")


if __name__ == "__main__":
    main()
