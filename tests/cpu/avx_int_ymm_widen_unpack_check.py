"""Differential, boundary, mutation, and rejection checks for 256-bit AVX2 extend and unpack ops."""

import ctypes
import os
from pathlib import Path
import random
import struct
import sys

from iced_x86 import Code, Decoder, Formatter, FormatterSyntax, Instruction, MemoryOperand, Register as R
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from avx_int_ymm_minmax_check import encode_seq, first_bad_word, invoke, load  # noqa: E402
from avx_int_ymm_pack_check import host_has_avx2  # noqa: E402
from cpu_avx_int_aot import lift_code  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

EXTEND_SPECS = {
    # export: (signed, src_width_bytes, dst_width_bytes)
    "op_vpmovsxbw_ymm": (True, 1, 2),
    "op_vpmovsxbd_ymm": (True, 1, 4),
    "op_vpmovsxbq_ymm": (True, 1, 8),
    "op_vpmovsxwd_ymm": (True, 2, 4),
    "op_vpmovsxwq_ymm": (True, 2, 8),
    "op_vpmovsxdq_ymm": (True, 4, 8),
    "op_vpmovzxbw_ymm": (False, 1, 2),
    "op_vpmovzxbd_ymm": (False, 1, 4),
    "op_vpmovzxbq_ymm": (False, 1, 8),
    "op_vpmovzxwd_ymm": (False, 2, 4),
    "op_vpmovzxwq_ymm": (False, 2, 8),
    "op_vpmovzxdq_ymm": (False, 4, 8),
}
UNPACK_EXPORTS = (
    "op_vpunpckldq_ymm",
    "op_vpunpckhdq_ymm",
    "op_vpunpcklqdq_ymm",
    "op_vpunpckhqdq_ymm",
)
BASE_EXPORTS = tuple(EXTEND_SPECS) + UNPACK_EXPORTS
SPECIAL_EXPORTS = (
    "op_vpmovsxbd_ymm_stale_dst",
    "op_vpmovsxbw_ymm_alias",
    "op_vpmovzxdq_ymm_alias",
    "op_vpunpckldq_ymm_dst_src1",
    "op_vpunpckhqdq_ymm_dst_src2",
    "op_vpunpckhdq_ymm_all",
    "op_vpmovsxwd_ymm_then_xmm_vpunpcklqdq",
    "op_chain_ymm_widen_unpack",
)
EXPORTS = BASE_EXPORTS + SPECIAL_EXPORTS

_FMT_SIGNED = {1: "b", 2: "h", 4: "i", 8: "q"}
_FMT_UNSIGNED = {1: "B", 2: "H", 4: "I", 8: "Q"}


# --- Independent Python reference: cross-lane XMM->YMM extend & per-128-bit-lane unpack ---

def _ref_extend(is_signed, src_w, dst_w, a):
    count = 32 // dst_w
    in_fmt = (_FMT_SIGNED if is_signed else _FMT_UNSIGNED)[src_w]
    out_fmt = (_FMT_SIGNED if is_signed else _FMT_UNSIGNED)[dst_w]
    vals = struct.unpack(f"<{count}{in_fmt}", a[:count * src_w])
    return struct.pack(f"<{count}{out_fmt}", *vals)


def _unpack_half(kind, a16, b16):
    if kind in ("ldq", "hdq"):
        xs = struct.unpack("<4I", a16)
        ys = struct.unpack("<4I", b16)
        off = 0 if kind == "ldq" else 2
        return struct.pack("<4I", xs[off], ys[off], xs[off + 1], ys[off + 1])
    if kind in ("lqdq", "hqdq"):
        xs = struct.unpack("<2Q", a16)
        ys = struct.unpack("<2Q", b16)
        idx = 0 if kind == "lqdq" else 1
        return struct.pack("<2Q", xs[idx], ys[idx])
    raise ValueError(kind)


def _ref_unpack(kind, a, b):
    return _unpack_half(kind, a[:16], b[:16]) + _unpack_half(kind, a[16:32], b[16:32])


def reference(name, a, b):
    if name in EXTEND_SPECS:
        return _ref_extend(*EXTEND_SPECS[name], a)
    if name == "op_vpmovsxbd_ymm_stale_dst":
        return _ref_extend(True, 1, 4, a)
    if name == "op_vpmovsxbw_ymm_alias":
        return _ref_extend(True, 1, 2, a)
    if name == "op_vpmovzxdq_ymm_alias":
        return _ref_extend(False, 4, 8, a)
    if name in ("op_vpunpckldq_ymm", "op_vpunpckldq_ymm_dst_src1"):
        return _ref_unpack("ldq", a, b)
    if name == "op_vpunpckhdq_ymm":
        return _ref_unpack("hdq", a, b)
    if name == "op_vpunpckhdq_ymm_all":
        return _ref_unpack("hdq", a, a)
    if name == "op_vpunpcklqdq_ymm":
        return _ref_unpack("lqdq", a, b)
    if name in ("op_vpunpckhqdq_ymm", "op_vpunpckhqdq_ymm_dst_src2"):
        return _ref_unpack("hqdq", a, b)
    if name == "op_vpmovsxwd_ymm_then_xmm_vpunpcklqdq":
        t = _ref_extend(True, 2, 4, a)
        return _unpack_half("lqdq", t[:16], b[:16]) + b"\x00" * 16
    if name == "op_chain_ymm_widen_unpack":
        t3 = _ref_extend(True, 2, 4, a)
        t4 = _ref_extend(False, 1, 4, b)
        return _ref_unpack("ldq", t3, t4)
    raise KeyError(name)


def all_pairs():
    """Construct sign-boundary vectors with non-zero ignored XMM/YMM upper bytes plus random."""
    pairs = []
    b_edges = (0x00, 0x01, 0x7F, 0x80, 0x81, 0xFE, 0xFF)
    w_edges = (0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF)
    d_edges = (0x00000000, 0x00000001, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF)

    for shift in range(len(b_edges)):
        a = bytes(b_edges[(i + shift) % len(b_edges)] for i in range(32))
        b = bytes(b_edges[(3 * i + shift + 1) % len(b_edges)] for i in range(32))
        pairs.append((a, b))
    for shift in range(len(w_edges)):
        a = struct.pack("<16H", *(w_edges[(i + shift) % len(w_edges)] for i in range(16)))
        b = struct.pack("<16H", *(w_edges[(3 * i + shift + 2) % len(w_edges)] for i in range(16)))
        pairs.append((a, b))
    for shift in range(len(d_edges)):
        a = struct.pack("<8I", *(d_edges[(i + shift) % len(d_edges)] for i in range(8)))
        b = struct.pack("<8I", *(d_edges[(3 * i + shift + 2) % len(d_edges)] for i in range(8)))
        pairs.append((a, b))

    # Distinct non-zero high bytes in xmm (bytes 4..15) and upper ymm lane (bytes 16..31).
    pairs.append((bytes(range(0x10, 0x30)), bytes(range(0xA0, 0xC0))))
    pairs.append((bytes(0x80 + i for i in range(32)), bytes(0x7F - i for i in range(32))))
    pairs.append((bytes(32), bytes(32)))
    pairs.append((b"\xff" * 32, b"\xff" * 32))
    pairs.append((b"\x80" * 16 + b"\x55" * 16, b"\x7f" * 16 + b"\xaa" * 16))

    rng = random.Random(0x57DE9)
    pairs += [(rng.randbytes(32), rng.randbytes(32)) for _ in range(160)]
    return pairs


def bind(path, prefix):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


def check_translator():
    ret = Instruction.create(Code.RETNQ)
    ymm_loads = (load(R.YMM1, R.RCX), load(R.YMM2, R.RDX))
    xmm_loads = (
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RCX)),
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM2, MemoryOperand(R.RDX)),
    )

    extend_codes = {
        "VPMOVSXBW": Code.VEX_VPMOVSXBW_YMM_XMMM128,
        "VPMOVSXBD": Code.VEX_VPMOVSXBD_YMM_XMMM64,
        "VPMOVSXBQ": Code.VEX_VPMOVSXBQ_YMM_XMMM32,
        "VPMOVSXWD": Code.VEX_VPMOVSXWD_YMM_XMMM128,
        "VPMOVSXWQ": Code.VEX_VPMOVSXWQ_YMM_XMMM64,
        "VPMOVSXDQ": Code.VEX_VPMOVSXDQ_YMM_XMMM128,
        "VPMOVZXBW": Code.VEX_VPMOVZXBW_YMM_XMMM128,
        "VPMOVZXBD": Code.VEX_VPMOVZXBD_YMM_XMMM64,
        "VPMOVZXBQ": Code.VEX_VPMOVZXBQ_YMM_XMMM32,
        "VPMOVZXWD": Code.VEX_VPMOVZXWD_YMM_XMMM128,
        "VPMOVZXWQ": Code.VEX_VPMOVZXWQ_YMM_XMMM64,
        "VPMOVZXDQ": Code.VEX_VPMOVZXDQ_YMM_XMMM128,
    }
    accepted = 0
    for name, code in extend_codes.items():
        source, _ = lift_code(
            name, 0x1000, encode_seq(xmm_loads[0], Instruction.create_reg_reg(code, R.YMM0, R.XMM1), ret))
        assert f"ymm0 = op_{name.lower()}_ymm(ymm1);" in source, name
        # Aliased dst == src with YMM source loaded must also lift cleanly.
        alias_src, _ = lift_code(
            f"{name}_alias", 0x1000, encode_seq(ymm_loads[0], Instruction.create_reg_reg(code, R.YMM1, R.XMM1), ret))
        assert f"ymm1 = op_{name.lower()}_ymm(ymm1);" in alias_src, name
        accepted += 1

    for name in ("VPUNPCKLDQ", "VPUNPCKHDQ", "VPUNPCKLQDQ", "VPUNPCKHQDQ"):
        code = getattr(Code, f"VEX_{name}_YMM_YMM_YMMM256")
        source, _ = lift_code(
            name, 0x1000, encode_seq(*ymm_loads, Instruction.create_reg_reg_reg(code, R.YMM0, R.YMM1, R.YMM2), ret))
        assert f"wide128(op_{name.lower()}, ymm1, ymm2)" in source, name
        accepted += 1

    # 128-bit forms keep their 128-bit single-helper path.
    src_ext128, _ = lift_code(
        "xmm_vpmovsxbw", 0x1000,
        encode_seq(xmm_loads[0], Instruction.create_reg_reg(Code.VEX_VPMOVSXBW_XMM_XMMM64, R.XMM0, R.XMM1), ret))
    assert "ymm0 = op_vpmovsxbw(ymm1);" in src_ext128 and "_ymm" not in src_ext128
    src_unpck128, _ = lift_code(
        "xmm_vpunpckldq", 0x1000,
        encode_seq(*xmm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKLDQ_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2), ret))
    assert "ymm0 = op_vpunpckldq(ymm1, ymm2);" in src_unpck128 and "wide128" not in src_unpck128

    mem_err = "unsupported memory operand for register-domain AVX integer instruction"
    mem = MemoryOperand(R.RDX)
    cases = []
    for name, code in extend_codes.items():
        cases.append((f"{name.lower()}_mem", (Instruction.create_reg_mem(code, R.YMM0, mem), ret), mem_err))
    for name in ("VPUNPCKLDQ", "VPUNPCKHDQ", "VPUNPCKLQDQ", "VPUNPCKHQDQ"):
        code = getattr(Code, f"VEX_{name}_YMM_YMM_YMMM256")
        cases.append((f"{name.lower()}_mem", (ymm_loads[0], Instruction.create_reg_reg_mem(code, R.YMM0, R.YMM1, mem), ret), mem_err))

    cases += [
        # EVEX variants
        ("vpmovsxbw_evex", (xmm_loads[0], Instruction.create_reg_reg(Code.EVEX_VPMOVSXBW_YMM_K1Z_XMMM128, R.YMM0, R.XMM1), ret),
         "unsupported EVEX instruction"),
        ("vpmovzxdq_evex", (xmm_loads[0], Instruction.create_reg_reg(Code.EVEX_VPMOVZXDQ_YMM_K1Z_XMMM128, R.YMM0, R.XMM1), ret),
         "unsupported EVEX instruction"),
        ("vpunpckldq_evex", (*ymm_loads, Instruction.create_reg_reg_reg(Code.EVEX_VPUNPCKLDQ_YMM_K1Z_YMM_YMMM256B32, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported EVEX instruction"),
        ("vpunpckhqdq_evex", (*ymm_loads, Instruction.create_reg_reg_reg(Code.EVEX_VPUNPCKHQDQ_YMM_K1Z_YMM_YMMM256B64, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported EVEX instruction"),
        # Nonvolatile YMM6..15 / XMM6..15
        ("vpmovsxbd_nonvol_dst", (xmm_loads[0], Instruction.create_reg_reg(Code.VEX_VPMOVSXBD_YMM_XMMM64, R.YMM6, R.XMM1), ret),
         "nonvolatile vector register"),
        ("vpmovzxwq_nonvol_src", (xmm_loads[0], Instruction.create_reg_reg(Code.VEX_VPMOVZXWQ_YMM_XMMM64, R.YMM0, R.XMM7), ret),
         "nonvolatile vector register"),
        ("vpunpckldq_nonvol_dst", (*ymm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKLDQ_YMM_YMM_YMMM256, R.YMM8, R.YMM1, R.YMM2), ret),
         "nonvolatile vector register"),
        ("vpunpcklqdq_nonvol_src2", (*ymm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKLQDQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM15), ret),
         "nonvolatile vector register"),
        # Unlowered YMM unpack forms whose 128-bit encodings are carved out to cpu_vex_aot.py
        ("vpunpcklbw_ymm", (*ymm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKLBW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported instruction"),
        ("vpunpckhbw_ymm", (*ymm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKHBW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported instruction"),
        ("vpunpcklwd_ymm", (*ymm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKLWD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported instruction"),
        ("vpunpckhwd_ymm", (*ymm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKHWD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported instruction"),
        # Undefined sources
        ("vpmovsxbw_undef_src", (Instruction.create_reg_reg(Code.VEX_VPMOVSXBW_YMM_XMMM128, R.YMM0, R.XMM3), ret),
         "undefined vector register ymm3"),
        ("vpunpckhdq_undef_src2", (ymm_loads[0], Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKHDQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM3), ret),
         "undefined vector register ymm3"),
        # Missing terminal ret
        ("vpmovzxdq_missing_ret", (xmm_loads[0], Instruction.create_reg_reg(Code.VEX_VPMOVZXDQ_YMM_XMMM128, R.YMM0, R.XMM1)),
         "missing terminal ret"),
    ]
    for label, seq, reason in cases:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted a rejected control")
    return accepted, len(cases)


def check_differential(native, lifted, altered, altered_lifted):
    pairs = all_pairs()
    total = 0
    for name in EXPORTS:
        for index, (a, b) in enumerate(pairs):
            want = reference(name, a, b)
            for label, funcs in (("native", native), ("lifted", lifted)):
                got = invoke(funcs[name], a, b)
                if got != want:
                    raise AssertionError(
                        f"{label} {name} case {index} word {first_bad_word(got, want)}: "
                        f"got {got.hex()} expected {want.hex()}")
            total += 1

    # Unaligned input and output buffers; guard bytes must remain intact.
    a, b = pairs[len(pairs) // 2]
    for name in EXPORTS:
        for a_offset, out_offset in ((1, 0), (0, 3), (7, 5)):
            want = reference(name, a, b)
            for label, funcs in (("native", native), ("lifted", lifted)):
                got = invoke(funcs[name], a, b, a_offset, out_offset)
                if got != want:
                    raise AssertionError(f"{label} {name} offsets {a_offset}/{out_offset}")
            total += 1

    # Altered build checks: swapped ops match altered native and diverge from original.
    for name in BASE_EXPORTS:
        changed = 0
        for a, b in pairs:
            alt_nat = invoke(altered[name], a, b)
            alt_lif = invoke(altered_lifted[name], a, b)
            if alt_lif != alt_nat:
                raise AssertionError(f"altered {name}: lifted differs from native")
            changed += alt_nat != invoke(native[name], a, b)
        if not changed:
            raise AssertionError(f"altered build did not diverge on {name}")

    # Mutant Python models: must fail against lifted in the expected 64-bit words.
    zero16 = b"\x00" * 16
    mutants = [
        # Upper 128-bit half dropped on extend and unpack
        ("op_vpmovsxbw_ymm", lambda x, _y: reference("op_vpmovsxbw_ymm", x, _y)[:16] + zero16, {2, 3}),
        ("op_vpunpckldq_ymm", lambda x, y: reference("op_vpunpckldq_ymm", x, y)[:16] + zero16, {2, 3}),
        # Lower 128-bit extend duplicated into upper 128-bit lane instead of cross-lane slicing
        ("op_vpmovsxbw_ymm", lambda x, _y: reference("op_vpmovsxbw_ymm", x, _y)[:16] * 2, {2, 3}),
        ("op_vpmovzxbd_ymm", lambda x, _y: reference("op_vpmovzxbd_ymm", x, _y)[:16] * 2, {2, 3}),
        ("op_vpmovsxdq_ymm", lambda x, _y: reference("op_vpmovsxdq_ymm", x, _y)[:16] * 2, {2, 3}),
        # Stale upper YMM bits (bytes 16..31) erroneously read for upper half
        ("op_vpmovsxbw_ymm", lambda x, _y: _ref_extend(True, 1, 2, x)[:16] + _ref_extend(True, 1, 2, x[16:])[:16], {2, 3}),
        # Wrong sign/zero extension
        ("op_vpmovsxbd_ymm", lambda x, y: reference("op_vpmovzxbd_ymm", x, y), {0, 1, 2, 3}),
        ("op_vpmovzxwq_ymm", lambda x, y: reference("op_vpmovsxwq_ymm", x, y), {0, 1, 2, 3}),
        # Ignored upper XMM bytes (bytes 8..15) erroneously read for *BD
        ("op_vpmovsxbd_ymm", lambda x, _y: _ref_extend(True, 1, 4, x[8:]), {0, 1, 2, 3}),
        # Cross-lane or swapped unpacks
        ("op_vpunpckldq_ymm", lambda x, y: reference("op_vpunpckhdq_ymm", x, y), {0, 1, 2, 3}),
        ("op_vpunpcklqdq_ymm", lambda x, y: reference("op_vpunpckhqdq_ymm", x, y), {0, 1, 2, 3}),
        ("op_vpunpckldq_ymm",
         lambda x, y: _unpack_half("ldq", x[:16], y[:16]) + _unpack_half("ldq", x[:16], y[:16]), {2, 3}),
        # Missing VEX.128 upper-half zeroing after YMM extend
        ("op_vpmovsxwd_ymm_then_xmm_vpunpcklqdq",
         lambda x, y: reference("op_vpmovsxwd_ymm_then_xmm_vpunpcklqdq", x, y)[:16] + _ref_extend(True, 2, 4, x)[16:],
         {2, 3}),
    ]
    for name, wrong, allowed_words in mutants:
        bad_words = set()
        for x, y in pairs:
            w = first_bad_word(invoke(lifted[name], x, y), wrong(x, y))
            if w is not None:
                bad_words.add(w)
        if not bad_words or not bad_words <= allowed_words:
            raise AssertionError(
                f"mutant on {name}: bad words {sorted(bad_words)}, expected subset of {sorted(allowed_words)}")

    return total, len(mutants), len(BASE_EXPORTS)


def record_covered_forms(original_path):
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    formatter = Formatter(FormatterSyntax.INTEL)
    forms = set()
    with pefile.PE(str(original_path), fast_load=True) as pe:
        for name in EXPORTS:
            entry = export_rva(pe, name)
            raw = bounded_code(pe, entry)
            for ins in Decoder(64, raw, ip=entry):
                forms.add(formatter.format_mnemonic(ins).lower())
    out_path = Path(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(word + "\n" for word in sorted(forms)), encoding="utf-8")


def main():
    if os.environ.get("FORM_COVERAGE_RECORD"):
        Path(os.environ["FORM_COVERAGE_RECORD"]).unlink(missing_ok=True)

    original_path, lifted_path, altered_path, altered_lifted_path = (Path(p) for p in sys.argv[1:5])

    accepted, rejected = check_translator()
    print(f"{accepted} 256-bit extend/unpack forms accepted; {rejected} rejection controls passed")

    lifted_lib, lifted = bind(lifted_path, "lifted_")
    if not host_has_avx2(lifted_lib):
        print("Host lacks CPUID.01H:ECX[28] + XGETBV + CPUID.07H:EBX[5] AVX2; skipping native differential check")
        sys.exit(77)

    _, native = bind(original_path, "")
    _, altered = bind(altered_path, "")
    _, altered_lifted = bind(altered_lifted_path, "lifted_")
    total, caught, altered_count = check_differential(native, lifted, altered, altered_lifted)
    print(f"{total} native/lifted/reference cases across {len(EXPORTS)} exports matched bit for bit; "
          f"{caught} mutant controls + {altered_count} altered-build controls caught")

    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
