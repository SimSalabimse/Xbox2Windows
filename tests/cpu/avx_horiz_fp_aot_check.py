"""Differential, boundary, mutation, refusal, and coverage-record checks for dedicated AVX horizontal/addsub FP AOT."""

import ctypes
import os
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory

from iced_x86 import (
    Code,
    Decoder,
    Encoder,
    Formatter,
    FormatterSyntax,
    Instruction,
    MemoryOperand,
    Register as R,
)
import pefile

THIS_DIR = Path(__file__).resolve().parent
TOOLS_DIR = THIS_DIR.parents[1] / "tools"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from avx_horiz_fp import F32, F64, _guarded_buf  # noqa: E402
from cpu_avx_horiz_fp_aot import lift_code, main as tool_main, translate  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402


EXPORTS = (
    "op_vhaddps_xmm_reg",
    "op_vhaddpd_xmm_reg",
    "op_vhsubps_xmm_reg",
    "op_vhsubpd_xmm_reg",
    "op_vaddsubps_xmm_reg",
    "op_vaddsubpd_xmm_reg",
    "op_vhaddps_xmm_mem",
    "op_vhaddpd_xmm_mem",
    "op_vhsubps_xmm_mem",
    "op_vhsubpd_xmm_mem",
    "op_vaddsubps_xmm_mem",
    "op_vaddsubpd_xmm_mem",
    "op_vhaddps_ymm_reg",
    "op_vhaddpd_ymm_reg",
    "op_vhsubps_ymm_reg",
    "op_vhsubpd_ymm_reg",
    "op_vaddsubps_ymm_reg",
    "op_vaddsubpd_ymm_reg",
    "op_vhaddps_ymm_mem",
    "op_vhaddpd_ymm_mem",
    "op_vhsubps_ymm_mem",
    "op_vhsubpd_ymm_mem",
    "op_vaddsubps_ymm_mem",
    "op_vaddsubpd_ymm_mem",
    "op_vhaddps_ymm_mem_rcx",
    "op_vaddsubpd_ymm_mem_r8",
    "op_vhaddps_ymm_alias_dst_src1",
    "op_vhaddpd_ymm_alias_dst_src2",
    "op_vhsubps_ymm_alias_src1_src2",
    "op_vaddsubps_ymm_alias_all",
    "op_vaddsubpd_xmm_alias_dst_src2",
)

ALTERED_CASES = (
    ("op_vhaddps_ymm_reg", "op_vhsubps_ymm_reg"),
    ("op_vaddsubps_ymm_reg", "op_vhaddps_ymm_reg"),
    ("op_vaddsubpd_ymm_mem", "op_vhsubpd_ymm_mem"),
)


def encode_seq(*instructions):
    encoder = Encoder(64)
    rip = 0x1000
    for ins in instructions:
        rip += encoder.encode(ins, rip)
    return encoder.take_buffer()


def _lane_ps(kind: str, lane_a: bytes, lane_b: bytes, mxcsr: int) -> bytes:
    xa = struct.unpack("<4I", lane_a)
    xb = struct.unpack("<4I", lane_b)
    if kind == "vhadd":
        vals = (
            F32.add(xa[0], xa[1], mxcsr),
            F32.add(xa[2], xa[3], mxcsr),
            F32.add(xb[0], xb[1], mxcsr),
            F32.add(xb[2], xb[3], mxcsr),
        )
    elif kind == "vhsub":
        vals = (
            F32.sub(xa[0], xa[1], mxcsr),
            F32.sub(xa[2], xa[3], mxcsr),
            F32.sub(xb[0], xb[1], mxcsr),
            F32.sub(xb[2], xb[3], mxcsr),
        )
    elif kind == "vaddsub":
        vals = (
            F32.sub(xa[0], xb[0], mxcsr),
            F32.add(xa[1], xb[1], mxcsr),
            F32.sub(xa[2], xb[2], mxcsr),
            F32.add(xa[3], xb[3], mxcsr),
        )
    else:
        raise ValueError(kind)
    return struct.pack("<4I", *vals)


def _lane_pd(kind: str, lane_a: bytes, lane_b: bytes, mxcsr: int) -> bytes:
    xa = struct.unpack("<2Q", lane_a)
    xb = struct.unpack("<2Q", lane_b)
    if kind == "vhadd":
        vals = (F64.add(xa[0], xa[1], mxcsr), F64.add(xb[0], xb[1], mxcsr))
    elif kind == "vhsub":
        vals = (F64.sub(xa[0], xa[1], mxcsr), F64.sub(xb[0], xb[1], mxcsr))
    elif kind == "vaddsub":
        vals = (F64.sub(xa[0], xb[0], mxcsr), F64.add(xa[1], xb[1], mxcsr))
    else:
        raise ValueError(kind)
    return struct.pack("<2Q", *vals)


def ref_op(name: str, a: bytes, b: bytes, mxcsr: int) -> bytes:
    if name == "op_vhaddps_ymm_mem_rcx":
        a, b = b, a
    elif name in ("op_vhsubps_ymm_alias_src1_src2", "op_vaddsubps_ymm_alias_all"):
        b = a
    is_xmm = "_xmm_" in name
    is_pd = "pd_" in name
    if "vhadd" in name:
        kind = "vhadd"
    elif "vhsub" in name:
        kind = "vhsub"
    elif "vaddsub" in name:
        kind = "vaddsub"
    else:
        raise KeyError(name)
    lane_fn = _lane_pd if is_pd else _lane_ps
    lo = lane_fn(kind, a[0:16], b[0:16], mxcsr)
    if is_xmm:
        return lo + b"\x00" * 16
    return lo + lane_fn(kind, a[16:32], b[16:32], mxcsr)


def check_translator(original_path: Path):
    source_cpp, meta = translate(original_path, EXPORTS)
    assert len(meta) == len(EXPORTS) and "host_has_avx" in source_cpp, meta

    ld_ymm0 = Instruction.create_reg_mem(Code.VEX_VMOVUPS_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX))
    ld_ymm1 = Instruction.create_reg_mem(Code.VEX_VMOVAPS_YMM_YMMM256, R.YMM1, MemoryOperand(R.RCX))
    ld_ymm2 = Instruction.create_reg_mem(Code.VEX_VMOVUPD_YMM_YMMM256, R.YMM2, MemoryOperand(R.RDX))
    ld_xmm1 = Instruction.create_reg_mem(Code.VEX_VMOVAPD_XMM_XMMM128, R.XMM1, MemoryOperand(R.RCX))
    ld_xmm2 = Instruction.create_reg_mem(Code.VEX_VMOVUPS_XMM_XMMM128, R.XMM2, MemoryOperand(R.RDX))
    ret_ins = Instruction.create(Code.RETNQ)
    vzero = Instruction.create(Code.VEX_VZEROUPPER)

    # All 12 horizontal/addsub codes (6 mnemonics x XMM/YMM) in reg & flat [rcx|rdx|r8] mem-src forms.
    accepted = 0
    for code, is_ymm, helper in (
        (Code.VEX_VHADDPS_XMM_XMM_XMMM128, False, "op_vhaddps"),
        (Code.VEX_VHADDPS_YMM_YMM_YMMM256, True, "op_vhaddps"),
        (Code.VEX_VHADDPD_XMM_XMM_XMMM128, False, "op_vhaddpd"),
        (Code.VEX_VHADDPD_YMM_YMM_YMMM256, True, "op_vhaddpd"),
        (Code.VEX_VHSUBPS_XMM_XMM_XMMM128, False, "op_vhsubps"),
        (Code.VEX_VHSUBPS_YMM_YMM_YMMM256, True, "op_vhsubps"),
        (Code.VEX_VHSUBPD_XMM_XMM_XMMM128, False, "op_vhsubpd"),
        (Code.VEX_VHSUBPD_YMM_YMM_YMMM256, True, "op_vhsubpd"),
        (Code.VEX_VADDSUBPS_XMM_XMM_XMMM128, False, "op_vaddsubps"),
        (Code.VEX_VADDSUBPS_YMM_YMM_YMMM256, True, "op_vaddsubps"),
        (Code.VEX_VADDSUBPD_XMM_XMM_XMMM128, False, "op_vaddsubpd"),
        (Code.VEX_VADDSUBPD_YMM_YMM_YMMM256, True, "op_vaddsubpd"),
    ):
        dst, s1, s2 = (R.YMM0, R.YMM1, R.YMM2) if is_ymm else (R.XMM0, R.XMM1, R.XMM2)
        loads = (ld_ymm1, ld_ymm2) if is_ymm else (ld_xmm1, ld_xmm2)
        st_code = Code.VEX_VMOVUPS_YMMM256_YMM if is_ymm else Code.VEX_VMOVUPS_XMMM128_XMM
        reg_src, _ = lift_code(
            f"{helper}_reg",
            0x1000,
            encode_seq(
                *loads,
                Instruction.create_reg_reg_reg(code, dst, s1, s2),
                Instruction.create_mem_reg(st_code, MemoryOperand(R.R8), dst),
                vzero,
                ret_ins,
            ),
        )
        assert f"{helper}(ymm1, ymm2, {'true' if is_ymm else 'false'})" in reg_src, reg_src
        accepted += 1

        for base in (R.RCX, R.RDX, R.R8):
            mem_src, _ = lift_code(
                f"{helper}_mem_{base}",
                0x1000,
                encode_seq(
                    loads[0],
                    Instruction.create_reg_reg_mem(code, dst, s1, MemoryOperand(base)),
                    Instruction.create_mem_reg(st_code, MemoryOperand(R.R8), dst),
                    ret_ins,
                ),
            )
            loader = "load256" if is_ymm else "load128"
            assert f"{helper}(ymm1, {loader}(" in mem_src, mem_src
            accepted += 1

    evex_ins = Instruction.create_reg_reg_reg(Code.EVEX_VADDPS_YMM_K1Z_YMM_YMMM256B32, R.YMM0, R.YMM1, R.YMM2)
    evex_ins.op_mask = R.K1
    seg_mem = MemoryOperand(R.RDX)
    seg_prefix_ins = Instruction.create_reg_reg_mem(Code.VEX_VHADDPS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, seg_mem)
    seg_prefix_ins.segment_prefix = R.FS

    rejects = [
        ("evex", (ld_ymm1, ld_ymm2, evex_ins, ret_ins), "unsupported EVEX instruction"),
        ("legacy_sse", (ld_xmm1, ld_xmm2, Instruction.create_reg_mem(Code.HADDPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX)), ret_ins), "unsupported instruction"),
        ("nonvol_ymm6_dst", (ld_ymm1, ld_ymm2, Instruction.create_reg_reg_reg(Code.VEX_VHADDPS_YMM_YMM_YMMM256, R.YMM6, R.YMM1, R.YMM2), ret_ins), "nonvolatile vector register"),
        ("nonvol_xmm15_src", (ld_xmm1, Instruction.create_reg_reg_reg(Code.VEX_VHSUBPD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM15), ret_ins), "nonvolatile vector register"),
        ("nonvol_ymm7_move", (Instruction.create_reg_mem(Code.VEX_VMOVAPS_YMM_YMMM256, R.YMM7, MemoryOperand(R.RCX)), ret_ins), "nonvolatile vector register"),
        ("undef_ymm3", (ld_ymm1, Instruction.create_reg_reg_reg(Code.VEX_VHADDPD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM3), ret_ins), "undefined vector register ymm3"),
        ("undef_alias_dst", (ld_ymm1, Instruction.create_reg_reg_reg(Code.VEX_VHADDPS_YMM_YMM_YMMM256, R.YMM0, R.YMM0, R.YMM1), ret_ins), "undefined vector register ymm0"),
        ("disp_mem_rcx", (ld_ymm1, Instruction.create_reg_reg_mem(Code.VEX_VHADDPS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RCX, displ=32, displ_size=1)), ret_ins), "unsupported memory operand"),
        ("disp_mem_rdx", (ld_xmm1, Instruction.create_reg_reg_mem(Code.VEX_VADDSUBPS_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX, displ=16, displ_size=1)), ret_ins), "unsupported memory operand"),
        ("disp_store_r8", (ld_ymm0, Instruction.create_mem_reg(Code.VEX_VMOVUPS_YMMM256_YMM, MemoryOperand(R.R8, displ=32, displ_size=1), R.YMM0), ret_ins), "unsupported memory operand"),
        ("non_r8_store_rcx", (ld_ymm0, Instruction.create_mem_reg(Code.VEX_VMOVUPS_YMMM256_YMM, MemoryOperand(R.RCX), R.YMM0), ret_ins), "unsupported memory operand"),
        ("non_r8_store_rdx", (ld_xmm1, Instruction.create_mem_reg(Code.VEX_VMOVAPD_XMMM128_XMM, MemoryOperand(R.RDX), R.XMM1), ret_ins), "unsupported memory operand"),
        ("indexed_mem", (ld_ymm1, Instruction.create_reg_reg_mem(Code.VEX_VHSUBPS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX, R.RCX, 1)), ret_ins), "unsupported memory operand"),
        ("rip_rel_mem", (ld_ymm1, Instruction.create_reg_reg_mem(Code.VEX_VADDSUBPD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RIP, displ=0x1040, displ_size=4)), ret_ins), "unsupported memory operand"),
        ("non_arg_base_rax", (ld_ymm1, Instruction.create_reg_reg_mem(Code.VEX_VHADDPS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RAX)), ret_ins), "unsupported memory operand"),
        ("addr32_ecx", (ld_xmm1, Instruction.create_reg_reg_mem(Code.VEX_VHADDPS_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.ECX)), ret_ins), "unsupported memory operand"),
        ("seg_prefix", (ld_ymm1, seg_prefix_ins, ret_ins), "unsupported instruction prefix"),
        ("missing_ret", (ld_ymm1, ld_ymm2, Instruction.create_reg_reg_reg(Code.VEX_VHADDPS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)), "missing terminal ret"),
        ("after_ret", (ld_ymm1, ret_ins, ld_ymm2), "unreachable instruction after ret"),
    ]
    for label, seq, reason in rejects:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted invalid sequence")

    # Verify CLI failure exit code 1 without traceback and zero temp files.
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_cpp = root / "out.cpp"
        try:
            tool_main([str(root / "missing.dll"), "op_vhaddps_ymm_reg", str(out_cpp)])
        except SystemExit as exc:
            assert exc.code == 1, exc.code
        else:
            raise AssertionError("missing DLL did not exit")
        assert not out_cpp.exists() and list(root.iterdir()) == []

    print(f"translator validated: {accepted} forms accepted, {len(rejects)} refusal controls passed")


def host_has_avx(lifted_lib) -> bool:
    if hasattr(lifted_lib, "host_has_avx"):
        lifted_lib.host_has_avx.argtypes = []
        lifted_lib.host_has_avx.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_avx())
    return False


def check_differential(native_path: Path, lifted_path: Path, altered_path: Path, altered_lifted_path: Path) -> None:
    native = ctypes.CDLL(str(native_path.resolve()))
    lifted = ctypes.CDLL(str(lifted_path.resolve()))
    altered_native = ctypes.CDLL(str(altered_path.resolve()))
    altered_lifted = ctypes.CDLL(str(altered_lifted_path.resolve()))

    exchange = lifted.host_exchange_mxcsr
    exchange.argtypes, exchange.restype = [ctypes.c_uint32], ctypes.c_uint32

    def _f32(*vals):
        words = [v if isinstance(v, int) else struct.unpack("<I", struct.pack("<f", v))[0] for v in vals]
        return struct.pack("<8I", *words)

    def _f64(*vals):
        words = [v if isinstance(v, int) else struct.unpack("<Q", struct.pack("<d", v))[0] for v in vals]
        return struct.pack("<4Q", *words)

    f32_qnan_pos, f32_qnan_neg = 0x7FC01234, 0xFFC05678
    f32_snan_pos, f32_snan_neg = 0x7F800042, 0xFFA00099
    f32_sub_pos, f32_sub_neg = 0x00400001, 0x80400001
    f32_min_norm = 0x00800000
    f32_one_half_min_norm = 0x00C00000
    f32_max_fin = 0x7F7FFFFF

    f32_pairs = (
        (_f32(1.5, -2.25, 1.0, 2.0 ** -24, 3.5, 0.125, -1.0, -(2.0 ** -24)),
         _f32(4.25, 0.75, f32_min_norm, f32_sub_neg, -7.5, 2.5, f32_sub_pos, f32_sub_pos)),
        (_f32(0x00000000, 0x00000000, 0x80000000, 0x80000000, f32_one_half_min_norm, f32_min_norm | 0x80000000, 1.5, -1.5),
         _f32(0x00000000, 0x80000000, 0x00000000, 0x80000000, f32_one_half_min_norm | 0x80000000, f32_min_norm, 2.5, 2.5)),
        (_f32(0x7F800000, 0x7F800000, 0x7F800000, 0xFF800000, f32_max_fin, f32_max_fin, f32_sub_pos, f32_sub_pos),
         _f32(0x7F800000, 1.0, 0xFF800000, 0xFF800000, f32_max_fin | 0x80000000, f32_max_fin | 0x80000000, f32_min_norm, f32_min_norm | 0x80000000)),
        (_f32(f32_qnan_pos, 1.0, 2.0, f32_snan_pos, f32_qnan_pos, f32_qnan_neg, f32_snan_pos, f32_snan_neg),
         _f32(f32_snan_neg, f32_qnan_neg, f32_qnan_neg, 3.0, f32_snan_neg, f32_qnan_pos, f32_qnan_pos, f32_snan_pos)),
    )

    f64_qnan_pos, f64_qnan_neg = 0x7FF8000012345678, 0xFFF8000087654321
    f64_snan_pos, f64_snan_neg = 0x7FF0000000000042, 0xFFF4000000000099
    f64_sub_pos, f64_sub_neg = 0x0008000000000001, 0x8008000000000001
    f64_min_norm = 0x0010000000000000
    f64_one_half_min_norm = 0x0018000000000000
    f64_max_fin = 0x7FEFFFFFFFFFFFFF

    f64_pairs = (
        (_f64(1.5, 2.0 ** -53, -3.5, -(2.0 ** -53)),
         _f64(f64_min_norm, f64_sub_neg, f64_sub_pos, f64_sub_pos)),
        (_f64(0x0000000000000000, 0x8000000000000000, f64_one_half_min_norm, f64_min_norm | 0x8000000000000000),
         _f64(2.25, -2.25, 0x0000000000000000, 0x8000000000000000)),
        (_f64(0x7FF0000000000000, 0xFFF0000000000000, f64_max_fin, f64_max_fin),
         _f64(0x7FF0000000000000, 0x7FF0000000000000, f64_max_fin | 0x8000000000000000, f64_max_fin | 0x8000000000000000)),
        (_f64(f64_qnan_pos, 1.0, f64_snan_pos, f64_qnan_neg),
         _f64(f64_snan_neg, f64_qnan_neg, 2.0, f64_snan_pos)),
    )

    mxcsr_modes = (0x1F80, 0x3F80, 0x5F80, 0x7F80, 0x1FC0, 0x9F80, 0x9FC0)
    cases = 0

    def invoke(dll, prefix: str, name: str, a_bytes: bytes, b_bytes: bytes, mxcsr: int, align: int = 0) -> bytes:
        _ra, pa = _guarded_buf(a_bytes, align if name == "op_vhaddps_ymm_mem_rcx" else 0)
        _rb, pb = _guarded_buf(b_bytes, 0 if name in ("op_vhaddps_ymm_mem_rcx", "op_vaddsubpd_ymm_mem_r8") else align)
        seed = b_bytes if name == "op_vaddsubpd_ymm_mem_r8" else b"\xCC" * 32
        _rc, pc = _guarded_buf(seed, align if name == "op_vaddsubpd_ymm_mem_r8" else 0)
        fn = getattr(dll, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = None
        previous = exchange(mxcsr)
        try:
            fn(pa, pb, pc)
        finally:
            exchange(previous)
        return ctypes.string_at(pc, 32)

    for name in EXPORTS:
        pairs = f64_pairs if "pd_" in name else f32_pairs
        for mxcsr in mxcsr_modes:
            for a_bytes, b_bytes in pairs:
                want = ref_op(name, a_bytes, b_bytes, mxcsr)
                got_n = invoke(native, "", name, a_bytes, b_bytes, mxcsr, 0)
                got_l = invoke(lifted, "lifted_", name, a_bytes, b_bytes, mxcsr, 0)
                if got_n != want or got_l != want:
                    raise AssertionError(
                        f"{name} mxcsr={mxcsr:#x}: want={want.hex()} native={got_n.hex()} lifted={got_l.hex()}"
                    )
                cases += 1
        if "_mem" in name:
            a_bytes, b_bytes = pairs[0]
            for off in (1, 3, 7):
                want = ref_op(name, a_bytes, b_bytes, 0x1F80)
                got_n = invoke(native, "", name, a_bytes, b_bytes, 0x1F80, off)
                got_l = invoke(lifted, "lifted_", name, a_bytes, b_bytes, 0x1F80, off)
                if got_n != want or got_l != want:
                    raise AssertionError(
                        f"{name} align={off}: want={want.hex()} native={got_n.hex()} lifted={got_l.hex()}"
                    )
                cases += 1

    # Altered controls must diverge from original and match altered native DLL.
    for name, alt_ref_name in ALTERED_CASES:
        pairs = f64_pairs if "pd_" in name else f32_pairs
        a_bytes, b_bytes = pairs[0]
        want_orig = ref_op(name, a_bytes, b_bytes, 0x1F80)
        want_alt = ref_op(alt_ref_name, a_bytes, b_bytes, 0x1F80)
        got_alt_n = invoke(altered_native, "", name, a_bytes, b_bytes, 0x1F80, 0)
        got_alt_l = invoke(altered_lifted, "lifted_", name, a_bytes, b_bytes, 0x1F80, 0)
        assert got_alt_n == want_alt and got_alt_l == want_alt, (name, got_alt_n.hex(), got_alt_l.hex(), want_alt.hex())
        assert got_alt_l != want_orig, f"altered {name} failed to diverge from original"

    # Altered XMM zeroing control: vhaddpd ymm0 instead of xmm0 leaves bits [255:128] non-zero.
    a64, b64 = f64_pairs[0]
    want_xmm_orig = ref_op("op_vhaddpd_xmm_reg", a64, b64, 0x1F80)
    got_xmm_alt_n = invoke(altered_native, "", "op_vhaddpd_xmm_reg", a64, b64, 0x1F80, 0)
    got_xmm_alt_l = invoke(altered_lifted, "lifted_", "op_vhaddpd_xmm_reg", a64, b64, 0x1F80, 0)
    assert got_xmm_alt_n == got_xmm_alt_l, (got_xmm_alt_n.hex(), got_xmm_alt_l.hex())
    assert got_xmm_alt_l[:16] == want_xmm_orig[:16] and got_xmm_alt_l[16:] != b"\x00" * 16, got_xmm_alt_l.hex()

    print(
        f"{cases} AVX horizontal/addsub FP cases across {len(EXPORTS)} exports and {len(mxcsr_modes)} MXCSR modes "
        f"matched native DLL and IEEE-754 reference; {len(ALTERED_CASES) + 1} altered controls diverged"
    )


def record_covered_forms(original_path: Path) -> None:
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


def main() -> None:
    if os.environ.get("FORM_COVERAGE_RECORD"):
        Path(os.environ["FORM_COVERAGE_RECORD"]).unlink(missing_ok=True)

    original_path, lifted_path, altered_path, altered_lifted_path = (Path(p) for p in sys.argv[1:5])
    check_translator(original_path)

    lifted_lib = ctypes.CDLL(str(lifted_path.resolve()))
    if not host_has_avx(lifted_lib):
        print("Host lacks CPUID.01H:ECX[27:28] + XGETBV[2:1]; skipping native differential check")
        sys.exit(77)

    check_differential(original_path, lifted_path, altered_path, altered_lifted_path)
    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
