"""Lift bounded Windows x64 leaf functions with AVX horizontal and add/sub FP ops to C++.

Each export uses the straight-line leaf ABI:
    void lifted_<name>(word rcx, word rdx, word r8)
where rcx, rdx, and r8 carry flat host pointers to caller-owned buffers. YMM0..YMM5
are modeled as portable 256-bit value structs (four 64-bit lanes); every VEX.128
write to an XMM destination zeroes bits [255:128] of the corresponding YMM register.

Supported AVX horizontal / alternating FP forms (docs/coverage/x64_zen2.tsv group AVX):
  - VHADDPS, VHADDPD, VHSUBPS, VHSUBPD, VADDSUBPS, VADDSUBPD
    in VEX.128 (xmm, xmm, xmm/m128) and VEX.256 (ymm, ymm, ymm/m256) forms.

Transport / ABI glue:
  - VMOVAPS, VMOVAPD, VMOVUPS, VMOVUPD (reg-reg, flat [rcx|rdx|r8] loads, flat [r8] stores)
  - VZEROUPPER and terminal RET.

EVEX encodings, legacy SSE forms, nonvolatile vector registers (XMM6..15/YMM6..15),
undefined vector reads, displaced or non-flat memory operands, non-[r8] stores,
instruction prefixes, branches, and trailing or missing ret are rejected.
"""

import argparse
from pathlib import Path
import tempfile

import pefile
from iced_x86 import (
    Code,
    Decoder,
    EncodingKind,
    MemorySizeExt,
    OpKind,
    Register as R,
    RegisterExt,
)

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}
VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
VOLATILE_YMM = {getattr(R, f"YMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = (
    {getattr(R, f"XMM{i}") for i in range(6, 32)}
    | {getattr(R, f"YMM{i}") for i in range(6, 32)}
    | {getattr(R, f"ZMM{i}") for i in range(32)}
)

# (helper_name, is_ymm)
HORIZ_FP_CODES = {
    Code.VEX_VHADDPS_XMM_XMM_XMMM128: ("op_vhaddps", False),
    Code.VEX_VHADDPS_YMM_YMM_YMMM256: ("op_vhaddps", True),
    Code.VEX_VHADDPD_XMM_XMM_XMMM128: ("op_vhaddpd", False),
    Code.VEX_VHADDPD_YMM_YMM_YMMM256: ("op_vhaddpd", True),
    Code.VEX_VHSUBPS_XMM_XMM_XMMM128: ("op_vhsubps", False),
    Code.VEX_VHSUBPS_YMM_YMM_YMMM256: ("op_vhsubps", True),
    Code.VEX_VHSUBPD_XMM_XMM_XMMM128: ("op_vhsubpd", False),
    Code.VEX_VHSUBPD_YMM_YMM_YMMM256: ("op_vhsubpd", True),
    Code.VEX_VADDSUBPS_XMM_XMM_XMMM128: ("op_vaddsubps", False),
    Code.VEX_VADDSUBPS_YMM_YMM_YMMM256: ("op_vaddsubps", True),
    Code.VEX_VADDSUBPD_XMM_XMM_XMMM128: ("op_vaddsubpd", False),
    Code.VEX_VADDSUBPD_YMM_YMM_YMMM256: ("op_vaddsubpd", True),
}

# is_ymm for full-width VEX FP moves
VEC_MOVE_CODES = {
    Code.VEX_VMOVAPS_XMM_XMMM128: False,
    Code.VEX_VMOVAPS_XMMM128_XMM: False,
    Code.VEX_VMOVAPS_YMM_YMMM256: True,
    Code.VEX_VMOVAPS_YMMM256_YMM: True,
    Code.VEX_VMOVAPD_XMM_XMMM128: False,
    Code.VEX_VMOVAPD_XMMM128_XMM: False,
    Code.VEX_VMOVAPD_YMM_YMMM256: True,
    Code.VEX_VMOVAPD_YMMM256_YMM: True,
    Code.VEX_VMOVUPS_XMM_XMMM128: False,
    Code.VEX_VMOVUPS_XMMM128_XMM: False,
    Code.VEX_VMOVUPS_YMM_YMMM256: True,
    Code.VEX_VMOVUPS_YMMM256_YMM: True,
    Code.VEX_VMOVUPD_XMM_XMMM128: False,
    Code.VEX_VMOVUPD_XMMM128_XMM: False,
    Code.VEX_VMOVUPD_YMM_YMMM256: True,
    Code.VEX_VMOVUPD_YMMM256_YMM: True,
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.VEX_VZEROUPPER,
    *HORIZ_FP_CODES,
    *VEC_MOVE_CODES,
}

PRELUDE = """#include <atomic>
#include <cstdint>
#include <cstring>
#include <pmmintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#define X2W_UNALIGNED __unaligned
#else
#define X2W_UNALIGNED
#endif

using word = std::uint64_t;

struct Vec256 {
    std::uint64_t q[4];
};

static inline Vec256 load128(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* p = reinterpret_cast<const volatile std::uint64_t X2W_UNALIGNED*>(address);
    const std::uint64_t q0 = p[0];
    const std::uint64_t q1 = p[1];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, 0ull, 0ull}};
}

static inline Vec256 load256(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* p = reinterpret_cast<const volatile std::uint64_t X2W_UNALIGNED*>(address);
    const std::uint64_t q0 = p[0];
    const std::uint64_t q1 = p[1];
    const std::uint64_t q2 = p[2];
    const std::uint64_t q3 = p[3];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, q2, q3}};
}

static inline void store128(word address, const Vec256& v) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store256(word address, const Vec256& v) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), v.q, 32);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline Vec256 copy128(const Vec256& v) {
    return {{v.q[0], v.q[1], 0ull, 0ull}};
}

template <typename LaneVec, typename Fn>
static inline Vec256 apply_sse3_lanes(const Vec256& a, const Vec256& b, bool wide, Fn fn) {
    Vec256 out{};
    const unsigned lanes = wide ? 2u : 1u;
    for (unsigned lane = 0; lane < lanes; ++lane) {
        LaneVec va;
        LaneVec vb;
        std::memcpy(&va, &a.q[lane * 2u], 16);
        std::memcpy(&vb, &b.q[lane * 2u], 16);
        const LaneVec vr = fn(va, vb);
        std::memcpy(&out.q[lane * 2u], &vr, 16);
    }
    return out;
}

static inline Vec256 op_vhaddps(const Vec256& a, const Vec256& b, bool wide) {
    return apply_sse3_lanes<__m128>(a, b, wide, [](__m128 x, __m128 y) { return _mm_hadd_ps(x, y); });
}

static inline Vec256 op_vhaddpd(const Vec256& a, const Vec256& b, bool wide) {
    return apply_sse3_lanes<__m128d>(a, b, wide, [](__m128d x, __m128d y) { return _mm_hadd_pd(x, y); });
}

static inline Vec256 op_vhsubps(const Vec256& a, const Vec256& b, bool wide) {
    return apply_sse3_lanes<__m128>(a, b, wide, [](__m128 x, __m128 y) { return _mm_hsub_ps(x, y); });
}

static inline Vec256 op_vhsubpd(const Vec256& a, const Vec256& b, bool wide) {
    return apply_sse3_lanes<__m128d>(a, b, wide, [](__m128d x, __m128d y) { return _mm_hsub_pd(x, y); });
}

static inline Vec256 op_vaddsubps(const Vec256& a, const Vec256& b, bool wide) {
    return apply_sse3_lanes<__m128>(a, b, wide, [](__m128 x, __m128 y) { return _mm_addsub_ps(x, y); });
}

static inline Vec256 op_vaddsubpd(const Vec256& a, const Vec256& b, bool wide) {
    return apply_sse3_lanes<__m128d>(a, b, wide, [](__m128d x, __m128d y) { return _mm_addsub_pd(x, y); });
}

#if defined(_MSC_VER)
extern "C" __declspec(dllexport) std::uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}

extern "C" __declspec(dllexport) std::uint32_t host_exchange_mxcsr(std::uint32_t next) {
    const std::uint32_t current = _mm_getcsr();
    _mm_setcsr(next);
    return current;
}
#endif
"""


def vec_index(register, ins, *, expected_wide):
    """Map XMM0..XMM5 or YMM0..YMM5 to register index, enforcing width and rejecting nonvolatile/other."""
    if register in VOLATILE_XMM:
        if expected_wide:
            raise ValueError(f"unsupported vector width for {register} at RVA {ins.ip:#x}: {ins}")
        return VOLATILE_XMM[register]
    if register in VOLATILE_YMM:
        if not expected_wide:
            raise ValueError(f"unsupported vector width for {register} at RVA {ins.ip:#x}: {ins}")
        return VOLATILE_YMM[register]
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, *, store=False, expected_size=None):
    """Lower a flat zero-displacement [rcx|rdx|r8] pointer operand (stores restricted to [r8])."""
    base = ins.memory_base
    allowed = {R.R8} if store else ARGUMENTS
    if (
        ins.is_ip_rel_memory_operand
        or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
        or ins.memory_index != R.NONE
        or ins.memory_displacement != 0
        or not RegisterExt.is_gpr64(base)
        or base not in allowed
    ):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    if expected_size is not None and MemorySizeExt.size(ins.memory_size) != expected_size:
        raise ValueError(f"unsupported memory size at RVA {ins.ip:#x}: {ins}")
    return ARGUMENTS[base]


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line AVX horizontal/addsub FP leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used = set()
    defined = set()
    statements = []
    returned = False

    def read_vec(op_index, ins, *, expected_wide):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(ins.op_register(op_index), ins, expected_wide=expected_wide)
        if idx not in defined:
            raise ValueError(f"undefined vector register ymm{idx} at RVA {ins.ip:#x}: {ins}")
        used.add(idx)
        return f"ymm{idx}"

    def write_vec(op_index, ins, *, expected_wide):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(ins.op_register(op_index), ins, expected_wide=expected_wide)
        used.add(idx)
        return idx, f"ymm{idx}"

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        if (
            ins.op_code().encoding == EncodingKind.EVEX
            or ins.op_mask != R.NONE
            or ins.zeroing_masking
            or ins.is_broadcast
        ):
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            statements.append("return;")
            returned = True
        elif ins.code == Code.VEX_VZEROUPPER:
            for idx in sorted(defined):
                statements.append(f"ymm{idx}.q[2] = 0ull; ymm{idx}.q[3] = 0ull;")
        elif ins.code in HORIZ_FP_CODES:
            helper, wide = HORIZ_FP_CODES[ins.code]
            if ins.op_count != 3 or ins.op0_kind != OpKind.REGISTER or ins.op1_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported AVX horizontal FP operands at RVA {ins.ip:#x}: {ins}")
            src1 = read_vec(1, ins, expected_wide=wide)
            if ins.op2_kind == OpKind.MEMORY:
                addr = flat_address(ins, expected_size=32 if wide else 16)
                src2 = f"{'load256' if wide else 'load128'}({addr})"
            elif ins.op2_kind == OpKind.REGISTER:
                src2 = read_vec(2, ins, expected_wide=wide)
            else:
                raise ValueError(f"unsupported AVX horizontal FP operands at RVA {ins.ip:#x}: {ins}")
            dst_idx, dst = write_vec(0, ins, expected_wide=wide)
            statements.append(f"{dst} = {helper}({src1}, {src2}, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif ins.code in VEC_MOVE_CODES:
            wide = VEC_MOVE_CODES[ins.code]
            expected_size = 32 if wide else 16
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, expected_size=expected_size)
                dst_idx, dst = write_vec(0, ins, expected_wide=wide)
                statements.append(f"{dst} = {'load256' if wide else 'load128'}({addr});")
                defined.add(dst_idx)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src = read_vec(1, ins, expected_wide=wide)
                addr = flat_address(ins, store=True, expected_size=expected_size)
                statements.append(f"{'store256' if wide else 'store128'}({addr}, {src});")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src = read_vec(1, ins, expected_wide=wide)
                dst_idx, dst = write_vec(0, ins, expected_wide=wide)
                statements.append(f"{dst} = {src};" if wide else f"{dst} = copy128({src});")
                defined.add(dst_idx)
            else:
                raise ValueError(f"unsupported vector move operands at RVA {ins.ip:#x}: {ins}")
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec256 ymm{idx}{{}};" for idx in sorted(used)]
    body = [f"    {stmt}" for stmt in statements]
    lines = [
        f'extern "C" __declspec(dllexport) void lifted_{name}(word rcx, word rdx, word r8) {{',
        "    (void)rcx; (void)rdx; (void)r8;",
        *decls,
        *body,
        "}",
    ]
    return "\n".join(lines), len(instructions)


def lift(pe, name):
    entry = export_rva(pe, name)
    raw = bounded_code(pe, entry)
    source, count = lift_code(name, entry, raw)
    return source, entry, count


def translate_pe(pe, names):
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    lifted = [lift(pe, name) for name in names]
    source = PRELUDE + "\n".join(f"\n{text}" for text, _, _ in lifted) + "\n"
    meta = [(name, entry, count) for name, (_, entry, count) in zip(names, lifted)]
    return source, meta


def translate(image, names):
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, names)


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated export names")
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        if not args.image.is_file():
            raise ValueError(f"input image must be a regular file: {args.image}")
        exports = [name for name in args.exports.split(",") if name]
        if not exports:
            raise ValueError("exports list must not be empty")
        source, meta = translate(args.image, exports)
        _write_atomic(args.output, source)
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x}: {count} instructions")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"cpu_avx_horiz_fp_aot: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
