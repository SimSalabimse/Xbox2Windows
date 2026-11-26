"""Differential, /O2, flag-preservation, mutant, and rejection checks for cpu_bmi2_bit_aot."""

import ctypes
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from iced_x86 import Code, Decoder, Encoder, Formatter, FormatterSyntax, Instruction, MemoryOperand, Register as R
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_bmi2_bit_aot import lift_code, translate  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

EXPORTS = (
    "op_bzhi32",
    "op_bzhi64",
    "op_mulx32",
    "op_mulx64",
    "op_mulx32_alias",
    "op_mulx64_alias",
    "op_pdep32",
    "op_pdep64",
    "op_pext32",
    "op_pext64",
    "op_bzhi32_mem",
    "op_bzhi64_mem",
    "op_mulx32_mem",
    "op_mulx64_mem",
    "op_pdep32_mem",
    "op_pdep64_mem",
    "op_pext32_mem",
    "op_pext64_mem",
    "op_mem_r8_source",
    "op_bzhi32_flags",
    "op_bzhi64_flags",
    "op_flags_preserve32",
    "op_flags_preserve64",
    "op_flags_preserve_pf",
    "op_flags_preserve_mem",
    "op_bmi2_bit_discarded_read",
)

ALTERED_EXPORTS = (
    "op_bzhi32",
    "op_bzhi64",
    "op_mulx32",
    "op_mulx64",
    "op_mulx32_alias",
    "op_mulx64_alias",
    "op_pdep32",
    "op_pdep64",
    "op_pext32",
    "op_pext64",
)

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))
OUT_FILL = b"\x5A" * 16


# --- Independent Python reference model ---

def ref_bzhi(val, idx, bits, *, bad_mask=False):
    mask = (1 << bits) - 1
    s = val & mask
    n = (idx & (bits - 1)) if bad_mask else (idx & 0xFF)
    if n >= bits:
        d = s
        cf = 1
    else:
        d = s & ((1 << n) - 1)
        cf = 0
    zf = 1 if d == 0 else 0
    sf = 1 if (d & (1 << (bits - 1))) else 0
    of = 0
    packed = (cf << 0) | (zf << 1) | (sf << 2) | (of << 3)
    return d, packed


def ref_mulx(src1, src2, bits):
    mask = (1 << bits) - 1
    prod = (src1 & mask) * (src2 & mask)
    lo = prod & mask
    hi = (prod >> bits) & mask
    return hi, lo


def ref_pdep(src, msk, bits):
    mask = (1 << bits) - 1
    s = src & mask
    m = msk & mask
    d = 0
    k = 0
    for i in range(bits):
        if (m >> i) & 1:
            if (s >> k) & 1:
                d |= 1 << i
            k += 1
    return d


def ref_pext(src, msk, bits):
    mask = (1 << bits) - 1
    s = src & mask
    m = msk & mask
    d = 0
    k = 0
    for i in range(bits):
        if (m >> i) & 1:
            if (s >> i) & 1:
                d |= 1 << k
            k += 1
    return d


def ref_cmp_flags(a, b, bits):
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    ua, ub = a & mask, b & mask
    r = (ua - ub) & mask
    cf = 1 if ua < ub else 0
    zf = 1 if r == 0 else 0
    sf = 1 if (r & sign) else 0
    of = 1 if (((ua ^ ub) & (ua ^ r)) & sign) else 0
    pf = 1 if (bin(r & 0xFF).count("1") % 2 == 0) else 0
    packed4 = (cf << 0) | (zf << 1) | (sf << 2) | (of << 3)
    return (cf, zf, sf, of, pf), packed4


VALUES = (
    0x0000000000000000,
    0x0000000000000001,
    0x0000000000000003,
    0x0000000055AA55AA,
    0x000000007FFFFFFF,
    0x0000000080000000,
    0x0000000080000001,
    0x00000000FFFFFFFF,
    0xDEADBEEF00000000,
    0xDEADBEEF80000001,
    0x7FFFFFFFFFFFFFFF,
    0x8000000000000000,
    0x8000000000000001,
    0xFFFFFFFFFFFFFFFE,
    0xFFFFFFFFFFFFFFFF,
    0x0123456789ABCDEF,
    0xFEDCBA9876543210,
)
BZHI_INDICES = (
    0, 1, 7, 15, 31, 32, 33, 63, 64, 65, 255,
    0x100, 0x101, 0x120, 0x13F, 0x140, 0xFFFFFFFF, 0x8000000000000000,
)
BIT_MASKS = (
    0x0000000000000000,
    0x0000000000000001,
    0x0000000080000000,
    0x0000000055555555,
    0x00000000AAAAAAAA,
    0x00000000FFFFFFFF,
    0x8000000000000001,
    0x0F0F0F0F0F0F0F0F,
    0x5555555555555555,
    0xAAAAAAAAAAAAAAAA,
    0xFFFFFFFFFFFFFFFF,
)


def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections(original_path):
    """Verify shifts, BMI1, EVEX/XOP, ABI, addressing, PF/AF, prefix, and state rejections."""
    ret = Instruction.create(Code.RETNQ)
    mov_rax_rcx = Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.RCX)
    mov_r9_rcx = Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R9, R.RCX)
    cmp_rcx_rdx = Instruction.create_reg_reg(Code.CMP_R64_RM64, R.RCX, R.RDX)
    bzhi_rax = Instruction.create_reg_reg_reg(Code.VEX_BZHI_R64_RM64_R64, R.RAX, R.RCX, R.RDX)

    alt_accept = [
        ("bzhi64_r11_r8_rcx", (
            Instruction.create_reg_mem_reg(Code.VEX_BZHI_R64_RM64_R64, R.R11, MemoryOperand(R.R8), R.RCX),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R11),
            ret,
        ), "r11 = op_bzhi64(ld_1000_1, rcx, flags);"),
        ("bzhi32_self_alias", (
            mov_r9_rcx,
            Instruction.create_reg_reg_reg(Code.VEX_BZHI_R32_RM32_R32, R.R9D, R.R9D, R.EDX),
            Instruction.create_reg_reg(Code.MOV_R32_RM32, R.EAX, R.R9D),
            ret,
        ), "r9 = op_bzhi32((r9 & 0xFFFFFFFFull), (rdx & 0xFFFFFFFFull), flags);"),
        ("mulx64_alias_mem", (
            Instruction.create_reg_reg_mem(Code.VEX_MULX_R64_R64_RM64, R.R9, R.R9, MemoryOperand(R.RCX)),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R9),
            ret,
        ), "r9 = mulx_hi_1000;"),
        ("pdep64_r9_r10_rcx", (
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R10, R.RDX),
            Instruction.create_reg_reg_mem(Code.VEX_PDEP_R64_R64_RM64, R.R9, R.R10, MemoryOperand(R.RCX)),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R9),
            ret,
        ), "r9 = op_pdep64(r10, ld_1003_2);"),
        ("pext32_r11_r9_r8", (
            Instruction.create_reg_reg(Code.MOV_R32_RM32, R.R9D, R.ECX),
            Instruction.create_reg_reg_mem(Code.VEX_PEXT_R32_R32_RM32, R.R11D, R.R9D, MemoryOperand(R.R8)),
            Instruction.create_reg_reg(Code.MOV_R32_RM32, R.EAX, R.R11D),
            ret,
        ), "r11 = op_pext32((r9 & 0xFFFFFFFFull), ld_1003_2);"),
    ]
    for label, seq, expected in alt_accept:
        src, count = lift_code(label, 0x1000, encode_seq(*seq))
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if expected not in src:
            raise AssertionError(f"alt {label}: expected {expected!r} in:\n{src}")

    seg_pext = Instruction.create_reg_reg_mem(Code.VEX_PEXT_R64_R64_RM64, R.RAX, R.RCX, MemoryOperand(R.RDX))
    seg_pext.segment_prefix = R.GS
    evex_ins = Instruction.create_reg_reg_reg(Code.EVEX_VPADDD_XMM_K1Z_XMM_XMMM128B32, R.XMM0, R.XMM1, R.XMM2)
    evex_ins.op_mask = R.K1

    unit_cases = [
        # EVEX / XOP
        ("evex_vpaddd", (evex_ins, mov_rax_rcx, ret), "unsupported EVEX instruction"),
        ("xop_bextr", (Instruction.create_reg_reg_u32(Code.XOP_BEXTR_R32_RM32_IMM32, R.EAX, R.ECX, 0x0804), ret),
         "unsupported XOP instruction"),
        # BMI2 shifts & legacy shifts (ALL shifts refused)
        ("bmi2_rorx", (Instruction.create_reg_reg_u32(Code.VEX_RORX_R64_RM64_IMM8, R.RAX, R.RCX, 5), ret),
         "unsupported VEX instruction"),
        ("bmi2_shlx", (Instruction.create_reg_reg_reg(Code.VEX_SHLX_R64_RM64_R64, R.RAX, R.RCX, R.RDX), ret),
         "unsupported VEX instruction"),
        ("bmi2_shrx", (Instruction.create_reg_reg_reg(Code.VEX_SHRX_R32_RM32_R32, R.EAX, R.ECX, R.EDX), ret),
         "unsupported VEX instruction"),
        ("bmi2_sarx", (Instruction.create_reg_reg_reg(Code.VEX_SARX_R64_RM64_R64, R.RAX, R.RCX, R.RDX), ret),
         "unsupported VEX instruction"),
        ("legacy_shl_imm", (mov_rax_rcx, Instruction.create_reg_u32(Code.SHL_RM32_IMM8, R.EAX, 1), ret),
         "unsupported instruction"),
        ("legacy_shl_1", (mov_rax_rcx, Instruction.create_reg_u32(Code.SHL_RM64_1, R.RAX, 1), ret),
         "unsupported instruction"),
        ("legacy_shr_imm", (mov_rax_rcx, Instruction.create_reg_u32(Code.SHR_RM32_IMM8, R.EAX, 3), ret),
         "unsupported instruction"),
        ("legacy_sar_cl", (mov_rax_rcx, Instruction.create_reg_reg(Code.SAR_RM32_CL, R.EAX, R.CL), ret),
         "unsupported instruction"),
        ("legacy_ror_imm", (mov_rax_rcx, Instruction.create_reg_u32(Code.ROR_RM32_IMM8, R.EAX, 2), ret),
         "unsupported instruction"),
        # BMI1 instructions
        ("bmi1_andn", (Instruction.create_reg_reg_reg(Code.VEX_ANDN_R32_R32_RM32, R.EAX, R.ECX, R.EDX), ret),
         "unsupported VEX instruction"),
        ("bmi1_bextr", (Instruction.create_reg_reg_reg(Code.VEX_BEXTR_R32_RM32_R32, R.EAX, R.ECX, R.EDX), ret),
         "unsupported VEX instruction"),
        ("bmi1_blsi", (Instruction.create_reg_reg(Code.VEX_BLSI_R32_RM32, R.EAX, R.ECX), ret),
         "unsupported VEX instruction"),
        ("bmi1_blsmsk", (Instruction.create_reg_reg(Code.VEX_BLSMSK_R64_RM64, R.RAX, R.RCX), ret),
         "unsupported VEX instruction"),
        ("bmi1_blsr", (Instruction.create_reg_reg(Code.VEX_BLSR_R64_RM64, R.RAX, R.RCX), ret),
         "unsupported VEX instruction"),
        ("bmi1_tzcnt", (Instruction.create_reg_reg(Code.TZCNT_R64_RM64, R.RAX, R.RCX), ret),
         "unsupported instruction"),
        # Nonvolatile GPRs
        ("nonvol_rbx_dst", (Instruction.create_reg_reg_reg(Code.VEX_BZHI_R64_RM64_R64, R.RBX, R.RCX, R.RDX), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        ("nonvol_rsi_src", (Instruction.create_reg_reg_reg(Code.VEX_PDEP_R64_R64_RM64, R.RAX, R.RSI, R.RDX), ret),
         "nonvolatile GPR"),
        ("nonvol_r12_idx", (Instruction.create_reg_reg_reg(Code.VEX_BZHI_R64_RM64_R64, R.RAX, R.RCX, R.R12), ret),
         "nonvolatile GPR"),
        ("nonvol_rsp_lo", (Instruction.create_reg_reg_reg(Code.VEX_MULX_R64_R64_RM64, R.RAX, R.RSP, R.RCX), ret),
         "nonvolatile GPR"),
        # Writes to argument GPRs (rcx/rdx/r8) including MULX op0 & op1
        ("write_arg_rcx", (Instruction.create_reg_reg_reg(Code.VEX_BZHI_R64_RM64_R64, R.RCX, R.RDX, R.R8), mov_rax_rcx, ret),
         "cannot write argument GPR rcx"),
        ("write_arg_mulx_hi_edx", (Instruction.create_reg_reg_reg(Code.VEX_MULX_R32_R32_RM32, R.EDX, R.EAX, R.ECX), ret),
         "cannot write argument GPR rdx"),
        ("write_arg_mulx_lo_r8", (Instruction.create_reg_reg_reg(Code.VEX_MULX_R64_R64_RM64, R.RAX, R.R8, R.RCX), ret),
         "cannot write argument GPR r8"),
        ("write_arg_pext_r8", (Instruction.create_reg_reg_reg(Code.VEX_PEXT_R64_R64_RM64, R.R8, R.RCX, R.RDX), mov_rax_rcx, ret),
         "cannot write argument GPR r8"),
        # High-byte GPRs
        ("high_byte_ah", (mov_rax_rcx, Instruction.create_reg_reg(Code.MOV_R8_RM8, R.AL, R.AH), ret),
         "unsupported high-byte GPR"),
        ("high_byte_ch_dst", (Instruction.create_reg_reg(Code.MOV_R8_RM8, R.CH, R.DL), mov_rax_rcx, ret),
         "unsupported high-byte GPR"),
        # Displaced memory operands (must be refused!)
        ("disp_mem_bzhi", (Instruction.create_reg_mem_reg(
            Code.VEX_BZHI_R64_RM64_R64, R.RAX, MemoryOperand(R.RCX, displ=8, displ_size=1), R.RDX), ret),
         "displaced memory operand"),
        ("disp_mem_mulx", (Instruction.create_reg_reg_mem(
            Code.VEX_MULX_R32_R32_RM32, R.R9D, R.EAX, MemoryOperand(R.RDX, displ=4, displ_size=1)), ret),
         "displaced memory operand"),
        ("disp_mem_pdep", (Instruction.create_reg_reg_mem(
            Code.VEX_PDEP_R64_R64_RM64, R.RAX, R.RCX, MemoryOperand(R.R8, displ=8, displ_size=1)), ret),
         "displaced memory operand"),
        ("disp_mem_pext", (Instruction.create_reg_reg_mem(
            Code.VEX_PEXT_R32_R32_RM32, R.EAX, R.ECX, MemoryOperand(R.RCX, displ=-4, displ_size=1)), ret),
         "displaced memory operand"),
        ("disp_mem_store_mov", (mov_rax_rcx, Instruction.create_mem_reg(
            Code.MOV_RM64_R64, MemoryOperand(R.R8, displ=8, displ_size=1), R.RAX), ret),
         "displaced memory operand"),
        ("disp_mem_store_setcc", (cmp_rcx_rdx, mov_rax_rcx, Instruction.create_mem(
            Code.SETB_RM8, MemoryOperand(R.R8, displ=1, displ_size=1)), ret),
         "displaced memory operand"),
        # RIP-relative, SIB-indexed, bad base, non-[r8] stores
        ("rip_rel_pdep", (Instruction.create_reg_reg_mem(
            Code.VEX_PDEP_R64_R64_RM64, R.RAX, R.RCX, MemoryOperand(R.RIP, displ=0x1020, displ_size=4)), ret),
         "RIP-relative memory operand"),
        ("sib_idx_pext", (Instruction.create_reg_reg_mem(
            Code.VEX_PEXT_R64_R64_RM64, R.RAX, R.RCX, MemoryOperand(R.RCX, R.RDX, 8)), ret),
         "SIB index memory operand"),
        ("bad_base_rbx", (Instruction.create_reg_reg_mem(
            Code.VEX_MULX_R64_R64_RM64, R.R9, R.RAX, MemoryOperand(R.RBX)), ret),
         "bad base register in memory operand"),
        ("bad_base_rsp", (Instruction.create_reg_mem_reg(
            Code.VEX_BZHI_R32_RM32_R32, R.EAX, MemoryOperand(R.RSP), R.EDX), ret),
         "bad base register in memory operand"),
        ("non_r8_store_mov", (mov_rax_rcx, Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RCX), R.RAX), ret),
         "non-r8 store memory operand"),
        ("non_r8_store_setcc", (cmp_rcx_rdx, mov_rax_rcx, Instruction.create_mem(Code.SETE_RM8, MemoryOperand(R.RDX)), ret),
         "non-r8 store memory operand"),
        # Undefined GPR / undefined flag / PF & AF readers after BZHI
        ("undef_gpr_src", (Instruction.create_reg_reg_reg(Code.VEX_PDEP_R64_R64_RM64, R.RAX, R.R9, R.RDX), ret),
         "undefined GPR r9"),
        ("undef_gpr_idx", (Instruction.create_reg_reg_reg(Code.VEX_BZHI_R64_RM64_R64, R.RAX, R.RCX, R.R10), ret),
         "undefined GPR r10"),
        ("undef_rax_ret", (Instruction.create_reg_reg_reg(Code.VEX_PEXT_R64_R64_RM64, R.R9, R.RCX, R.RDX), ret),
         "return with undefined RAX"),
        ("undef_flags_setb", (mov_rax_rcx, Instruction.create_mem(Code.SETB_RM8, MemoryOperand(R.R8)), ret),
         "undefined flags read"),
        ("undef_flags_after_pdep", (Instruction.create_reg_reg_reg(Code.VEX_PDEP_R64_R64_RM64, R.RAX, R.RCX, R.RDX),
            Instruction.create_mem(Code.SETE_RM8, MemoryOperand(R.R8)), ret), "undefined flags read"),
        ("undef_pf_setp_after_bzhi", (cmp_rcx_rdx, bzhi_rax, Instruction.create_mem(Code.SETP_RM8, MemoryOperand(R.R8)), ret),
         "undefined flags read"),
        ("undef_pf_setnp_after_bzhi", (cmp_rcx_rdx, bzhi_rax, Instruction.create_mem(Code.SETNP_RM8, MemoryOperand(R.R8)), ret),
         "undefined flags read"),
        ("af_reader_lahf_after_bzhi", (bzhi_rax, Instruction.create(Code.LAHF), ret),
         "unsupported instruction"),
        ("af_reader_pushfq_after_bzhi", (bzhi_rax, Instruction.create(Code.PUSHFQ), ret),
         "unsupported instruction"),
        # Prefixes, branches, missing/trailing ret
        ("seg_prefix_pext", (seg_pext, ret), "unsupported instruction prefix"),
        ("branch_jmp", (mov_rax_rcx, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret),
         "non-leaf branch or call"),
        ("missing_ret", (mov_rax_rcx,), "missing terminal ret"),
        ("after_ret", (mov_rax_rcx, ret, mov_r9_rcx), "unreachable instruction after ret"),
    ]

    raw_cases = [
        ("lock_prefix", bytes.fromhex("f00fb109") + encode_seq(mov_rax_rcx, ret), "unsupported instruction prefix"),
        ("rep_prefix", bytes.fromhex("f3a4") + encode_seq(mov_rax_rcx, ret), "unsupported instruction prefix"),
        ("empty_stream", b"", "truncated instruction stream"),
    ]

    for label, seq, reason in unit_cases:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted rejected control")

    for label, raw, reason in raw_cases:
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted raw rejected control")

    # End-to-end PE patching and CLI non-traceback rejection controls.
    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_flags_preserve64")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry

    by_label = {lbl: (seq, reason) for lbl, seq, reason in unit_cases}
    pe_labels = (
        "evex_vpaddd", "bmi2_rorx", "legacy_shl_imm", "bmi1_blsi", "nonvol_rbx_dst",
        "write_arg_mulx_lo_r8", "disp_mem_bzhi", "undef_pf_setp_after_bzhi",
        "non_r8_store_setcc", "seg_prefix_pext",
    )
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_bmi2_bit_aot.py"
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        patched_path = tmp / "patched.dll"
        out_cpp = tmp / "rejected.cpp"
        for label in pe_labels:
            seq, reason = by_label[label]
            patch = encode_seq(*seq)
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds max_len ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off : file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["op_flags_preserve64"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"pe_{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"pe_{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched_path), "op_flags_preserve64", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 1 or "Traceback" in proc.stderr or "unsupported instruction prefix" not in proc.stderr:
            raise AssertionError(f"CLI rejection unexpected: rc={proc.returncode}, stderr={proc.stderr!r}")
        if out_cpp.exists():
            raise AssertionError("CLI failure left behind output file")

    print(
        f"rejections passed: {len(alt_accept)} alt + {len(unit_cases)} unit + {len(raw_cases)} raw + "
        f"{len(pe_labels)} PE-patched + CLI"
    )


def bind_exports(dll_path, prefix=""):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = ctypes.c_uint64
        funcs[name] = fn
    return lib, funcs


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def host_has_bmi2(lifted_lib):
    if hasattr(lifted_lib, "host_has_bmi2"):
        lifted_lib.host_has_bmi2.argtypes = []
        lifted_lib.host_has_bmi2.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_bmi2())
    try:
        return " bmi2" in Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace")
    except OSError:
        return False


def alloc_guarded(payload, offset=0):
    data = GUARD_PRE + b"\x77" * offset + payload + b"\x77" * (8 - offset) + GUARD_POST
    buf = ctypes.create_string_buffer(data, len(data))
    ptr = ctypes.addressof(buf) + len(GUARD_PRE) + offset
    return buf, ptr, offset, len(payload)


def read_guarded(buf, offset, length, label):
    raw = buf.raw
    if raw[:16] != GUARD_PRE or raw[-16:] != GUARD_POST:
        raise AssertionError(f"guard corrupted around {label}")
    if raw[16 : 16 + offset] != b"\x77" * offset or raw[16 + offset + length : -16] != b"\x77" * (8 - offset):
        raise AssertionError(f"padding corrupted around {label}")
    return raw[16 + offset : 16 + offset + length]


def check_discarded_read(fns, label):
    buf = ctypes.create_string_buffer(struct.pack("<Q", 0x0123456789ABCDEF))
    got = fns["op_bmi2_bit_discarded_read"](0xCAFEBABE12345678, ctypes.addressof(buf), 0)
    if got != 0xCAFEBABE12345678:
        raise AssertionError(f"{label} op_bmi2_bit_discarded_read(valid) = {got:#x}")
    try:
        fns["op_bmi2_bit_discarded_read"](0xCAFEBABE12345678, 0, 0)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} op_bmi2_bit_discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_bmi2_bit_discarded_read(NULL) dropped faulting m64 read")


def verify_positive_suite(native_fns, target_fns, label):
    cases = 0

    # 1) Register forms: BZHI, MULX (normal + op0==op1 alias), PDEP, PEXT
    for bits, bzhi_name in ((32, "op_bzhi32"), (64, "op_bzhi64")):
        for a in VALUES:
            for idx in BZHI_INDICES:
                want, _ = ref_bzhi(a, idx, bits)
                got_n = native_fns[bzhi_name](a, idx, 0)
                got_t = target_fns[bzhi_name](a, idx, 0)
                if got_n != want or got_t != want:
                    raise AssertionError(f"{label} {bzhi_name}({a:#x}, {idx:#x}): n={got_n:#x} t={got_t:#x} exp={want:#x}")
                cases += 1

    for bits, mulx_name, alias_name in (
        (32, "op_mulx32", "op_mulx32_alias"),
        (64, "op_mulx64", "op_mulx64_alias"),
    ):
        for s2 in VALUES:
            for s1 in VALUES:
                want_hi, want_lo = ref_mulx(s1, s2, bits)
                exp_out = struct.pack("<Q", want_lo) + OUT_FILL[8:]
                bo_n, po_n, oo_n, lo_n = alloc_guarded(OUT_FILL)
                bo_t, po_t, oo_t, lo_t = alloc_guarded(OUT_FILL)
                got_n = native_fns[mulx_name](s2, s1, po_n)
                got_t = target_fns[mulx_name](s2, s1, po_t)
                out_n = read_guarded(bo_n, oo_n, lo_n, f"{mulx_name}_n")
                out_t = read_guarded(bo_t, oo_t, lo_t, f"{mulx_name}_t")
                if got_n != want_hi or got_t != want_hi or out_n != exp_out or out_t != exp_out:
                    raise AssertionError(
                        f"{label} {mulx_name}({s2:#x}, {s1:#x}): hi n={got_n:#x} t={got_t:#x} exp={want_hi:#x}"
                    )
                got_an = native_fns[alias_name](s2, s1, 0)
                got_at = target_fns[alias_name](s2, s1, 0)
                if got_an != want_hi or got_at != want_hi:
                    raise AssertionError(
                        f"{label} {alias_name}({s2:#x}, {s1:#x}): n={got_an:#x} t={got_at:#x} exp={want_hi:#x}"
                    )
                cases += 2

    for bits, name, ref_fn in (
        (32, "op_pdep32", ref_pdep),
        (64, "op_pdep64", ref_pdep),
        (32, "op_pext32", ref_pext),
        (64, "op_pext64", ref_pext),
    ):
        for s in VALUES:
            for m in BIT_MASKS:
                want = ref_fn(s, m, bits)
                got_n = native_fns[name](s, m, 0)
                got_t = target_fns[name](s, m, 0)
                if got_n != want or got_t != want:
                    raise AssertionError(f"{label} {name}({s:#x}, {m:#x}): n={got_n:#x} t={got_t:#x} exp={want:#x}")
                cases += 1

    # 2) Flat zero-displacement [rcx|rdx|r8] memory forms (aligned and misaligned buffers)
    rng = random.Random(0xB2B17A07)
    mem_inputs = [
        (0x800000017FFFFFFF, 0x0000000000000020, 0x0F0F0F0F0F0F0F0F),
        (0xFFFFFFFF80000000, 0x0000000000000100, 0x55555555AAAAAAAA),
        (0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF, 0x8000000000000001),
    ] + [(rng.getrandbits(64), rng.getrandbits(64), rng.getrandbits(64)) for _ in range(16)]

    for a_val, b_val, r8_val in mem_inputs:
        a_bytes = struct.pack("<Q", a_val)
        b_bytes = struct.pack("<Q", b_val)
        r8_init = struct.pack("<Q", r8_val) + OUT_FILL[8:]
        for mis in (0, 3):
            ba, pa, oa, la = alloc_guarded(a_bytes, mis)
            bb, pb, ob, lb = alloc_guarded(b_bytes, mis)

            for bits, name in ((32, "op_bzhi32_mem"), (64, "op_bzhi64_mem")):
                want, _ = ref_bzhi(a_val, b_val, bits)
                got_n = native_fns[name](pa, pb, 0)
                got_t = target_fns[name](pa, pb, 0)
                if got_n != want or got_t != want:
                    raise AssertionError(f"{label} {name} mis={mis}: n={got_n:#x} t={got_t:#x} exp={want:#x}")
                cases += 1

            for bits, name in ((32, "op_mulx32_mem"), (64, "op_mulx64_mem")):
                want_hi, want_lo = ref_mulx(b_val, a_val, bits)
                exp_out = struct.pack("<Q", want_lo) + OUT_FILL[8:]
                bo_n, po_n, oo_n, lo_n = alloc_guarded(OUT_FILL, mis)
                bo_t, po_t, oo_t, lo_t = alloc_guarded(OUT_FILL, mis)
                got_n = native_fns[name](pa, b_val, po_n)
                got_t = target_fns[name](pa, b_val, po_t)
                out_n = read_guarded(bo_n, oo_n, lo_n, f"{name}_n")
                out_t = read_guarded(bo_t, oo_t, lo_t, f"{name}_t")
                if got_n != want_hi or got_t != want_hi or out_n != exp_out or out_t != exp_out:
                    raise AssertionError(f"{label} {name} mis={mis} mismatch")
                cases += 1

            for bits, name, ref_fn in (
                (32, "op_pdep32_mem", ref_pdep),
                (64, "op_pdep64_mem", ref_pdep),
                (32, "op_pext32_mem", ref_pext),
                (64, "op_pext64_mem", ref_pext),
            ):
                want = ref_fn(a_val, b_val, bits)
                got_n = native_fns[name](a_val, pb, 0)
                got_t = target_fns[name](a_val, pb, 0)
                if got_n != want or got_t != want:
                    raise AssertionError(f"{label} {name} mis={mis}: n={got_n:#x} t={got_t:#x} exp={want:#x}")
                cases += 1

            # op_mem_r8_source: reads [rcx], [rdx], [r8] BEFORE storing r10 to [r8] (mulx uses rdx=pb).
            r10 = ref_pdep(a_val, r8_val, 64)
            r11 = ref_pext(r10, b_val, 64)
            _, mul_lo = ref_mulx(pb, r8_val, 64)
            want_rax = mul_lo ^ r11
            exp_r8 = struct.pack("<Q", r10) + OUT_FILL[8:]
            bo_n, po_n, oo_n, lo_n = alloc_guarded(r8_init, mis)
            bo_t, po_t, oo_t, lo_t = alloc_guarded(r8_init, mis)
            got_n = native_fns["op_mem_r8_source"](pa, pb, po_n)
            got_t = target_fns["op_mem_r8_source"](pa, pb, po_t)
            out_n = read_guarded(bo_n, oo_n, lo_n, "r8_src_n")
            out_t = read_guarded(bo_t, oo_t, lo_t, "r8_src_t")
            if got_n != want_rax or got_t != want_rax or out_n != exp_r8 or out_t != exp_r8:
                raise AssertionError(f"{label} op_mem_r8_source mis={mis} mismatch")
            assert read_guarded(ba, oa, la, "a") == a_bytes and read_guarded(bb, ob, lb, "b") == b_bytes
            cases += 1

    # 3) BZHI flag definitions (CF, ZF, SF, OF=0)
    bzhi_flag_pairs = [
        (0x0000000000000000, 0),
        (0x0000000000000000, 32),
        (0x0000000000000000, 64),
        (0xFFFFFFFF80000000, 0),
        (0xFFFFFFFF80000000, 31),
        (0xFFFFFFFF80000000, 32),
        (0x8000000000000000, 63),
        (0x8000000000000000, 64),
        (0x0000000080000001, 0x100),
        (0x8000000080000001, 0x120),
        (0x8000000080000001, 0x140),
        (0x0123456789ABCDEF, 17),
    ]
    for bits, name in ((32, "op_bzhi32_flags"), (64, "op_bzhi64_flags")):
        seen_bzhi = set()
        for a, idx in bzhi_flag_pairs:
            _, packed = ref_bzhi(a, idx, bits)
            seen_bzhi.add(packed)
            exp_buf = struct.pack("<I", packed) + OUT_FILL[4:]
            bo_n, po_n, oo_n, lo_n = alloc_guarded(OUT_FILL)
            bo_t, po_t, oo_t, lo_t = alloc_guarded(OUT_FILL)
            got_n = native_fns[name](a, idx, po_n)
            got_t = target_fns[name](a, idx, po_t)
            buf_n = read_guarded(bo_n, oo_n, lo_n, f"{name}_n")
            buf_t = read_guarded(bo_t, oo_t, lo_t, f"{name}_t")
            if got_n != packed or got_t != packed or buf_n != exp_buf or buf_t != exp_buf:
                raise AssertionError(f"{label} {name}({a:#x}, {idx:#x}): n={got_n:#x} t={got_t:#x} exp={packed:#x}")
            cases += 1
        for bit_idx, fname in ((0, "CF"), (1, "ZF"), (2, "SF")):
            if {(p >> bit_idx) & 1 for p in seen_bzhi} != {0, 1}:
                raise AssertionError(f"{name} did not exercise both 0 and 1 for {fname}")
        if {(p >> 3) & 1 for p in seen_bzhi} != {0}:
            raise AssertionError(f"{name} observed non-zero OF")

    # 4) Flag preservation across CMP -> PDEP/PEXT/MULX -> SETcc (CF, ZF, SF, OF, PF)
    flag_pairs = [
        (0x0000000000000000, 0x0000000000000000),
        (0x0000000000000000, 0x0000000000000001),
        (0x0000000000000003, 0x0000000000000002),
        (0x0000000080000000, 0x0000000000000001),
        (0x000000007FFFFFFF, 0x00000000FFFFFFFF),
        (0x8000000000000000, 0x0000000000000001),
        (0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
    ] + [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(24)]

    for bits, name in ((32, "op_flags_preserve32"), (64, "op_flags_preserve64")):
        seen = set()
        for a, b in flag_pairs:
            f_tuple, packed4 = ref_cmp_flags(a, b, bits)
            seen.add(f_tuple[:4])
            exp_buf = struct.pack("<I", packed4) + OUT_FILL[4:]
            bo_n, po_n, oo_n, lo_n = alloc_guarded(OUT_FILL)
            bo_t, po_t, oo_t, lo_t = alloc_guarded(OUT_FILL)
            got_n = native_fns[name](a, b, po_n)
            got_t = target_fns[name](a, b, po_t)
            buf_n = read_guarded(bo_n, oo_n, lo_n, f"{name}_n")
            buf_t = read_guarded(bo_t, oo_t, lo_t, f"{name}_t")
            if got_n != packed4 or got_t != packed4 or buf_n != exp_buf or buf_t != exp_buf:
                raise AssertionError(f"{label} {name}({a:#x}, {b:#x}): n={got_n:#x} t={got_t:#x} exp={packed4:#x}")
            cases += 1
        for idx, fname in enumerate(("CF", "ZF", "SF", "OF")):
            if {t[idx] for t in seen} != {0, 1}:
                raise AssertionError(f"{name} did not exercise both 0 and 1 for {fname}")

    seen_pf = set()
    for a, b in flag_pairs:
        f_tuple, packed4 = ref_cmp_flags(a, b, 64)
        pf = f_tuple[4]
        seen_pf.add(pf)
        _, want_rax = ref_mulx(b, a, 64)
        exp_buf = bytes((pf,)) + OUT_FILL[1:]
        bo_n, po_n, oo_n, lo_n = alloc_guarded(OUT_FILL)
        bo_t, po_t, oo_t, lo_t = alloc_guarded(OUT_FILL)
        got_n = native_fns["op_flags_preserve_pf"](a, b, po_n)
        got_t = target_fns["op_flags_preserve_pf"](a, b, po_t)
        buf_n = read_guarded(bo_n, oo_n, lo_n, "flags_pf_n")
        buf_t = read_guarded(bo_t, oo_t, lo_t, "flags_pf_t")
        if got_n != want_rax or got_t != want_rax or buf_n != exp_buf or buf_t != exp_buf:
            raise AssertionError(f"{label} op_flags_preserve_pf({a:#x}, {b:#x}) mismatch")

        ba, pa, _, _ = alloc_guarded(struct.pack("<Q", a))
        bb, pb, _, _ = alloc_guarded(struct.pack("<Q", b))
        exp_mbuf = struct.pack("<I", packed4) + OUT_FILL[4:]
        bo_mn, po_mn, oo_mn, lo_mn = alloc_guarded(OUT_FILL)
        bo_mt, po_mt, oo_mt, lo_mt = alloc_guarded(OUT_FILL)
        got_mn = native_fns["op_flags_preserve_mem"](pa, pb, po_mn)
        got_mt = target_fns["op_flags_preserve_mem"](pa, pb, po_mt)
        buf_mn = read_guarded(bo_mn, oo_mn, lo_mn, "flags_mem_n")
        buf_mt = read_guarded(bo_mt, oo_mt, lo_mt, "flags_mem_t")
        if got_mn != packed4 or got_mt != packed4 or buf_mn != exp_mbuf or buf_mt != exp_mbuf:
            raise AssertionError(f"{label} op_flags_preserve_mem({a:#x}, {b:#x}) mismatch")
        cases += 2
    if seen_pf != {0, 1}:
        raise AssertionError(f"op_flags_preserve_pf did not exercise both 0 and 1 for PF: {seen_pf}")

    check_discarded_read(native_fns, "native")
    check_discarded_read(target_fns, label)
    return cases


def check_altered_and_mutants(lifted_fns, alt_native_fns, alt_lifted_fns):
    diverged = {}
    for name in ALTERED_EXPORTS:
        diffs = 0
        for a in VALUES[:8]:
            for b in VALUES[:8]:
                bo_n, po_n, _, _ = alloc_guarded(OUT_FILL)
                bo_l, po_l, _, _ = alloc_guarded(OUT_FILL)
                bo_o, po_o, _, _ = alloc_guarded(OUT_FILL)
                got_an = alt_native_fns[name](a, b, po_n)
                got_al = alt_lifted_fns[name](a, b, po_l)
                if got_an != got_al or bo_n.raw != bo_l.raw:
                    raise AssertionError(f"altered {name}({a:#x}, {b:#x}) native/lifted mismatch")
                got_orig = lifted_fns[name](a, b, po_o)
                if got_orig != got_al or bo_o.raw != bo_l.raw:
                    diffs += 1
        if diffs == 0:
            raise AssertionError(f"ALTER_BMI2_BIT failed to diverge on {name}")
        diverged[name] = diffs

    # Mutant 1: BZHI index masking to 5/6 bits instead of [7:0] (index=32/64 or 0x100).
    if lifted_fns["op_bzhi32"](0x12345678, 32, 0) != 0x12345678:
        raise AssertionError("mutant bzhi32(32) failed to preserve src")
    if lifted_fns["op_bzhi32"](0x12345678, 0x100, 0) != 0:
        raise AssertionError("mutant bzhi32(0x100) failed to return 0")
    if lifted_fns["op_bzhi64"](0x123456789ABCDEF0, 64, 0) != 0x123456789ABCDEF0:
        raise AssertionError("mutant bzhi64(64) failed to preserve src")
    if lifted_fns["op_bzhi64"](0x123456789ABCDEF0, 0x100, 0) != 0:
        raise AssertionError("mutant bzhi64(0x100) failed to return 0")
    # Check CF=1 when index==32 (32-bit) vs CF=0 when index==0x100.
    bo1, po1, _, _ = alloc_guarded(OUT_FILL)
    bo2, po2, _, _ = alloc_guarded(OUT_FILL)
    if (lifted_fns["op_bzhi32_flags"](0x12345678, 32, po1) & 1) != 1:
        raise AssertionError("mutant bzhi32_flags(32) CF!=1")
    if (lifted_fns["op_bzhi32_flags"](0x12345678, 0x100, po2) & 1) != 0:
        raise AssertionError("mutant bzhi32_flags(0x100) CF!=0")
    del bo1, bo2

    # Mutant 2: MULX op0 == op1 must yield high half, not low half.
    if lifted_fns["op_mulx32_alias"](0x80000000, 4, 0) != 2:
        raise AssertionError("mutant mulx32_alias returned low half instead of high half")
    if lifted_fns["op_mulx64_alias"](0x8000000000000000, 4, 0) != 2:
        raise AssertionError("mutant mulx64_alias returned low half instead of high half")

    # Mutant 3: PDEP/PEXT/MULX must preserve flags (e.g. CF=1, ZF=0, PF=0 from cmp(1, 20)).
    bo_f, po_f, oo_f, lo_f = alloc_guarded(OUT_FILL)
    packed = lifted_fns["op_flags_preserve32"](1, 20, po_f)
    if packed != 0b0101:  # CF=1, ZF=0, SF=1, OF=0
        raise AssertionError(f"flag-preservation witness failed: got {packed:#b}")
    assert read_guarded(bo_f, oo_f, lo_f, "flag_mutant")[:4] == struct.pack("<I", 0b0101)
    return diverged


def compile_o2_and_verify(original_path, native_fns, cxx_path):
    source_text, _ = translate(original_path, list(EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "bmi2_bit_o2.cpp"
        dll_path = tmp / "bmi2_bit_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'bmi2_bit_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind_exports(dll_path, "lifted_")
        try:
            return verify_positive_suite(native_fns, o2_fns, "lifted_o2")
        finally:
            o2_fns.clear()
            unload_lib(o2_lib)


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

    original_path, lifted_path, altered_path, altered_lifted_path, cxx_path = (
        Path(p) for p in sys.argv[1:6]
    )
    check_rejections(original_path)

    lifted_lib, lifted_fns = bind_exports(lifted_path, "lifted_")
    if not host_has_bmi2(lifted_lib):
        lifted_fns.clear()
        unload_lib(lifted_lib)
        print("Host lacks BMI2 (CPUID leaf 7 subleaf 0 EBX bit 8); skipping native differential check")
        sys.exit(77)

    orig_lib, orig_fns = bind_exports(original_path)
    alt_lib, alt_fns = bind_exports(altered_path)
    alt_lifted_lib, alt_lifted_fns = bind_exports(altered_lifted_path, "lifted_")
    try:
        lifted_cases = verify_positive_suite(orig_fns, lifted_fns, "lifted")
        diverged = check_altered_and_mutants(lifted_fns, alt_fns, alt_lifted_fns)
        o2_cases = compile_o2_and_verify(original_path, orig_fns, cxx_path)
    finally:
        for fn_map, lib in (
            (orig_fns, orig_lib),
            (lifted_fns, lifted_lib),
            (alt_fns, alt_lib),
            (alt_lifted_fns, alt_lifted_lib),
        ):
            fn_map.clear()
            unload_lib(lib)

    record_covered_forms(original_path)
    print(
        f"bmi2_bit_aot passed: {lifted_cases} lifted + {o2_cases} /O2 cases across {len(EXPORTS)} exports; "
        f"ALTER_BMI2_BIT diverged on {diverged}"
    )


if __name__ == "__main__":
    main()
