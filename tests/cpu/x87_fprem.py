"""Compare native FPREM and FPREM1 code with the checked entry of the x87 leaf-function profile.

Usage:
    python tests/cpu/x87_fprem.py asm <asm> <def>             regenerate the MASM source and exports
    python tests/cpu/x87_fprem.py emit <dll> <cpp> [control]   translate (and optionally mutate) the fixture
    python tests/cpu/x87_fprem.py check <harness> <native> <lifted> [controls...]

CTest runs the check-* parts as separate cases in parallel; `check` runs every part serially for a direct run.

Each export loads ST(1) from offset 16 and ST(0) from offset 0 of its input block, takes the partial
remainder, and stores the result, the divisor and records of the status word in its output block. The
translation is lower(..., x87=Checked(REJECT)) inside a checked entry, uint32_t name(word input,
word output, word* result), which returns the profile-rejection status and writes *result only for
status 0. The native exports run through x87_call of tests/cpu/x87_harness.asm.

The checked entry rejects two cases whose outcome the SDM leaves open. When the exponent of ST(0) exceeds
that of ST(1) by 64 or more, the instruction reduces ST(0) only partially, by an implementation-dependent
amount, and sets C2 (status "partial"). When the result is a NaN, the SDM does not say what C0, C1 and C3
become, so a function that reads the status word with FNSTSW is rejected there (status "nanremainder");
a function without FNSTSW keeps the NaN result, which follows the arithmetic NaN rules.

The model divides with Fraction arithmetic and shares no code with tools/x87_soft.h. For the native runs
it leaves the partial result and the status bits after a NaN result unknown and compares the rest. Native
C2 must be 1 exactly in the cases the checked entry rejects as partial, and the native result must be a
NaN exactly where the model's is. The loop exports repeat the instruction while C2 is 1, as compilers emit
fmod. Natively they reduce partially and loop; their final value is the full remainder, which does not
depend on how far each partial step reduced, and the model compares it. They are not run natively on NaN
results, where the SDM leaves C2 open and a loop could run forever.
"""
from fractions import Fraction
import ctypes
from pathlib import Path
import platform
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import x87_add as base  # noqa: E402  (tests/cpu is the script directory)
from x87_add import INDEFINITE, J, Q, classify, floor_log2, m80_bytes  # noqa: E402

INPUT, OUTPUT = base.INPUT, base.OUTPUT
A, B, C, CW, RESTORE = 0, 16, 32, 48, 50  # input offsets: ST(0), ST(1), a third value and two control words
STATUS = {"ok": 0, "partial": 4, "nanremainder": 5}  # x87_lower.STATUS, and 0 for success
SENTINEL = 0x5A5A5A5A5A5A5A5A  # *result before the call; the checked entry keeps it on a rejection
REJECT = "{{ x87_status = {status}; return 0; }}"
WORDS = [0x007F | 0x40 | pc << 8 | rc << 10 for pc in (0, 2, 3) for rc in range(4)]  # the twelve masked words

LOAD2 = [("fld", B), ("fld", A)]  # ST(1) = b, ST(0) = a
STORE2 = [("fstp", 0), ("fstp", 16)]  # the result to 0 and the divisor to 16
# name: (input kind, steps). Every export ends with MOV EAX, 0 and RET.
SEQUENCES = {
    "prem": ("pair", [*LOAD2, ("fprem",), ("sw", 32), *STORE2]),
    "prem1": ("pair", [*LOAD2, ("fprem1",), ("sw", 32), *STORE2]),
    # Without FNSTSW a NaN result is kept.
    "prem_value": ("pair", [*LOAD2, ("fprem",), *STORE2]),
    "prem1_value": ("pair", [*LOAD2, ("fprem1",), *STORE2]),
    # The flag readers: SAHF then JB (C0), JE (C3) and JP (C2), and TEST AH, 02h then JNE (C1).
    "prem_readers": ("pair", [*LOAD2, ("fprem",), ("readers", 32), *STORE2]),
    "prem1_readers": ("pair", [*LOAD2, ("fprem1",), ("readers", 32), *STORE2]),
    # The fmod and remainder loops: repeat while C2 (PF after SAHF) is 1, then store ST(0) and pop ST(1).
    "fmod_loop": ("pair", [*LOAD2, ("loop", "fprem"), ("fstp_st1",), ("fstp", 0)]),
    "remainder_loop": ("pair", [*LOAD2, ("loop", "fprem1"), ("fstp_st1",), ("fstp", 0)]),
    # A remainder of a remainder: FPREM, then FPREM1 of its result by the same divisor.
    "prem_twice": ("pair", [*LOAD2, ("fprem",), ("fprem1",), ("sw", 32), *STORE2]),
    # Eight values: c and five copies of it below the divisor.
    "prem_depth8": ("pair", [("fld", C), *[("fld_st", 0)] * 5, *LOAD2, ("fprem",), ("sw", 48), *STORE2,
                             ("fstp", 32), *[("pop",)] * 5]),
    # Under each of the twelve control words, which neither instruction reads.
    "prem_cw": ("cw", [("fldcw", CW), *LOAD2, ("fprem1",), ("sw", 32), *STORE2, ("fldcw", RESTORE)]),
}
READER_JUMPS = ("jb", "je", "jp")  # after SAHF: C0, C3 and C2
LOOPS = {"fmod_loop", "remainder_loop"}


# ---- MASM generation and decoding ------------------------------------------------------------------

def taken(label, offset, jump):
    """Store 1 at offset if jump is taken, else 0; MOV leaves the flags unchanged."""
    return [f"    mov byte ptr [rdx+{offset}], 1", f"    {jump} {label}", f"    mov byte ptr [rdx+{offset}], 0",
            f"{label}:"]


def asm_lines(step, name):
    op, *args = step
    if op == "fld":
        return [f"    fld tbyte ptr [rcx+{args[0]}]"]
    if op == "fstp":
        return [f"    fstp tbyte ptr [rdx+{args[0]}]"]
    if op == "fld_st":
        return [f"    fld st({args[0]})"]
    if op == "pop":
        return ["    fstp st(0)"]
    if op == "fstp_st1":
        return ["    fstp st(1)"]
    if op in ("fprem", "fprem1"):
        return [f"    {op}"]
    if op == "fldcw":
        return [f"    fldcw word ptr [rcx+{args[0]}]"]
    if op == "sw":
        return ["    fnstsw ax", "    and ah, 47h", f"    mov byte ptr [rdx+{args[0]}], ah"]
    if op == "readers":
        o = args[0]
        lines = ["    fnstsw ax", "    sahf"]
        for k, jump in enumerate(READER_JUMPS):
            lines += taken(f"{name}_{jump}", o + k, jump)
        return lines + ["    test ah, 02h", *taken(f"{name}_c1", o + 3, "jne")]
    if op == "loop":
        return [f"{name}_again:", f"    {args[0]}", "    fnstsw ax", "    sahf", f"    jp {name}_again"]
    raise AssertionError(op)


def expected_forms(step):
    """The x87 instructions of a step, as (mnemonic, operand): m80, m16, ax, st0 to st7 or none."""
    op, *args = step
    if op in ("fld", "fstp"):
        return [(op, "m80")]
    if op == "fld_st":
        return [("fld", f"st{args[0]}")]
    if op in ("pop", "fstp_st1"):
        return [("fstp", "st0" if op == "pop" else "st1")]
    if op in ("fprem", "fprem1"):
        return [(op, "")]
    if op == "fldcw":
        return [("fldcw", "m16")]
    if op in ("sw", "readers"):
        return [("fnstsw", "ax")]
    if op == "loop":
        return [(args[0], ""), ("fnstsw", "ax")]
    raise AssertionError(op)


def decoded_form(i):
    from iced_x86 import MemorySizeExt, Mnemonic as M, OpKind as K, Register as R
    name = next(n for n, v in vars(M).items() if v == i.mnemonic).lower()
    if i.op_count == 0:
        return name, ""
    if i.op0_kind == K.MEMORY:
        return name, f"m{8 * MemorySizeExt.size(i.memory_size)}"
    if i.op0_register == R.AX:
        return name, "ax"
    return name, f"st{i.op0_register - R.ST0}"


def write_asm(asm_path, def_path):
    lines = ["; coverage: source cpu_aot_x87_fprem",
             "; Generated by tests/cpu/x87_fprem.py asm; edit SEQUENCES there and regenerate.",
             "; Source-owned differential fixture for FPREM and FPREM1 in the checked entry of the x87",
             "; leaf-function profile. Each export reads an input block at RCX, writes an output block at RDX",
             "; and returns with the x87 stack empty.", ".code", ""]
    for name, (_, steps) in SEQUENCES.items():
        body = [line for s in steps for line in asm_lines(s, name)]
        lines += [f"{name} PROC FRAME", "    .endprolog", *body, "    mov eax, 0", "    ret", f"{name} ENDP", ""]
    lines.append("END")
    Path(asm_path).write_text("\n".join(lines) + "\n", encoding="ascii")
    Path(def_path).write_text("EXPORTS\n" + "".join(f"    {name}\n" for name in SEQUENCES), encoding="ascii")


# ---- Translation -----------------------------------------------------------------------------------

def checked_entry(name, code):
    return (f'extern "C" __declspec(dllexport) uint32_t {name}(word a, word b, word* result) {{\n'
            "    uint32_t x87_status = 0;\n"
            f"    const word value = [&]() -> word {{\n{code}\n    }}();\n"
            "    if (x87_status == 0) *result = value;\n"
            "    return x87_status;\n}\n")


PARTIAL_REJECT = r"if \(q == x87_prem_partial\) \{ x87_status = 4; return 0; \} "
NAN_REJECT = r"if \(q == x87_prem_nan\) \{ x87_status = 5; return 0; \} "
QBITS = "(q & 4 ? 0x0100 : 0) | (q & 2 ? 0x4000 : 0) | (q & 1 ? 0x0200 : 0)"
CONTROLS = {
    # A partial reduction runs on: the result keeps ST(0) and the status is 0.
    "nopartial": (re.compile(PARTIAL_REJECT), ""),
    # A NaN result in a function with FNSTSW is not rejected.
    "nonan": (re.compile(NAN_REJECT), ""),
    # The partial reduction starts at an exponent difference of 63.
    "d63": (re.compile(r"if \(d >= 64\) \{ out = a; return x87_prem_partial; \}"),
            "if (d >= 63) { out = a; return x87_prem_partial; }"),
    # FPREM1 truncates the quotient like FPREM.
    "truncate1": (re.compile(r"if \(nearest && \(r > y\.sig - r"), "if (false && (r > y.sig - r"),
    # A tie rounds the quotient up instead of to even.
    "tieup": (re.compile(r"\(r == y\.sig - r && \(q & 1\)\)"), "(r == y.sig - r)"),
    # FPREM1 keeps an ST(0) between half of ST(1) and ST(1) when the exponents differ by one.
    "halfnear": (re.compile(r"\} else if \(nearest && d == -1 && x\.sig > y\.sig\) \{"), "} else if (false) {"),
    # C1 and C3 exchanged.
    "qswap": (re.compile(re.escape(QBITS)), "(q & 4 ? 0x0100 : 0) | (q & 1 ? 0x4000 : 0) | (q & 2 ? 0x0200 : 0)"),
    # A zero remainder is +0.
    "zerosign": (re.compile(r"if \(r == 0\) \{ out = \{0, static_cast<uint16_t>\(a\.se & 0x8000\)\};"),
                 "if (r == 0) { out = {0, 0};"),
}


def emit(image, output, control="lifted"):
    import pefile
    from cpu_leaf_aot import export_rva
    from cpu_ranges import function_ranges
    from ffx_version_query import INTEGER_HEADER, lower, region
    from x87_lower import X87_HEADER, Checked, is_x87

    parts = [INTEGER_HEADER, X87_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name, (_, steps) in SEQUENCES.items():
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            decoded = [decoded_form(i) for i in body if is_x87(i)]
            wanted = [form for s in steps for form in expected_forms(s)]
            if decoded != wanted:
                raise AssertionError(f"{name} assembled as {decoded}, expected {wanted}")
            code = lower(body, ["rcx = a; rdx = b;"], {}, {}, {}, x87=Checked(REJECT))
            parts.append(checked_entry(name, code))
    text = "".join(parts)
    if control != "lifted":
        expression, replacement = CONTROLS[control]
        text, count = expression.subn(replacement, text)
        if not count:
            raise AssertionError(f"control {control} changed nothing")
        print(f"control {control}: {count} replacements")
    output.write_text(text, encoding="ascii")


# ---- Independent model -----------------------------------------------------------------------------
# Values are raw 80-bit pairs (sig, se), or None for a value the native run leaves unknown.

def encode(sign, q):
    """The m80 encoding of sign * q for an exact q >= 0: normal, or denormal below 2^-16382."""
    if q == 0:
        return 0, sign << 15
    e = floor_log2(q)
    if e >= 1 - 16383:
        return base.m80_exact(sign, q)
    sig = q / Fraction(2) ** (1 - 16383 - 63)
    assert sig.denominator == 1 and sig < J, q
    return int(sig), sign << 15


def nan_result(a, b):
    """A NaN or unsupported operand: the NaN rules of the arithmetic instructions (as for FADD)."""
    return base.add(a, b)


def prem(a, b, nearest, partial=True):
    """FPREM (nearest False) or FPREM1 of ST(0) = a by ST(1) = b: (outcome, result, q) with outcome "ok",
    "partial" (exponent difference 64 or more, when partial is set) or "nan", and q the quotient
    magnitude for "ok"."""
    ca, cb = classify(a), classify(b)
    if "unsupported" in (ca[0], cb[0]) or "nan" in (ca[0], cb[0]):
        return "nan", nan_result(a, b), None
    if ca[0] == "inf" or cb[0] == "zero":
        return "nan", INDEFINITE, None
    if ca[0] == "zero":
        return "ok", a, 0
    sign, x = ca[1], ca[2]
    if cb[0] == "inf":
        return "ok", encode(sign, x), 0
    y = cb[2]
    if partial and floor_log2(x) - floor_log2(y) >= 64:
        return "partial", None, None
    ratio = x / y
    q = ratio.numerator // ratio.denominator
    rest = ratio - q
    if nearest and (rest > Fraction(1, 2) or (rest == Fraction(1, 2) and q & 1)):
        q += 1
    r = x - q * y  # negative when FPREM1 rounded up: the result's sign is then the opposite of ST(0)
    return "ok", encode(sign ^ (r < 0), abs(r)), q


def c_bits(q):
    """AH & 47h after FNSTSW: C0 (0x01) = bit 2 of q, C1 (0x02) = bit 0, C2 (0x04) = 0, C3 (0x40) = bit 1."""
    return (0x01 if q & 4 else 0) | (0x02 if q & 1 else 0) | (0x40 if q & 2 else 0)


class Outcome:
    """A modelled run: the status, the output block and, for each output byte, the mask of its known bits."""

    def __init__(self, status, out, known):
        self.status, self.out, self.known = status, bytes(out), bytes(known)


C2 = 0x04  # in AH


def run(name, data, native=False):
    """The model of an export, for the checked translation or (native set) for a native run, where the
    partial result and the status bits other than C2 after a partial or NaN result are unknown."""
    steps = SEQUENCES[name][1]
    has_status_reader = any(s[0] in ("sw", "readers", "loop") for s in steps)
    out, known = bytearray(b"\xcc" * OUTPUT), bytearray(b"\xff" * OUTPUT)
    rejected = (bytes(b"\xcc" * OUTPUT), bytes(b"\xff" * OUTPUT))  # nothing is stored before a remainder
    slots = []
    ah, ah_known = 0, 0  # AH & 47h after the last remainder, and the mask of its known bits

    def write(offset, value, size):
        if value is None:
            known[offset:offset + size] = b"\0" * size
        else:
            out[offset:offset + size] = value.to_bytes(size, "little")

    def bit(mask):  # a reader's byte: 1 if the bit is set, with its known mask
        return int(ah & mask != 0), 0xFF if ah_known & mask else 0

    for op, *args in steps:
        if op == "fld":
            raw = int.from_bytes(data[args[0]:args[0] + 10], "little")
            slots.append((raw & (2 ** 64 - 1), raw >> 64))
        elif op == "fld_st":
            slots.append(slots[-1 - args[0]])
        elif op in ("fstp", "fstp_st1", "pop"):
            v = slots.pop()
            if op == "fstp":
                write(args[0], None if v is None else v[0] | v[1] << 64, 10)
            elif op == "fstp_st1":
                slots[-1] = v
        elif op in ("fprem", "fprem1", "loop"):
            nearest = (args[0] if op == "loop" else op) == "fprem1"
            a, b = slots[-1], slots[-2]
            if a is None or b is None:  # a remainder of an unknown partial result
                slots[-1], ah, ah_known = None, 0, 0
                continue
            outcome, result, q = prem(a, b, nearest, partial=not (native and op == "loop"))
            if outcome == "partial":
                if not native:
                    return Outcome(STATUS["partial"], *rejected)
                slots[-1], ah, ah_known = None, C2, C2  # C2 set; the rest is left to the implementation
            elif outcome == "nan":
                if not native and has_status_reader:
                    return Outcome(STATUS["nanremainder"], *rejected)
                slots[-1], ah, ah_known = result, 0, 0  # the SDM does not give the status bits
            else:
                slots[-1], ah, ah_known = result, c_bits(q), 0x47
        elif op == "sw":
            out[args[0]], known[args[0]] = ah, ah_known
        elif op == "readers":  # JB (C0), JE (C3), JP (C2) and JNE after TEST AH, 02h (C1)
            for k, mask in enumerate((0x01, 0x40, C2, 0x02)):
                out[args[0] + k], known[args[0] + k] = bit(mask)
        elif op == "fldcw":
            pass  # neither instruction reads the control word
        else:
            raise AssertionError(op)
    return Outcome(STATUS["ok"], out, known)


def check_model():
    """Spot checks of the model against values worked by hand."""
    one, two, three, seven = (J, 0x3FFF), (J, 0x4000), (J | Q, 0x4000), (J | Q | Q >> 1, 0x4001)
    rows = [
        (prem(seven, two, False), ("ok", one, 3)),                   # 7 = 3 * 2 + 1
        (prem(seven, two, True), ("ok", (J, 0xBFFF), 4)),           # 7 = 4 * 2 - 1, the tie to even
        (prem(three, two, True), ("ok", (J, 0xBFFF), 2)),           # 3 = 2 * 2 - 1, the tie to even
        (prem((J, 0x3FFE), (J, 0x3FFF), True), ("ok", (J, 0x3FFE), 0)),  # 0.5 by 1: the tie keeps q = 0
        (prem((J | Q, 0x3FFE), (J, 0x3FFF), True), ("ok", (J, 0xBFFD), 1)),  # 0.75 by 1: -0.25, q = 1
        (prem((J, 0x8000 | 0x4000), two, False), ("ok", (0, 0x8000), 1)),   # -2 by 2: -0
        (prem((J, 0x3FFF + 64), one, False)[0], "partial"),
        (prem((J, 0x3FFF + 63), one, False), ("ok", (0, 0), 1 << 63)),
        (prem((J | 3, 0), two, False), ("ok", (J | 3, 1), 0)),       # a pseudo-denormal re-encoded
        (prem(one, (J, 0x7FFF), False), ("ok", one, 0)),
        (prem((J, 0x7FFF), one, False), ("nan", INDEFINITE, None)),
        (prem(one, (0, 0x8000), False), ("nan", INDEFINITE, None)),
    ]
    for got, want in rows:
        if got != want:
            raise AssertionError(f"model gave {got}, expected {want}")
    return len(rows)


# ---- Cases -----------------------------------------------------------------------------------------

def specials():
    return base.special_m80()


def pairs(rng):
    """(a, b) pairs: every pair of special encodings, then finite pairs chosen to reach each branch."""
    s = specials()
    rows = [(a, b) for a in s for b in s]
    sign = lambda: rng.choice((0, 0x8000))  # noqa: E731
    sig = lambda: J | rng.getrandbits(63)  # noqa: E731
    for _ in range(900):  # exponent differences from -66 to 70 over the whole exponent range
        eb = rng.randint(1, 0x7FFE - 70)
        d = rng.randint(-66, 70)
        if 1 <= eb + d <= 0x7FFE:
            rows.append(((sig(), (eb + d) | sign()), (sig(), eb | sign())))
    for d in range(60, 68):  # each difference around the boundary, with short and full significands
        for _ in range(12):
            eb = rng.randint(1, 0x7FFE - 70)
            short = J | rng.getrandbits(8) << 55
            rows.append(((rng.choice((sig(), short)), (eb + d) | sign()), (rng.choice((sig(), short)), eb | sign())))
    for _ in range(120):  # partial reductions up to the largest differences
        d = rng.choice((rng.randint(64, 300), rng.randint(300, 0x7FFE - 1)))
        eb = rng.randint(1, 0x7FFE - d)
        rows.append(((sig(), (eb + d) | sign()), (sig(), eb | sign())))
    for d in (62, 63, 64, 65):  # denormal and pseudo-denormal divisors; d is the normalized difference
        for _ in range(16):
            bits = rng.randint(1, 63)
            b = (rng.getrandbits(bits) | 1 << (bits - 1), sign()) if rng.random() < 0.75 else (sig(), sign())
            eb = floor_log2(classify(b)[2]) + 16383
            if 1 <= eb + d <= 0x7FFE:
                rows.append(((sig(), (eb + d) | sign()), b))
    for _ in range(60):  # denormal dividends, results below the normal range
        rows.append(((rng.getrandbits(64) or 1, sign()), (rng.getrandbits(rng.randint(1, 64)) or 1, sign())))
        rows.append(((sig(), rng.randint(1, 70) | sign()), (sig(), rng.randint(1, 3) | sign())))
    for _ in range(200):  # exact multiples, ties and neighbours of ties
        b = (sig() & ~((1 << 40) - 1), rng.randint(0x3FFF - 200, 0x3FFF + 200) | sign())
        y = classify(b)[2]
        q = rng.getrandbits(rng.randint(1, 20)) | 1
        for x in (q * y, q * y + y / 2, (q + 1) * y + y / 2, q * y + y / 2 + y / 2 ** 16, q * y + y / 2 - y / 2 ** 16):
            rows.append((encode(rng.getrandbits(1), x), b))
    for _ in range(60):  # FPREM1 with |a| between half of |b| and |b| (exponent difference -1), and the tie
        b = (sig(), rng.randint(2, 0x7FFE) | sign())
        rows.append(((sig(), ((b[1] & 0x7FFF) - 1) | sign()), b))
        rows.append(((b[0], ((b[1] & 0x7FFF) - 1) | sign()), b))
    return rows


def cases():
    rng = random.Random(406)
    rows = pairs(rng)
    for name, (kind, _) in SEQUENCES.items():
        if kind == "pair":
            for a, b in rows:
                c = rng.choice(rows)[0]
                yield name, (m80_bytes(a) + m80_bytes(b) + m80_bytes(c)).ljust(INPUT, b"\0")
        else:  # the control words, over a sample of the pairs
            for k, (a, b) in enumerate(rows[::3]):
                block = bytearray((m80_bytes(a) + m80_bytes(b)).ljust(INPUT, b"\0"))
                block[CW:CW + 2] = WORDS[k % len(WORDS)].to_bytes(2, "little")
                block[RESTORE:RESTORE + 2] = (0x027F).to_bytes(2, "little")
                yield name, bytes(block)


# ---- Check -----------------------------------------------------------------------------------------

class CheckedRunner:
    """Calls the checked entries of a translated DLL with guarded buffers."""

    def __init__(self, harness_path, path):
        self.buffers = base.Runner(harness_path, path)
        self.lib = self.buffers.lib

    def __call__(self, name, data):
        guard, source, target = self.buffers._buffers(data)
        result = ctypes.c_uint64(SENTINEL)
        function = getattr(self.lib, name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        function.restype = ctypes.c_uint32
        status = function(ctypes.addressof(source) + base.GUARD, ctypes.addressof(target) + base.GUARD,
                          ctypes.byref(result))
        return status, self.buffers._output(name, data, guard, source, target), result.value


def expected_lifted(model):
    return model.status, model.out, 0 if model.status == STATUS["ok"] else SENTINEL


def is_nan(raw):
    return raw[1] & 0x7FFF == 0x7FFF and raw[0] & J and raw[0] << 1 & (2 ** 64 - 1)


def check(harness_path, native_path, lifted_path, *control_paths):
    check_light()
    expected = build_expected()
    check_lifted(harness_path, lifted_path, expected)
    check_native(harness_path, native_path, expected)
    check_controls(harness_path, expected, control_paths)


def check_light():
    print(f"the model agreed with {check_model()} remainders worked by hand")
    check_rejections()


def build_expected():
    return [(name, data, run(name, data)) for name, data in cases()]


def check_lifted(harness_path, lifted_path, expected):
    lifted = CheckedRunner(harness_path, lifted_path)
    statuses = dict.fromkeys(STATUS.values(), 0)
    bad = 0
    for name, data, model in expected:
        got = lifted(name, data)
        statuses[model.status] += 1
        if got != expected_lifted(model):
            if bad < 3:
                print(f"lifted mismatch: {name}(input {data[:32].hex()})\n  got      status {got[0]}, "
                      f"{got[1][:34].hex()}, {got[2]:#x}\n  expected status {model.status}, {model.out[:34].hex()}")
            bad += 1
    if bad:
        raise AssertionError(f"lifted: {bad}/{len(expected)} mismatches")
    print(f"{len(expected)} cases over {len(SEQUENCES)} exports matched the model in the checked translation (status, "
          "output block and guards, and *result, which only status 0 writes)")
    print("translated statuses: " + ", ".join(f"{name} {statuses[code]}" for name, code in STATUS.items()))


def check_native(harness_path, native_path, expected):
    native = base.Runner(harness_path, native_path)
    compared = partial_records = nan_results = looped = skipped = 0
    for name, data, model in expected:
        a = int.from_bytes(data[A:A + 10], "little")
        b = int.from_bytes(data[B:B + 10], "little")
        a, b = (a & (2 ** 64 - 1), a >> 64), (b & (2 ** 64 - 1), b >> 64)
        if name in LOOPS and prem(a, b, False)[0] == "nan":
            skipped += 1  # C2 after a NaN result is not specified: the native loop might not end
            continue
        want = run(name, data, native=True)
        got, state = native(name, data)
        if (state & 0xFFFF, (state >> 32) & 0xFFFF, (state >> 16) & 0x3800) != (0x027F, 0xFFFF, 0):
            raise AssertionError(f"native {name} left state {state:#x}; the profile requires an empty stack at TOP 0")
        # The known bits include C2 of every record: 1 where the checked entry rejects a partial reduction
        # and 0 after a full one.
        diff = [k for k in range(OUTPUT) if (got[k] ^ want.out[k]) & want.known[k]]
        if diff:
            raise AssertionError(f"native {name}(input {data[:32].hex()}) gave {got[:34].hex()}, expected "
                                 f"{want.out[:34].hex()} (bytes {diff[:8]} differ)")
        steps = SEQUENCES[name][1]
        first = next(s for s in steps if s[0] in ("fprem", "fprem1", "loop"))
        outcome = prem(a, b, (first[1] if first[0] == "loop" else first[0]) == "fprem1")[0]
        if name in LOOPS:
            looped += outcome == "partial"
        elif name != "prem_twice":  # the native result of a single step is a NaN exactly where the model's is
            result = int.from_bytes(got[0:10], "little")
            if bool(is_nan((result & (2 ** 64 - 1), result >> 64))) != (outcome == "nan"):
                raise AssertionError(f"native {name}(input {data[:32].hex()}): NaN result disagrees with {outcome}")
            nan_results += outcome == "nan"
            partial_records += outcome == "partial" and any(s[0] in ("sw", "readers") for s in steps)
        compared += 1
    print(f"native: {compared} cases matched the model where it is defined; C2 was 1 in the {partial_records} "
          f"single-step records the checked entry rejects as partial and 0 after every full reduction, the result "
          f"was a NaN in the {nan_results} single-step cases with a NaN model result, and the loops reduced partially "
          f"and iterated in {looped} cases; {skipped} loop cases with a NaN result ran only translated")
    print(f"processor: {platform.processor()}")


def check_controls(harness_path, expected, control_paths):
    for path in control_paths:
        control = Path(path).stem.rsplit("_", 1)[-1]
        run_control = CheckedRunner(harness_path, path)
        bad = sum(1 for name, data, model in expected if run_control(name, data) != expected_lifted(model))
        print(f"control {control}: {bad}/{len(expected)} mismatches")
        if not bad:
            raise AssertionError(f"control {control} matched every case")


def check_rejections():
    from iced_x86 import Decoder
    from ffx_version_query import lower
    from x87_lower import Checked

    def attempt(hex_bytes, checked=True):
        body = list(Decoder(64, bytes.fromhex(hex_bytes), ip=0x1000))
        try:
            lower(body, [], {}, {}, {}, x87=Checked(REJECT) if checked else True)
        except ValueError as exc:
            return str(exc)
        return None

    fld, pop, clear = "db 29", "dd d8", "31 c0 c3"  # fld tbyte ptr [rcx]; fstp st(0); xor eax, eax; ret
    fprem, fprem1, sw, sahf = "d9 f8", "d9 f5", "df e0", "9e"
    def pair(middle, end=clear):  # two values loaded, middle, both popped
        return f"{fld} {fld} {middle} {pop} {pop} {end}"

    zf = "reads ZF, which depends on a status-word bit"
    rejected = {
        "fprem in the fixed profile": (attempt(pair(fprem), False), "unsupported x87 operation"),
        "fprem1 in the fixed profile": (attempt(pair(fprem1), False), "unsupported x87 operation"),
        "fprem with one value": (attempt(f"{fld} {fprem} {pop} {clear}"), "x87 stack underflow"),
        "fprem1 on an empty stack": (attempt(f"{fprem1} {clear}"), "x87 stack underflow"),
        "TEST AH, 80h (the busy bit) after FPREM": (attempt(pair(f"{fprem} {sw} f6 c4 80 75 00")), zf),
        "FADD between FPREM and FNSTSW": (attempt(pair(f"{fprem} d8 c1 {sw} f6 c4 02 75 00")), zf),
        "a push in the fmod loop": (attempt(f"{fld} {fld} {fprem} {sw} {sahf} {fld} 7a f7 {pop} {pop} {pop} {clear}"),
                                    "meet at"),
        "a return with the status word in AX": (attempt(pair(f"{fprem} {sw}", "c3")),
                                                "RAX holds unknown status-word bits"),
    }
    for case, (message, needle) in rejected.items():
        if message is None or needle not in message:
            raise AssertionError(f"{case}: expected a rejection containing {needle!r}, got {message!r}")
    print(f"rejected {len(rejected)} sequences outside the profile: " + "; ".join(rejected))
    accepted = {
        "TEST AH, 02h (C1) after FPREM1": attempt(pair(f"{fprem1} {sw} f6 c4 02 75 00")),
        "SAHF and JE (C3) after FPREM": attempt(pair(f"{fprem} {sw} {sahf} 74 00")),
        "the fmod loop": attempt(pair(f"{fprem} {sw} {sahf} 7a f9")),
        "fprem without fnstsw": attempt(pair(fprem)),
    }
    for case, message in accepted.items():
        if message is not None:
            raise AssertionError(f"{case}: expected translation, got {message!r}")
    print(f"translated {len(accepted)} sequences: " + "; ".join(accepted))


if __name__ == "__main__":
    mode, *args = sys.argv[1:]  # check-light takes no paths
    first, second = (args + [None, None])[:2]
    rest = args[2:]
    if mode == "asm":
        write_asm(first, second)
    elif mode == "emit":
        emit(Path(first), Path(second), *rest)
    elif mode == "check":
        check(first, second, *rest)
    elif mode == "check-light":  # the hand-worked model and the rejections; no fixture needed
        check_light()
    elif mode == "check-lifted":
        check_lifted(first, second, build_expected())
    elif mode == "check-native":
        check_native(first, second, build_expected())
    elif mode == "check-controls":  # args: harness, unused, then the control DLLs
        check_controls(first, build_expected(), rest)
    else:
        raise ValueError(mode)
