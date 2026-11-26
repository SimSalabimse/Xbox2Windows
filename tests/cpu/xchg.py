"""Compare native and translated XCHG, and check that the memory form stays atomic under contention.

CTest runs the parts as separate cases (check-native, check-lifted, check-controls, in parallel) and keeps
check-rejections on the aggregate. The `check` mode still runs every part serially for a direct run; the
parts above are not the only coverage of that mode, they are the same checks split up."""
from collections import Counter
import ctypes
import os
from pathlib import Path
import re
import sys
import threading

import pefile
from iced_x86 import Code, Decoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64
MASK64 = (1 << 64) - 1
REGISTERS = ("rax", "rcx", "r8", "r9")  # state[0..3]; entry: rax = r9 = x, rcx = r8 = y

# export: (operand 0, operand 1); a register is (name in REGISTERS, bits, bit position),
# a memory operand is ("slot", bits, byte offset).
VALUE_EXPORTS = {
    "xchg_al_cl": (("rax", 8, 0), ("rcx", 8, 0)),
    "xchg_ah_al": (("rax", 8, 8), ("rax", 8, 0)),
    "xchg_ah_ch": (("rax", 8, 8), ("rcx", 8, 8)),
    "xchg_r8b_al": (("r8", 8, 0), ("rax", 8, 0)),
    "xchg_ax_cx": (("rax", 16, 0), ("rcx", 16, 0)),
    "xchg_r9w_r8w": (("r9", 16, 0), ("r8", 16, 0)),
    "xchg_eax_ecx": (("rax", 32, 0), ("rcx", 32, 0)),
    "xchg_ecx_r9d": (("rcx", 32, 0), ("r9", 32, 0)),
    "xchg_eax_eax": (("rax", 32, 0), ("rax", 32, 0)),
    "xchg_rax_rcx": (("rax", 64, 0), ("rcx", 64, 0)),
    "xchg_r8_r9": (("r8", 64, 0), ("r9", 64, 0)),
    "xchg_m8_al": (("slot", 8, 0), ("rax", 8, 0)),
    "xchg_ah_m8": (("rax", 8, 8), ("slot", 8, 1)),
    "xchg_m16_cx": (("slot", 16, 2), ("rcx", 16, 0)),
    "xchg_m32_eax": (("slot", 32, 4), ("rax", 32, 0)),
    "xchg_r8d_m32": (("r8", 32, 0), ("slot", 32, 0)),
    "xchg_m64_r9": (("slot", 64, 0), ("r9", 64, 0)),
    "xchg_m64_misaligned": (("slot", 64, 3), ("rcx", 64, 0)),
}
TOKEN_EXPORTS = {"swap_tokens64": 8, "swap_tokens32": 4}
EXPORTS = (*VALUE_EXPORTS, *TOKEN_EXPORTS)

PLAIN_EXCHANGE = "template<class T, class V> T plain_exchange(T volatile* p, V v) { const T old = *p; *p = T(v); return old; }\n"
# Source mutations of the generated C++. Each control DLL must disagree with the reference somewhere.
CONTROLS = {
    # A flat load and store instead of the interlocked exchange: correct alone, racy under contention.
    "plain": (re.compile(r"_InterlockedExchange(?:8|16|64)?\("), "plain_exchange("),
    # Only the first of the two register writes of a register swap.
    "oneway": (re.compile(r"(\{ const word x = [^;]*; [^;]*;) [^;]*; \}"), r"\1 }"),
}

VALUES = (0x0123456789abcdef, 0xfedcba9876543210, 0, MASK64, 0x8000000080000080, 0x00000000ffffffff,
          0xffffffff00000000, 0x5a5a5a5a5a5a5a5a)
SLOT = bytes(range(0xa0, 0xb0))  # 16 bytes; every operand lies inside


def emit(image, output, control="lifted"):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            if name == "xchg_eax_eax" and body[6].code != Code.XCHG_RM32_R32:
                raise AssertionError(f"xchg_eax_eax assembled as {body[6]}, not 87 C0")
            code = lower(body, ["rcx = a; rdx = b; r8 = c; r9 = d;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word c, word d) {{\n{code}\n}}\n')
    text = "".join(parts)
    if control != "lifted":
        expression, replacement = CONTROLS[control]
        text, count = expression.subn(replacement, text)
        if not count:
            raise AssertionError(f"control {control} changed nothing")
        text = text.replace(INTEGER_HEADER, INTEGER_HEADER + PLAIN_EXCHANGE, 1)
        print(f"control {control}: {count} replacements")
    output.write_text(text, encoding="ascii")


def reference(name, x, y):
    """(state[0..3], slot bytes) after one exchange."""
    regs = {"rax": x, "r9": x, "rcx": y, "r8": y}
    slot = bytearray(SLOT)

    def get(operand):
        where, bits, at = operand
        if where == "slot":
            return int.from_bytes(slot[at:at + bits // 8], "little")
        return (regs[where] >> at) & ((1 << bits) - 1)

    def put(operand, value):
        where, bits, at = operand
        if where == "slot":
            slot[at:at + bits // 8] = value.to_bytes(bits // 8, "little")
        elif bits == 32:
            regs[where] = value  # a 32-bit register write zero-extends
        else:
            mask = ((1 << bits) - 1) << at
            regs[where] = (regs[where] & ~mask) | (value << at)

    first, second = VALUE_EXPORTS[name]
    a, b = get(first), get(second)
    put(first, b)
    put(second, a)
    return tuple(regs[r] for r in REGISTERS), bytes(slot)


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, U64, ctypes.c_void_p, ctypes.c_void_p], U64
    return lib


def value_mismatches(lib):
    found, cases = [], 0
    slot = (ctypes.c_uint8 * len(SLOT))()
    state = (U64 * 4)()
    for name in VALUE_EXPORTS:
        for x in VALUES:
            for y in VALUES:
                ctypes.memmove(slot, SLOT, len(SLOT))
                getattr(lib, name)(x, y, ctypes.addressof(slot), ctypes.addressof(state))
                got = (tuple(state), bytes(slot))
                want = reference(name, x, y)
                cases += 1
                if got != want:
                    found.append((name, x, y, want, got))
    return cases, found


def token_run(lib, name, threads, count):
    """Every thread exchanges its own distinct tokens into one shared slot. With atomic exchanges, the
    returned values and the final slot are exactly the tokens plus the initial value, each once."""
    width = TOKEN_EXPORTS[name]
    ctype = ctypes.c_uint64 if width == 8 else ctypes.c_uint32
    slot = ctype(0)
    tokens = [(ctype * count)(*((t + 1) << (8 * width - 8) | n for n in range(count))) for t in range(threads)]
    previous = [(ctype * count)() for _ in range(threads)]
    start = threading.Barrier(threads)
    fn = getattr(lib, name)

    def worker(t):
        start.wait()
        fn(ctypes.addressof(slot), ctypes.addressof(tokens[t]), ctypes.addressof(previous[t]), count)

    pool = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    seen = Counter(v for p in previous for v in p)
    seen[slot.value] += 1
    expected = Counter(v for t in tokens for v in t)
    expected[0] += 1
    return sum(((seen - expected) + (expected - seen)).values())


def token_errors(lib, rounds):
    threads = max(2, min(8, os.cpu_count() or 2))
    worst = {}
    for name in TOKEN_EXPORTS:
        worst[name] = max(token_run(lib, name, threads, 100_000) for _ in range(rounds))
    return threads, worst


def check_library(label, path):
    lib = load_lib(path)
    cases, bad = value_mismatches(lib)
    if bad:
        name, x, y, want, got = bad[0]
        raise AssertionError(f"{label}: {len(bad)} mismatches; first {name}({x:#x}, {y:#x}) = {got}, expected {want}")
    threads, errors = token_errors(lib, rounds=3)
    if any(errors.values()):
        raise AssertionError(f"{label}: token exchange lost or duplicated values: {errors}")
    print(f"{label}: {cases} value cases matched; {threads} threads x 100000 tokens x 3 rounds, "
          f"lost or duplicated tokens {errors}")


def check_controls(control_paths):
    for path in control_paths:
        control = Path(path).stem.rsplit("_", 1)[-1]
        lib = load_lib(path)
        cases, bad = value_mismatches(lib)
        threads, errors = token_errors(lib, rounds=3)
        print(f"control {control}: {len(bad)}/{cases} value mismatches; lost or duplicated tokens {errors}")
        if not bad and not any(errors.values()):
            raise AssertionError(f"control {control} matched every case")


def check(native_path, lifted_path, *control_paths):
    check_library("native", native_path)
    check_library("lifted", lifted_path)
    check_controls(control_paths)


def check_rejections():
    # An explicit LOCK on XCHG is redundant but still goes through the prefix rejection.
    try:
        lower(list(Decoder(64, bytes.fromhex("f0 48 87 01 c3"), ip=0x1000)), [], {}, {}, {})
    except ValueError as exc:
        if not str(exc).startswith("unsupported instruction prefix at 0x1000"):
            raise AssertionError(f"unexpected diagnostic {exc}") from exc
    else:
        raise AssertionError("accepted lock xchg")
    # 90 and 66 90 stay NOPs; 87 C0 writes EAX.
    for code, expected in (("90 c3", "block_1000: ;"), ("66 90 c3", "block_1000: ;"),
                           ("87 c0 c3", "rax = static_cast<uint32_t>(y);")):
        text = lower(list(Decoder(64, bytes.fromhex(code), ip=0x1000)), [], {}, {}, {})
        if expected not in text:
            raise AssertionError(f"{code}: {expected!r} not in {text!r}")
    print("rejected lock xchg; 90 and 66 90 lower as NOP, 87 c0 writes EAX")


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "emit":
        emit(Path(args[0]), Path(args[1]), *args[2:])
    elif mode == "check":
        check(*args)
        check_rejections()
    elif mode == "check-native":
        check_library("native", *args)
    elif mode == "check-lifted":
        check_library("lifted", *args)
    elif mode == "check-controls":
        check_controls(args)
    elif mode == "check-rejections":  # no fixture needed
        check_rejections()
    else:
        raise ValueError(mode)
