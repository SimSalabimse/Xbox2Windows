"""Scalar bit counts and scans in tools/startup/gfx2hlsl.py (Frontier next batch 3 #1, #1723).

usage: startup_gfx2hlsl_s_bit_scan_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]

With `draw` the same instructions also run as a pixel shader on the adapter and the result words are read back
and compared with the integer reference (D3D12 readback, FXC 5_1 and DXC 6_0).

s_bcnt0/1_i32_b32/b64 count zero/one bits (SCC = result != 0). s_ff0/ff1_i32_b64 return the index of the lowest
zero/one bit of a 64-bit source, s_flbit_i32_b64 the number of leading zeros, s_flbit_i32_i64 the number of leading
bits equal to the sign bit; the scans return -1 when nothing qualifies and leave SCC alone. The translator refused
all of them with `Unsupported`. The instruction words are built from the SOP1 layout the decoder reads, the emitted
HLSL runs on the integer evaluator of startup_gfx2hlsl_bit_alu_check.py, and every result is compared with a plain
integer reference written from the ISA description (string and loop based, not the lowering's expressions).
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
failures = []
OPCODES = {'s_bcnt0_i32_b32': 0x0d, 's_bcnt0_i32_b64': 0x0e, 's_bcnt1_i32_b32': 0x0f, 's_bcnt1_i32_b64': 0x10,
           's_ff0_i32_b64': 0x12, 's_ff1_i32_b64': 0x14, 's_flbit_i32_b64': 0x16, 's_flbit_i32_i64': 0x18}
EXEC_LO, M0 = 126, 124


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def encode(name, sdst, ssrc0):
    return struct.pack('<I', (0b101111101 << 23) | (sdst << 16) | (OPCODES[name] << 8) | ssrc0)


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified, self.sampler_heap = {}, set(), set(), False


class Stage(G.Stage):
    def __init__(self, masks=()):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports, self.masks = {}, set(masks)
        self.pipe = Pipeline()


def translate(name, sdst, ssrc0, masks=()):
    stage = Stage(masks)
    decoded = G.decode(encode(name, sdst, ssrc0))
    for _offset, _size, op, asm in decoded:
        stage.one(op, asm)
    return [asm for _o, _s, _n, asm in decoded], stage.lines


def refused(name, sdst, ssrc0, masks=()):
    try:
        translate(name, sdst, ssrc0, masks)
    except G.Unsupported:
        return True
    return False


def run(lines, regs):
    """Evaluate the emitted `sN = expr;` and `scc = expr;` lines over the register dict."""
    env = dict(regs)
    for line in lines:
        target, expr = line.rstrip(';').split(' = ', 1)
        env[target] = Expr(expr).run(env)[0]
    return env


def bits(value, width):
    return [(value >> i) & 1 for i in range(width)]


def ref(name, value):
    """(result, scc or None) from the ISA text, with the source as a 32- or 64-bit integer."""
    width = 32 if name.endswith('b32') else 64
    b = bits(value, width)
    if 'bcnt0' in name:
        return b.count(0), True
    if 'bcnt1' in name:
        return b.count(1), True
    if 'ff0' in name or 'ff1' in name:
        want = 0 if 'ff0' in name else 1
        return next((i for i, x in enumerate(b) if x == want), MASK), False
    sign = b[63] if name.endswith('i64') else 0
    n = 0
    for x in reversed(b):
        if x != sign:
            return n, False
        n += 1
    return MASK, False


VALUES = [0, 1, 2, 0x80, 0xff, 0x7fffffff, 0x80000000, 0xffffffff, 0x12345678, 0xfffffffe, 0xdeadbeef]
WIDE = [0, 1, 1 << 31, 1 << 32, 1 << 33, 1 << 62, 1 << 63, (1 << 64) - 1, (1 << 63) - 1, (1 << 64) - 2,
        0x00000000ffffffff, 0xffffffff00000000, 0x0123456789abcdef, 0xfedcba9876543210, 0x8000000000000001,
        0xffffffff7fffffff] + [1 << k for k in range(64)] + [((1 << 64) - 1) ^ (1 << k) for k in range(64)]

for name in OPCODES:
    wide = not name.endswith('b32')
    asm, lines = translate(name, 3, 4 if wide else 2)
    src = 's[4:5]' if wide else 's2'
    check('%s decodes' % name, asm == ['%s s3, %s' % (name, src)], repr(asm))
    bad = []
    for value in (WIDE if wide else VALUES):
        regs = {'scc': 0, 's4': value & MASK, 's5': value >> 32, 's2': value & MASK}
        env = run(lines, regs)
        want, scc = ref(name, value)
        want_scc = int(want != 0) if scc else 0
        if env['s3'] != want or env['scc'] != want_scc:
            bad.append((hex(value), env['s3'], env['scc'], want, want_scc))
            if len(bad) == 3:
                break
    check('%s values' % name, not bad, repr(bad))
    scc_emitted = any(line.startswith('scc') for line in lines)
    check('%s SCC %s' % (name, 'follows result' if 'bcnt' in name else 'untouched'), scc_emitted == ('bcnt' in name),
          repr(lines))

# Refusals that must not widen: a 64-bit scan needs a register pair, 32-bit one register, and no lane-mask data.
check('b64 refuses an inline constant source', refused('s_ff1_i32_b64', 3, 0x80))
check('b32 refuses exec destination', refused('s_bcnt1_i32_b32', EXEC_LO, 2))
check('b32 refuses m0 destination', refused('s_bcnt1_i32_b32', M0, 2))
check('b32 refuses exec source', refused('s_bcnt1_i32_b32', 3, EXEC_LO))
check('b64 refuses a lane-mask source', refused('s_bcnt1_i32_b64', 3, 4, masks=(5,)))
check('b32 refuses a lane-mask source', refused('s_bcnt0_i32_b32', 3, 2, masks=(2,)))


# ---- a real draw: the translated pixel shader runs on the adapter and its words are read back ----------------
# Each channel loads a 64-bit (or 32-bit) literal into s4/s5 and runs the scalar op into s3.
# Counts and bit indices are 0..64 or 0xffffffff. XOR moves each expected word into a distinct
# finite normal float bit pattern before the RGBA32_FLOAT export. A raw small result can flush to zero.
DRAW_TRANSPORT_XOR = 0x3f000000
DRAW_VALUES = {
    32: [0, MASK, 0x80000000, 0x12345678, 1, 0xfffffffe, 0x0000ff00, 0xdeadbeef],
    64: [0, (1 << 64) - 1, 1 << 63, 1 << 32, 0x00000000ffffffff, 0xffffffff00000000, 0x0123456789abcdef,
         0xfedcba9876543210, 0x8000000000000001, (1 << 64) - 2, 1 << 31, 0xffffffff7fffffff],
}
POST_WORDS = bytes.fromhex('0f1800f80c0d0e0f000081bf')  # exp mrt0 v12..v15 done vm; s_endpgm


def sop1(op, sdst, ssrc0):
    return struct.pack('<I', (0b101111101 << 23) | (sdst << 16) | (op << 8) | ssrc0)


def lane_words(name, value):
    """Load the literals, evaluate the scalar operation, then encode its result for the export."""
    out = sop1(0x03, 4, 255) + struct.pack('<I', value & MASK)
    if not name.endswith('b32'):
        out += sop1(0x03, 5, 255) + struct.pack('<I', value >> 32)
    xor_s3_literal = (0b10 << 30) | (0x12 << 23) | (3 << 16) | (255 << 8) | 3
    return out + sop1(OPCODES[name], 3, 4) + struct.pack('<2I', xor_s3_literal, DRAW_TRANSPORT_XOR)


def ps_words_for(name, chunk):
    from startup_gfx2hlsl_bit_alu_check import PRE_WORDS
    body = b''
    for lane, value in enumerate(chunk):
        body += lane_words(name, value) + struct.pack('<I', 0x7e000000 | (12 + lane) << 17 | 1 << 9 | 3)
    return PRE_WORDS + body + POST_WORDS


def draw_check(dxc, renderer, adapter, fxc=None):
    import tempfile
    from startup_gfx2hlsl_bit_alu_check import vs_words, words_text
    from startup_flat_param_check import compile_hlsl, draw, root_blob
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for name in OPCODES:
                values = DRAW_VALUES[32 if name.endswith('b32') else 64]
                for index in range(0, len(values), 4):
                    chunk = values[index:index + 4]
                    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words_for(name, chunk), root_blob())
                    tag = '%s_%s_%d' % (name, kind, index)
                    ps = compile_hlsl(compiler, kind, pipe.hlsl_ps(), 'ps_' + tag, ps_profile, directory)
                    vs = compile_hlsl(compiler, kind, pipe.hlsl_vs(), 'vs_' + tag, vs_profile, directory)
                    expected = words_text(ref(name, v)[0] ^ DRAW_TRANSPORT_XOR for v in chunk)
                    run = draw(renderer, adapter, vs, ps, expected)
                    if run.returncode:
                        raise AssertionError('%s %s %s\n%s\n%s' % (kind, name, [hex(v) for v in chunk], run.stdout,
                                                                   run.stderr))
                    print('%s %s %s: %s' % (kind, name, [hex(v) for v in chunk], expected), flush=True)
                    if index == 0 and name == 's_bcnt1_i32_b64':
                        wrong = words_text(((ref(name, v)[0] + 1) & MASK) ^ DRAW_TRANSPORT_XOR for v in chunk)
                        run = draw(renderer, adapter, vs, ps, wrong)
                        if run.returncode != 1 or 'startup_flat_param mismatch' not in run.stderr:
                            raise AssertionError('%s wrong count did not produce a readback mismatch: %s%s' %
                                                 (kind, run.stdout, run.stderr))
                        print('%s wrong-expectation control: fails as expected' % kind, flush=True)


if failures:
    print('FAIL: %s' % ', '.join(failures))
    sys.exit(1)
if len(sys.argv) > 1:
    if sys.argv[1] != 'draw':
        raise SystemExit(__doc__)
    draw_check(*sys.argv[2:])
print('PASS')
