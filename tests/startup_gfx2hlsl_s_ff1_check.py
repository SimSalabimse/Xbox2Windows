"""s_ff1_i32_b32 / s_ff1_i32_b64 of a general scalar in tools/startup/gfx2hlsl.py (#2022 Item F).

usage: startup_gfx2hlsl_s_ff1_check.py

RDNA 2 ISA SOP1: S_FF1_I32_B32 finds the lowest set bit of a 32-bit word (0xffffffff if zero);
S_FF1_I32_B64 scans the low word first, then the high word at offset 32. SCC is unchanged.
The compute-entry special case on exec_lo still folds to lane 0. Lane masks and descriptor
sources are refused. The reference is plain Python; the translated text is evaluated on the CPU.
"""
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
SOP1 = {name: op for op, (name, _) in D.SOP1_OPS.items()}
WORDS = (0x00000000, 0xffffffff, 0x80000001, 0x00010020, 0x55555555, 0xaaaaaaaa, 0x12345678, 0x10)
SCC_UNSET = 2
VCC_LO, EXEC_LO = 106, 126

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def sop1(name, sdst, ssrc):
    return struct.pack('<I', (0b101111101 << 23) | (sdst << 16) | (SOP1[name] << 8) | ssrc)


def ff1_32(word):
    word &= MASK
    return MASK if word == 0 else (word & -word).bit_length() - 1


def ff1_64(lo, hi):
    lo, hi = lo & MASK, hi & MASK
    if lo:
        return ff1_32(lo)
    if hi:
        return 32 + ff1_32(hi)
    return MASK


class Probe(G.Stage):
    def __init__(self, kind='vs'):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True
        self.kind = kind
        self.past_entry = False


def translate(words, masks=(), sym=None, kind='vs'):
    decoded = G.decode(words)
    assert len(decoded) == 1 and decoded[0][1] == len(words), decoded
    stage = Probe(kind=kind)
    stage.masks = set(masks)
    if sym:
        stage.sym.update(sym)
    try:
        stage.one(decoded[0][2], decoded[0][3])
    except G.Unsupported as error:
        stage.lines = ['refused: %s' % error]
    return stage.lines, stage.sym


def run(lines, env):
    env = dict(env)
    for line in lines:
        m = re.fullmatch(r'(\w+) = (.*);', line)
        assert m, line
        env[m.group(1)] = Expr(m.group(2)).run(env)[0]
    return env


# --- s_ff1_i32_b32 of a general SGPR ---
bad = None
lines, _ = translate(sop1('s_ff1_i32_b32', 3, 1))
assert lines and not lines[0].startswith('refused'), lines
for word in WORDS:
    out = run(lines, {'s1': word, 'scc': SCC_UNSET})
    want = (ff1_32(word), SCC_UNSET)
    if (out['s3'], out['scc']) != want:
        bad = (hex(word), (out['s3'], out['scc']), want)
        break
check('s_ff1_i32_b32 SGPR source', bad is None, str(bad))

bad = None
for word in WORDS:
    stage = Probe()
    try:
        stage.one('s_ff1_i32_b32', 's_ff1_i32_b32 s3, %d' % word)
    except G.Unsupported as error:
        bad = ('refused %s' % error,)
        break
    out = run(stage.lines, {'scc': SCC_UNSET})
    want = (ff1_32(word), SCC_UNSET)
    if (out['s3'], out['scc']) != want:
        bad = (hex(word), (out['s3'], out['scc']), want)
        break
check('s_ff1_i32_b32 immediate source', bad is None, str(bad))

# Constant source folds.
lines, sym = translate(sop1('s_ff1_i32_b32', 3, 1), sym={1: ('const', 0x10)})
check('s_ff1_i32_b32 folds a constant source',
      lines == ['s3 = 0x00000004u;'] and sym.get(3) == ('const', 4),
      str((lines, sym.get(3))))

# Compute-entry exec_lo still folds to lane 0.
lines, sym = translate(sop1('s_ff1_i32_b32', 5, EXEC_LO), kind='cs')
check('s_ff1_i32_b32 of compute-entry exec_lo folds to 0',
      lines == ['s5 = 0u;'] and sym.get(5) == ('const', 0),
      str((lines, sym.get(5))))

# Outside compute, or past entry, exec_lo is refused.
stage = Probe(kind='vs')
try:
    stage.one('s_ff1_i32_b32', 's_ff1_i32_b32 s5, exec_lo')
    check('s_ff1_i32_b32 of exec_lo outside compute is refused', False, stage.lines)
except G.Unsupported as error:
    check('s_ff1_i32_b32 of exec_lo outside compute is refused', 'exec_lo' in str(error), str(error))

# --- s_ff1_i32_b64 ---
# Encoding: ssrc names the low half of the pair (even SGPR).
bad = None
lines, _ = translate(sop1('s_ff1_i32_b64', 3, 4))
assert lines and not lines[0].startswith('refused'), lines
PAIRS = ((0, 0), (0x10, 0), (0, 0x10), (0x80000000, 1), (0, 0x80000000), (0xffffffff, 0xffffffff))
for lo, hi in PAIRS:
    out = run(lines, {'s4': lo, 's5': hi, 'scc': SCC_UNSET})
    want = (ff1_64(lo, hi), SCC_UNSET)
    if (out['s3'], out['scc']) != want:
        bad = (hex(lo), hex(hi), (out['s3'], out['scc']), want)
        break
check('s_ff1_i32_b64 SGPR pair', bad is None, str(bad))

lines, sym = translate(sop1('s_ff1_i32_b64', 3, 4),
                       sym={4: ('const', 0), 5: ('const', 0x10)})
check('s_ff1_i32_b64 folds a constant pair',
      lines == ['s3 = 0x00000024u;'] and sym.get(3) == ('const', 0x24),
      str((lines, sym.get(3))))

# Refusals: lane masks and descriptors.
refused = [translate(sop1('s_ff1_i32_b32', d, s))[0][0] for d, s in (
    (3, VCC_LO), (3, EXEC_LO), (VCC_LO, 1), (EXEC_LO, 1))]
check('s_ff1_i32_b32 refuses lane-mask destinations/sources',
      all(r.startswith('refused: ') for r in refused), str(refused))

stage = Probe()
stage.masks.add(1)
try:
    stage.one('s_ff1_i32_b32', 's_ff1_i32_b32 s3, s1')
    check('s_ff1_i32_b32 refuses a tracked lane-mask source', False, stage.lines)
except G.Unsupported as error:
    check('s_ff1_i32_b32 refuses a tracked lane-mask source', 'lane mask' in str(error), str(error))

stage = Probe()
stage.sym[1] = ('desc', ('root', 0), 0, 0)
try:
    stage.one('s_ff1_i32_b32', 's_ff1_i32_b32 s3, s1')
    check('s_ff1_i32_b32 refuses a descriptor source', False, stage.lines)
except G.Unsupported as error:
    check('s_ff1_i32_b32 refuses a descriptor source', 'desc' in str(error), str(error))

# Negative control: replace firstbitlow with a constant; evaluation must diverge.
stage = Probe()
stage.one('s_ff1_i32_b32', 's_ff1_i32_b32 s3, s1')
mutated = [re.sub(r'firstbitlow\(s1\)', '0u', stage.lines[0])]
assert mutated != stage.lines, stage.lines
out_ok = run(stage.lines, {'s1': 0x10, 'scc': 0})
out_bad = run(mutated, {'s1': 0x10, 'scc': 0})
check('mutation that drops firstbitlow diverges',
      out_ok['s3'] == 4 and out_bad['s3'] == 0,
      str((out_ok['s3'], out_bad['s3'])))

if failures:
    print('FAILED: %s' % ', '.join(failures))
    sys.exit(1)
print('all ok')
