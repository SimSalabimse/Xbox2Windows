"""s_flbit_i32_b32 in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_s_flbit_check.py

RDNA 2 ISA SOP1: S_FLBIT_I32_B32 returns the number of leading zero bits
(0xffffffff if the word is zero). SCC is unchanged. Matches pixel_aot /
dpp_ffbh_u32. Lane masks and descriptor sources are refused. The reference is
plain Python; the translated text is evaluated on the CPU.
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
WORDS = (0x00000000, 0x00000001, 0x80000000, 0xffffffff, 0x00010020, 0x55555555, 0x12345678)
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


def ref_flbit(word):
    word &= MASK
    return MASK if word == 0 else 31 - (word.bit_length() - 1)


class Probe(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True


def translate(words, masks=(), sym=None):
    decoded = G.decode(words)
    assert len(decoded) == 1 and decoded[0][1] == len(words), decoded
    stage = Probe()
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


bad = None
lines, _ = translate(sop1('s_flbit_i32_b32', 3, 1))
assert lines and not lines[0].startswith('refused'), lines
for word in WORDS:
    out = run(lines, {'s1': word, 'scc': SCC_UNSET})
    want = (ref_flbit(word), SCC_UNSET)
    if (out['s3'], out['scc']) != want:
        bad = (hex(word), (out['s3'], out['scc']), want)
        break
check('s_flbit_i32_b32 SGPR source', bad is None, str(bad))

bad = None
for word in WORDS:
    stage = Probe()
    try:
        stage.one('s_flbit_i32_b32', 's_flbit_i32_b32 s3, %d' % word)
    except G.Unsupported as error:
        bad = ('refused %s' % error,)
        break
    out = run(stage.lines, {'scc': SCC_UNSET})
    want = (ref_flbit(word), SCC_UNSET)
    if (out['s3'], out['scc']) != want:
        bad = (hex(word), (out['s3'], out['scc']), want)
        break
check('s_flbit_i32_b32 immediate source', bad is None, str(bad))

lines, sym = translate(sop1('s_flbit_i32_b32', 3, 1), sym={1: ('const', 0x10)})
# 0x10 bit_length=5 -> firstbithigh=4 -> 31-4=27
check('s_flbit_i32_b32 folds a constant source',
      lines == ['s3 = 0x0000001bu;'] and sym.get(3) == ('const', 0x1b),
      str((lines, sym.get(3))))

lines, sym = translate(sop1('s_flbit_i32_b32', 3, 1), sym={1: ('const', 0)})
check('s_flbit_i32_b32 of zero is -1',
      lines == ['s3 = 0xffffffffu;'] and sym.get(3) == ('const', MASK),
      str((lines, sym.get(3))))

refused = [translate(sop1('s_flbit_i32_b32', d, s))[0][0] for d, s in (
    (3, VCC_LO), (3, EXEC_LO), (VCC_LO, 1), (EXEC_LO, 1))]
check('s_flbit_i32_b32 refuses lane-mask destinations/sources',
      all(r.startswith('refused: ') for r in refused), str(refused))

stage = Probe()
stage.masks.add(1)
try:
    stage.one('s_flbit_i32_b32', 's_flbit_i32_b32 s3, s1')
    check('s_flbit_i32_b32 refuses a tracked lane-mask source', False, stage.lines)
except G.Unsupported as error:
    check('s_flbit_i32_b32 refuses a tracked lane-mask source', 'lane mask' in str(error), str(error))

stage = Probe()
stage.sym[1] = ('desc', ('root', 0), 0, 0)
try:
    stage.one('s_flbit_i32_b32', 's_flbit_i32_b32 s3, s1')
    check('s_flbit_i32_b32 refuses a descriptor source', False, stage.lines)
except G.Unsupported as error:
    check('s_flbit_i32_b32 refuses a descriptor source', 'desc' in str(error), str(error))

stage = Probe()
stage.one('s_flbit_i32_b32', 's_flbit_i32_b32 s3, s1')
mutated = [re.sub(r'31u - firstbithigh\(s1\)', 'firstbithigh(s1)', stage.lines[0])]
assert mutated != stage.lines, stage.lines
out_ok = run(stage.lines, {'s1': 0x10, 'scc': 0})
out_bad = run(mutated, {'s1': 0x10, 'scc': 0})
check('mutation that drops the 31-offset diverges',
      out_ok['s3'] == 27 and out_bad['s3'] == 4,
      str((out_ok['s3'], out_bad['s3'])))

if failures:
    print('FAILED: %s' % ', '.join(failures))
    sys.exit(1)
print('all ok')
