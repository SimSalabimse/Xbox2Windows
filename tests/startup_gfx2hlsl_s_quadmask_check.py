"""s_quadmask_b32 in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_s_quadmask_check.py

RDNA 2 ISA SOP1: each output bit is the OR of one four-bit group of S0 (8-bit
result). SCC = (D != 0). Matches pixel_aot. Lane masks and descriptor sources
are refused. The reference is plain Python; the translated text is evaluated on
the CPU.
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
WORDS = (0x00000000, 0xffffffff, 0x00000001, 0x80000000, 0x11111111, 0x000f0000, 0x12345678, 0xf0f0f0f0)
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


def quadmask32(word):
    return sum((((word >> (4 * group)) & 0xf) != 0) << group for group in range(8))


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
lines, _ = translate(sop1('s_quadmask_b32', 3, 1))
assert lines and not lines[0].startswith('refused'), lines
for word in WORDS:
    out = run(lines, {'s1': word, 'scc': 0})
    want = (quadmask32(word), int(quadmask32(word) != 0))
    if (out['s3'], int(bool(out['scc']))) != want:
        bad = (hex(word), (out['s3'], out['scc']), want)
        break
check('s_quadmask_b32 SGPR source', bad is None, str(bad))

bad = None
for word in WORDS:
    stage = Probe()
    try:
        stage.one('s_quadmask_b32', 's_quadmask_b32 s3, %d' % word)
    except G.Unsupported as error:
        bad = ('refused %s' % error,)
        break
    out = run(stage.lines, {'scc': 0})
    want = (quadmask32(word), int(quadmask32(word) != 0))
    if (out['s3'], int(bool(out['scc']))) != want:
        bad = (hex(word), (out['s3'], out['scc']), want)
        break
check('s_quadmask_b32 immediate source', bad is None, str(bad))

# 0x000f0000 -> nibble 4 set -> bit 4 -> 0x10
lines, sym = translate(sop1('s_quadmask_b32', 3, 1), sym={1: ('const', 0x000f0000)})
check('s_quadmask_b32 folds a constant source',
      lines[0].startswith('s3 = 0x00000010u;') and sym.get(3) == ('const', 0x10)
      and any(l.startswith('scc = ') for l in lines),
      str((lines, sym.get(3))))

refused = [translate(sop1('s_quadmask_b32', d, s))[0][0] for d, s in (
    (3, VCC_LO), (3, EXEC_LO), (VCC_LO, 1), (EXEC_LO, 1))]
check('s_quadmask_b32 refuses lane-mask destinations/sources',
      all(r.startswith('refused: ') for r in refused), str(refused))

stage = Probe()
stage.masks.add(1)
try:
    stage.one('s_quadmask_b32', 's_quadmask_b32 s3, s1')
    check('s_quadmask_b32 refuses a tracked lane-mask source', False, stage.lines)
except G.Unsupported as error:
    check('s_quadmask_b32 refuses a tracked lane-mask source', 'lane mask' in str(error), str(error))

stage = Probe()
stage.sym[1] = ('desc', ('root', 0), 0, 0)
try:
    stage.one('s_quadmask_b32', 's_quadmask_b32 s3, s1')
    check('s_quadmask_b32 refuses a descriptor source', False, stage.lines)
except G.Unsupported as error:
    check('s_quadmask_b32 refuses a descriptor source', 'desc' in str(error), str(error))

# Negative control: drop the first nibble contribution; must diverge.
stage = Probe()
stage.one('s_quadmask_b32', 's_quadmask_b32 s3, s1')
mutated = [re.sub(r'\(\(s1 & 0xfu\) != 0u \? 1u : 0u\)', '0u', stage.lines[0], count=1)]
assert mutated != stage.lines, stage.lines
out_ok = run(stage.lines, {'s1': 0x0000000f, 'scc': 0})
out_bad = run([mutated[0]] + stage.lines[1:], {'s1': 0x0000000f, 'scc': 0})
check('mutation that drops the first nibble bit diverges',
      out_ok['s3'] == 1 and out_bad['s3'] == 0,
      str((out_ok['s3'], out_bad['s3'])))

if failures:
    print('FAILED: %s' % ', '.join(failures))
    sys.exit(1)
print('all ok')
