"""s_bitset0_b32 and s_bitset1_b32 in tools/startup/gfx2hlsl.py (#2023 Item J).

usage: startup_gfx2hlsl_s_bitset_check.py

RDNA 2 ISA SOP1: S_BITSET0_B32 clears bit S0[4:0] of D in place; S_BITSET1_B32 sets that bit.
SCC is unchanged. The reference is plain Python integer arithmetic; the translated text is evaluated
on the CPU. Lane-mask and descriptor-bearing destinations or sources are refused. A mutation that
drops the bit update must disagree with the reference.
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
WORDS = (0x00000000, 0xffffffff, 0x80000001, 0x00010020, 0x55555555, 0xaaaaaaaa, 0x12345678)
INDEXES = (0, 1, 5, 16, 31, 32, 33, 0x21, 0xffffffe5)
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


def ref(clear, word, index):
    bit = 1 << (index & 31)
    return (word & ~bit if clear else word | bit) & MASK


for clear, name in ((True, 's_bitset0_b32'), (False, 's_bitset1_b32')):
    bad = None
    lines, _ = translate(sop1(name, 3, 1))
    assert lines and not lines[0].startswith('refused'), lines
    for word in WORDS:
        for index in INDEXES:
            out = run(lines, {'s1': index, 's3': word, 'scc': SCC_UNSET})
            want = (ref(clear, word, index), SCC_UNSET)
            if (out['s3'], out['scc']) != want:
                bad = (name, hex(word), hex(index), (out['s3'], out['scc']), want)
                break
        if bad:
            break
    check('%s SGPR bit index' % name, bad is None, str(bad))

    bad = None
    for word in WORDS:
        for index in (0, 1, 5, 16, 31, 32, 0x21):
            stage = Probe()
            try:
                stage.one(name, '%s s3, %d' % (name, index))
            except G.Unsupported as error:
                bad = ('refused %s' % error,)
                break
            out = run(stage.lines, {'s3': word, 'scc': SCC_UNSET})
            want = (ref(clear, word, index), SCC_UNSET)
            if (out['s3'], out['scc']) != want:
                bad = (hex(word), hex(index), (out['s3'], out['scc']), want)
                break
        if bad:
            break
    check('%s immediate bit index' % name, bad is None, str(bad))

    # Constant destination folds when the bit index is a known immediate (ssrc 128+5 = 5).
    lines, sym = translate(sop1(name, 3, 128 + 5), sym={3: ('const', 0x100)})
    want_const = (0x100 & ~(1 << 5) if clear else 0x100 | (1 << 5)) & MASK
    check('%s folds a constant destination with an immediate bit' % name,
          lines == ['s3 = 0x%08xu;' % want_const] and sym.get(3) == ('const', want_const),
          str((lines, sym.get(3), want_const)))

    refused = [translate(sop1(name, d, s))[0][0] for d, s in (
        (3, VCC_LO), (3, EXEC_LO), (VCC_LO, 1), (EXEC_LO, 1), (125, 1))]
    check('%s refuses lane-mask and discard destinations/sources' % name,
          all(r.startswith('refused: ') for r in refused), str(refused))

    stage = Probe()
    stage.masks.add(3)
    try:
        stage.one(name, '%s s3, 1' % name)
        check('%s refuses a tracked lane-mask destination' % name, False, stage.lines)
    except G.Unsupported as error:
        check('%s refuses a tracked lane-mask destination' % name, 'lane mask' in str(error), str(error))

    stage = Probe()
    stage.sym[3] = ('desc', ('root', 0), 0, 0)
    try:
        stage.one(name, '%s s3, 1' % name)
        check('%s refuses a descriptor destination' % name, False, stage.lines)
    except G.Unsupported as error:
        check('%s refuses a descriptor destination' % name, 'desc' in str(error), str(error))

# Negative control: drop the bit update from the emitted expression; the CPU evaluation must diverge.
stage = Probe()
stage.one('s_bitset1_b32', 's_bitset1_b32 s3, s1')
mutated = [re.sub(r'\(s3 \| \(1u << \(s1 & 31\)\)\)', 's3', stage.lines[0])]
assert mutated != stage.lines, stage.lines
out_ok = run(stage.lines, {'s1': 5, 's3': 0x100, 'scc': 0})
out_bad = run(mutated, {'s1': 5, 's3': 0x100, 'scc': 0})
check('mutation that drops the set diverges', out_ok['s3'] == 0x120 and out_bad['s3'] == 0x100,
      str((out_ok['s3'], out_bad['s3'])))

if failures:
    print('FAILED: %s' % ', '.join(failures))
    sys.exit(1)
print('all ok')
