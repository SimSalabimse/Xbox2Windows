"""Check descriptor pointer propagation through s_add_u32 / s_addc_u32 in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_desc_ptr_prop_check.py

A shader advances a descriptor table or root pointer with `s_add_u32 sD, sP, <literal>` and
`s_addc_u32 sD+1, sP+1, 0`, into the same or another register pair, then loads descriptors through
the result. The load must read at the advanced byte offset. A non-literal addend must be refused,
never silently treated as the unchanged pointer.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH  ' + detail}")
    if not condition:
        failures.append(label)


class Pipe:
    sampler_heap = False

    def __init__(self):
        self.keys = []
        self.modified = set()

    def ud(self, key):
        return 'ud_%s' % '_'.join(map(str, key))

    def resource(self, key, kind):
        if (key, kind) not in self.keys:
            self.keys.append((key, kind))
        return 'r%d' % self.keys.index((key, kind))


def stage(symbols):
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.sym, s.lines = Pipe(), dict(symbols), []
    s.used_s, s.used_v, s.uniform_v, s.exec_known = set(), set(), set(), True
    s.masks, s.varying_s = set(), set()
    s.scc_uniform = True
    return s


def run(symbols, program, branches=False):
    """Symbols after the (op, operands) steps, or the Unsupported message."""
    s = stage(symbols)
    s.has_branches = branches
    s.ins = [(4 * i, 4, op, '') for i, (op, _operands) in enumerate(program)]
    try:
        for i, (op, operands) in enumerate(program):
            s.pc = 4 * i
            if op == 's_load_dwordx4':
                s.op_s_load_dwordx4(operands.split(', '), [])
            else:
                s.lower(op, '%s %s' % (op, operands), operands.split(', '), [])
    except G.Unsupported as error:
        return 'refused: %s' % error
    return s


TABLE = {8: ('ptr', ('table', 0))}   # a table pointer is 32 bits wide: no high word
ROOT = {8: ('ptr', ('root', 1)), 9: ('ptrhi', ('root', 1))}
ADVANCE = [('s_add_u32', 's20, s8, 0x40'), ('s_addc_u32', 's21, s9, 0')]
TABLE_ADVANCE = [('s_add_u32', 's20, s8, 0x40')]
LOAD = [('s_load_dwordx4', 's[4:7], s[20:21], 0x10')]

s = run(TABLE, TABLE_ADVANCE + LOAD)
check('table pointer advanced into another pair, descriptor read at pointer + 0x40 + 0x10',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x50, 0), str(s if isinstance(s, str) else s.sym))

same = [('s_add_u32', 's8, s8, 0x20')]
s = run(TABLE, same + [('s_load_dwordx4', 's[4:7], s[8:9], 0')])
check('table pointer advanced in place',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x20, 0), str(s if isinstance(s, str) else s.sym))

twice = [('s_add_u32', 's20, s8, 0x40'), ('s_add_u32', 's22, s20, 0x10')]
s = run(TABLE, twice + [('s_load_dwordx4', 's[4:7], s[22:23], 0x4')])
check('two advances add up',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x54, 0), str(s if isinstance(s, str) else s.sym))

s = run(ROOT, ADVANCE + LOAD)
check('root constant-buffer pointer advanced, buffer read at the advanced offset',
      not isinstance(s, str) and any('Load(80)' in line for line in s.lines), str(s if isinstance(s, str) else s.lines))

s = run(TABLE, [('s_add_u32', 's8, s8, s2')])
check('a register addend on a pointer is refused', isinstance(s, str) and 'non-literal' in s, str(s))

s = run(ROOT, ADVANCE + [('s_load_dwordx4', 's[4:7], s[20:21], s2')])
check('a register offset through an advanced pointer is still refused', isinstance(s, str), str(s))

# The untracked-offset mark of a branchy program must be refused here too, never a TypeError.
s = stage({8: ('ptr', ('root', 1), 'added'), 9: ('ptrhi', ('root', 1))})
try:
    s.descriptor('s[8:9]')
    check('an untracked-offset pointer used whole as a descriptor is refused', False, 'accepted')
except G.Unsupported:
    check('an untracked-offset pointer used whole as a descriptor is refused', True)

s = run(ROOT, ADVANCE)
try:
    s.descriptor('s[20:21]')
    check('an advanced pointer used whole as a descriptor is refused', False, 'accepted')
except G.Unsupported:
    check('an advanced pointer used whole as a descriptor is refused', True)

s = run(ROOT, ADVANCE, branches=True)
check('a pointer advance in a program with branches is refused', isinstance(s, str), str(s))

# The high-word add is the carry of the low-word add right before it, never of an unrelated carry in SCC.
s = run(ROOT, [('s_cmp_eq_u32', '0, 0'), ('s_addc_u32', 's9, s9, 0')])
check('s_addc_u32 without the low-word add is refused', isinstance(s, str), str(s))

s = run(ROOT, [ADVANCE[0], ('s_cmp_eq_u32', '0, 0'), ADVANCE[1]])
check('s_addc_u32 separated from its low-word add is refused', isinstance(s, str), str(s))

# A fold needs the complete matching pair: the load must not trust the low word alone.
LOW = ADVANCE[0]
s = run(ROOT, [LOW] + LOAD)
check('missing high-word add is refused', isinstance(s, str), str(s))

s = run(ROOT, [('s_add_u32', 's8, s8, 0x20'), ('s_load_dwordx4', 's[4:7], s[8:9], 0')])
check('missing high-word add in place is refused', isinstance(s, str), str(s))

s = run(ROOT, ADVANCE + [('s_mov_b32', 's21, 0')] + LOAD)
check('overwritten high word is refused', isinstance(s, str), str(s))

mixed = dict(ROOT)
mixed[10], mixed[11] = ('ptr', ('root', 2)), ('ptrhi', ('root', 2))
s = run(mixed, [LOW, ('s_addc_u32', 's21, s11, 0')] + LOAD)
check('high word from another pointer is refused', isinstance(s, str), str(s))

s = run(mixed, [LOW, ('s_addc_u32', 's21, s10, 0')] + LOAD)
check('high word from a different register pair is refused', isinstance(s, str), str(s))

s = run(ROOT, [('s_add_u32', 's20, s8, 0xfffff'), ('s_addc_u32', 's21, s9, 0')] + LOAD)
check('complete pair with a large low-word add is folded',
      not isinstance(s, str) and any('Load(%d)' % (0xfffff + 0x10) in line for line in s.lines), str(s))

# The high word is bound to the accumulated offset: the original word never stands in for an advanced one.
s = run(ROOT, ADVANCE + [('s_mov_b32', 's21, s9')] + LOAD)
check('completed high word overwritten by the original one is refused', isinstance(s, str), str(s))

SECOND = [('s_add_u32', 's22, s20, 0x20')]
s = run(ROOT, ADVANCE + SECOND + [('s_addc_u32', 's23, s9, 0'), ('s_load_dwordx4', 's[4:7], s[22:23], 0x10')])
check('second advance with the original high word is refused', isinstance(s, str), str(s))

s = run(ROOT, ADVANCE + SECOND + [('s_addc_u32', 's23, s21, 0'), ('s_load_dwordx4', 's[4:7], s[22:23], 0x10')])
check('second advance with the completed high word of the first pair is folded',
      not isinstance(s, str) and any('Load(%d)' % (0x40 + 0x20 + 0x10) in line for line in s.lines), str(s))

s = run(ROOT, [('s_add_u32', 's8, s8, 0x20'), ('s_addc_u32', 's9, s9, 0'), ('s_add_u32', 's8, s8, 0x20'),
               ('s_addc_u32', 's9, s9, 0'), ('s_load_dwordx4', 's[4:7], s[8:9], 0')])
check('in-place advance twice with each completed high word is folded',
      not isinstance(s, str) and any('Load(64)' in line for line in s.lines), str(s))

# Real entry state: a table pointer is one 32-bit register with no high word, so it advances with
# the low add alone and its neighbour register stays untouched.
def entry(params, kind='vs'):
    s = stage({})
    s.kind, s.pipe.root = kind, {'params': params, 'flags': 0}
    s.setup_entry()
    return s.sym


REAL = entry([{'type': 0}, {'type': 2}])
s = run(REAL, [('s_add_u32', 's20, s8, 0x40')] + LOAD)
check('real table pointer advanced by the low add alone, descriptor read at +0x50',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x50, 0), str(s))

s = run(REAL, [('s_add_u32', 's8, s8, 0x40'), ('s_load_dwordx4', 's[4:7], s[8:9], 0x10')])
check('real table pointer advanced in place keeps the next register symbol',
      not isinstance(s, str) and s.sym.get(9) == REAL[9] and s.sym.get(4) == ('desc', ('table', 0), 0x50, 0), str(s))

s = run(REAL, [('s_add_u32', 's20, s9, 0x40'), ('s_addc_u32', 's21, s10, 0')] + LOAD)
check('real root pointer pair advanced through the load',
      not isinstance(s, str) and any('Load(80)' in line for line in s.lines), str(s))

s = run(REAL, [('s_add_u32', 's20, s9, 0x40')] + LOAD)
check('real root pointer without its high add is refused', isinstance(s, str), str(s))

copied = dict(ROOT)
copied[30] = ('ptrhi', ('root', 1))
s = run(copied, [LOW, ('s_addc_u32', 's21, s30, 0')] + LOAD)
check('a copied high word completes the pair', not isinstance(s, str), str(s))

viam0 = dict(ROOT)
viam0['m0'] = ('ptr', ('root', 1))
s = run(viam0, [('s_add_u32', 's20, m0, 0x40'), ('s_addc_u32', 's21, s9, 0')] + LOAD)
check('a low word held in m0 completes the pair', not isinstance(s, str), str(s))

# A program with branches does not track the added offset: the load through the advanced pointer is a refusal
# (never a crash), and the same straight-line shape keeps folding.
IN_PLACE = [('s_add_u32', 's8, s8, 0x40'), ('s_load_dwordx4', 's[4:7], s[8:9], 0x10')]
s = run(TABLE, IN_PLACE, branches=True)
check('in-place pointer advance then load in a program with branches is refused, not a crash',
      isinstance(s, str) and 'untracked offset' in s, str(s))

s = run(TABLE, IN_PLACE)
check('the same in-place advance without branches still folds to +0x50',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x50, 0), str(s if isinstance(s, str) else s.sym))

# The pair is complete only after its s_addc_u32: an in-place pair folds, and a high word that was only
# copied (never advanced) must not stand in for it.
s = run(ROOT, [('s_add_u32', 's8, s8, 0x20'), ('s_addc_u32', 's9, s9, 0'),
               ('s_load_dwordx4', 's[4:7], s[8:9], 0x10')])
check('complete in-place pair folds to +0x30',
      not isinstance(s, str) and any('Load(%d)' % 0x30 in line for line in s.lines), str(s if isinstance(s, str) else s.lines))

s = run(ROOT, [ADVANCE[0], ('s_mov_b32', 's21, s9')] + LOAD)
check('out-of-place low add with a copied, unadvanced high word is refused', isinstance(s, str), str(s))

s = run(ROOT, [ADVANCE[0], ADVANCE[1]] + LOAD)
check('complete out-of-place pair leaves the original high source untouched',
      not isinstance(s, str) and s.sym.get(9) == ('ptrhi', ('root', 1)), str(s if isinstance(s, str) else s.sym))

if failures:
    print('FAILED: %s' % ', '.join(failures))
    sys.exit(1)
print('startup_gfx2hlsl_desc_ptr_prop: ok')
