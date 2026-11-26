"""s_quadmask_b32 in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_s_quadmask_check.py
       startup_gfx2hlsl_s_quadmask_check.py dispatch <dxc> <offscreen_check> <warp|hardware>

RDNA 2 ISA SOP1: each output bit is the OR of one four-bit group of S0 (8-bit
result). SCC = (D != 0). Matches pixel_aot. Lane masks and descriptor sources
are refused. The reference is plain Python; the translated text is evaluated on
the CPU.

With `dispatch`, the lowering runs as a compute shader on the adapter (D3D12 UAV
readback via offscreen_check, same pattern as startup_gfx2hlsl_s_ff1_check /
startup_gfx2hlsl_s_abs_check). Each case stores D, SCC, and an SCC-dependent
branch token so the nibble collapse and nonzero SCC test are exercised on device.
"""
import re
import struct
import subprocess
import sys
import tempfile
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
# GPU cases: zero (SCC clear), low nibble, high bit, mid nibble-only.
DRAW = (0x00000000, 0x0000000f, 0x80000000, 0x000f0000)
SCC_BRANCH_TAKEN = 0x00a11ced
SCC_BRANCH_NOT = 0x00dead00
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


def controls():
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


def dispatch_quadmask(dxc, runner, adapter, cases, directory):
    """Translate s_quadmask_b32 once; feed each case through a UAV and read D, SCC, and a branch token."""
    lines, _ = translate(sop1('s_quadmask_b32', 3, 1))
    assert lines and not lines[0].startswith('refused'), lines
    fill = tuple(((i + 1) * 0x9e3779b1 & MASK) | 1 for i in range(len(cases)))
    result_width = 3
    hlsl = ['RWByteAddressBuffer data : register(u0);', '[numthreads(1, 1, 1)]', 'void main() {']
    for i, word in enumerate(cases):
        dst_byte = 4 * (len(fill) + result_width * i)
        hlsl.extend((
            '    {',
            '        uint s1 = data.Load(%d) ^ %s;' % (4 * i, '%#010xu' % (fill[i] ^ word)),
            '        uint s3;',
            '        bool scc = true;',
        ))
        hlsl.extend('        ' + line for line in lines)
        hlsl.append('        data.Store(%d, s3);' % dst_byte)
        hlsl.append('        data.Store(%d, uint(scc));' % (dst_byte + 4))
        hlsl.append('        if (scc) {')
        hlsl.append('            data.Store(%d, %#010xu);' % (dst_byte + 8, SCC_BRANCH_TAKEN))
        hlsl.append('        } else {')
        hlsl.append('            data.Store(%d, %#010xu);' % (dst_byte + 8, SCC_BRANCH_NOT))
        hlsl.append('        }')
        hlsl.append('    }')
    hlsl.append('}')

    src, cso, output = (directory / ('s_quadmask_b32' + suffix) for suffix in ('.hlsl', '.cso', '.bin'))
    src.write_text('\n'.join(hlsl) + '\n', encoding='ascii')
    compile_result = subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)],
                                    stderr=subprocess.PIPE, text=True)
    if compile_result.returncode or compile_result.stderr:
        raise AssertionError(compile_result.stderr or 'DXC exited with %d' % compile_result.returncode)
    word_count = len(fill) + result_width * len(cases)
    dump_words = (word_count + 3) // 4 * 4
    run_result = subprocess.run(
        [runner, adapter, '--raw-uav-dump', str(cso), '1', str(4 * dump_words), str(output)],
        capture_output=True, text=True)
    print(run_result.stdout, end='')
    if run_result.returncode == 77:
        sys.exit(77)
    if run_result.returncode or run_result.stderr:
        raise AssertionError(run_result.stderr or 'raw UAV dump exited with %d' % run_result.returncode)
    actual = struct.unpack('<%dI' % dump_words, output.read_bytes())
    assert actual[:len(fill)] == fill, 'operand region changed: %s' % (actual[:len(fill)],)
    return actual[len(fill):word_count]


def check_dispatch(dxc, runner, adapter):
    controls()
    runner = str(Path(runner).resolve())
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        got = dispatch_quadmask(dxc, runner, adapter, DRAW, Path(tmp))
        for i, word in enumerate(DRAW):
            result = quadmask32(word)
            scc_v = int(result != 0)
            branch = SCC_BRANCH_TAKEN if scc_v else SCC_BRANCH_NOT
            base = 3 * i
            label = 'dispatch s_quadmask_b32 %#010x' % word
            check('%s result' % label, got[base] == result,
                  'got %#x want %#x' % (got[base], result))
            check('%s SCC' % label, got[base + 1] == scc_v,
                  'got %#x want %#x' % (got[base + 1], scc_v))
            check('%s SCC branch' % label, got[base + 2] == branch,
                  'got %#x want %#x' % (got[base + 2], branch))
        check('dispatch quadmask(0) clears SCC',
              got[1] == 0 and got[2] == SCC_BRANCH_NOT,
              str((got[1], got[2])))
        check('dispatch quadmask(0x000f0000) is 0x10', got[3 * 3] == 0x10, hex(got[3 * 3]))
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        sys.exit(1)
    print('dispatch ok')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        check_dispatch(*sys.argv[2:])
    else:
        sys.exit(__doc__)
