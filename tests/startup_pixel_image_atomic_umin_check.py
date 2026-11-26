"""Check gfx2hlsl's pixel-stage image_atomic_umin on a hand-assembled gfx1013 program (M200-01 A).

The pixel program loads an image descriptor from descriptor table 0, sets the texel address (1, 2)
and the high-bit unsigned operand 0x80000011, runs image_atomic_umin ... dmask:0x1 dim:SQ_RSRC_IMG_2D,
and exports that word to mrt0. Words were assembled by hand from the MIMG layout and are decoded back
by tools/disasm_rdna2.py; --binary reassembles the source and compares the words.

Without arguments: translation controls and refusals. With draw: also compile the translation with DXC
and draw it through offscreen_check --startup-image-atomic-umin, which reads every texel of a 4x4
R32_UINT image back. The UAV starts as 0xffffffff; umin with 0x80000011 writes that operand at (1, 2).
Unsigned ordering differs from signed smin on that operand (bit 31 set). Negative controls: Max-for-Min
mutant, and an inactive-lane case that must leave the texel at its starting value.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'tools'))
from startup_image_3d_check import FULLSCREEN_VS, G, VS_WORDS, assemble, compile_hlsl, root_blob  # noqa: E402
import disasm_rdna2 as D  # noqa: E402

OPERAND = 0x80000011
INITIAL = 0xffffffff
EXPECTED = OPERAND  # umin(0xffffffff, 0x80000011) as unsigned

SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
v_mov_b32 v0, 1
v_mov_b32 v1, 2
v_mov_b32 v4, 0x80000011
s_waitcnt lgkmcnt(0)
image_atomic_umin v4, v[0:1], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v4, v4, v4 done vm
s_endpgm
"""
ATOMIC = bytes.fromhex('080154f000040400')
PREFIX = bytes.fromhex('04040cf4000000fa' '8102007e' '8202027e' 'ff02087e11000080' '7fc08cbf')
SUFFIX = bytes.fromhex('703f8cbf' '0f1800f804040404' '000081bf')
WORDS = PREFIX + ATOMIC + SUFFIX
ATOMIC_LINE = 'InterlockedMin(w0[uint2(v0, v1)], v4);'
GUARDED_LINE = 'if (exec & 1) ' + ATOMIC_LINE


def cmpx_eq(constant):
    """v_cmpx_eq_u32_e32 <constant>, v0: VOPC writes only exec in gfx10."""
    return struct.pack('<I', 0x7c000000 | (0xd2 << 17) | (0 << 9) | (0x80 + constant))


EXEC_ON = PREFIX + cmpx_eq(1) + ATOMIC + SUFFIX
EXEC_OFF = PREFIX + cmpx_eq(2) + ATOMIC + SUFFIX


def translate(words):
    return G.Pipeline(VS_WORDS, words, root_blob())


def refused(words, *needles):
    try:
        translate(words)
    except G.Unsupported as error:
        text = str(error)
        assert all(n in text for n in needles), (needles, text)
        return text
    raise AssertionError('accepted: %s' % (needles,))


def with_atomic(atomic):
    return PREFIX + atomic + SUFFIX


def patched(byte, mask, value=None):
    atomic = bytearray(ATOMIC)
    atomic[byte] = value if value is not None else atomic[byte] | mask
    return bytes(atomic)


def controls():
    text = D.decode_instruction(ATOMIC, 0)[2]
    assert text == 'image_atomic_umin v4, v[0:1], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D', text
    pipe = translate(WORDS)
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'RWTexture2D<uint> w0 : register(u0);' in ps, ps
    assert '    ' + ATOMIC_LINE in ps, ps
    assert 'asint(' not in ATOMIC_LINE and ATOMIC_LINE in ps, 'umin must stay unsigned (no asint)'
    assert 'InterlockedMax' not in ps, ps
    assert 'register(t' not in ps, ps
    assert 'uav 0 tex2du table 0 0' in manifest and 'srv ' not in manifest, manifest
    assert manifest.startswith('x2w-manifest 2\n'), manifest
    # High-bit unsigned ordering: against 0x7fffffff, umin keeps the low half while smin takes OPERAND.
    assert OPERAND >= 0x80000000
    assert min(INITIAL, OPERAND) == EXPECTED
    hi_lo = 0x7fffffff
    unsigned_min = min(hi_lo, OPERAND)
    signed_min = min(struct.unpack('<i', struct.pack('<I', hi_lo))[0],
                     struct.unpack('<i', struct.pack('<I', OPERAND))[0])
    assert unsigned_min == hi_lo
    assert struct.unpack('<I', struct.pack('<i', signed_min))[0] == OPERAND
    print('umin: %s (high-bit ordering differs from smin)' % ATOMIC_LINE)
    print('manifest uav 0 tex2du table 0 0')

    for name, atomic in (('slc', patched(3, 0x02)), ('dlc', patched(0, 0x80)),
                         ('unorm', patched(1, 0x10))):
        assert D.decode_instruction(atomic, 0)[2].endswith(' ' + name), D.decode_instruction(atomic, 0)
        assert ATOMIC_LINE in translate(with_atomic(atomic)).hlsl_ps(), name
    print('slc, dlc and unorm accepted')

    # glc would return the old value into VDATA; not lowered here.
    glc = patched(1, 0x20)
    assert D.decode_instruction(glc, 0)[2].endswith(' glc'), D.decode_instruction(glc, 0)
    refused(with_atomic(glc), 'image_atomic_umin', 'modifier glc')
    print('refused glc (return form)')

    cases = (('d16', patched(7, 0x80)), ('a16', patched(7, 0x40)), ('tfe', patched(2, 0x01)),
             ('lwe', patched(2, 0x02)), ('r128', patched(1, 0x80)),
             ('SQ_RSRC_IMG_3D', bytes.fromhex('100154f000040400')),
             ('SQ_RSRC_IMG_2D_ARRAY', bytes.fromhex('280154f000040400')),
             ('SQ_RSRC_IMG_1D', bytes.fromhex('000154f000040400')))
    for name, atomic in cases:
        decoded = D.decode_instruction(atomic, 0)[2]
        message = refused(with_atomic(atomic), 'image_atomic_umin')
        print('refused %-20s %s' % (name + ':', message))
        assert decoded.startswith('image_atomic_umin'), decoded

    # Wider dmask sizes VDATA from the mask; refuse the mask itself.
    for mask, data in ((0x3, 'v[4:5]'), (0xf, 'v[4:7]')):
        atomic = patched(1, 0, mask)
        decoded = D.decode_instruction(atomic, 0)[2]
        assert decoded.startswith('image_atomic_umin %s, ' % data) and 'dmask:%#x' % mask in decoded, decoded
        message = refused(with_atomic(atomic), 'dmask %#x' % mask, 'red channel')
        print('refused %-20s %s' % ('dmask %#x:' % mask, message))

    # Other integer image atomics stay refused (handler missing → unsupported opcode name).
    for op, atomic in (('umax', bytes.fromhex('08015cf000040400')),
                       ('add', bytes.fromhex('080144f000040400')),
                       ('smin', bytes.fromhex('080150f000040400'))):
        decoded = D.decode_instruction(atomic, 0)[2]
        assert decoded.startswith('image_atomic_%s' % op), decoded
        refused(with_atomic(atomic), 'image_atomic_%s' % op)
        print('refused image_atomic_%s' % op)

    try:
        pipe.vs.op_image_atomic_umin(['v4', 'v[0:1]', 's[16:23]'], ['dmask:0x1', 'dim:SQ_RSRC_IMG_2D'])
    except G.Unsupported as error:
        assert 'outside a pixel stage' in str(error), error
    else:
        raise AssertionError('vertex-stage image_atomic_umin accepted')
    print('refused image_atomic_umin outside a pixel stage')

    for constant, words in ((1, EXEC_ON), (2, EXEC_OFF)):
        decoded = D.decode_instruction(cmpx_eq(constant), 0)[2]
        assert decoded == 'v_cmpx_eq_u32_e32 %d, v0' % constant, decoded
        ps = translate(words).hlsl_ps()
        assert '    exec = ((0x%08xu == v0) ? 1u : 0u);' % constant in ps, ps
        assert '    ' + GUARDED_LINE in ps, ps
        assert ps.count(ATOMIC_LINE) == 1, ps
    assert GUARDED_LINE not in translate(WORDS).hlsl_ps()
    print('exec after v_cmpx: %s' % GUARDED_LINE)

    lowering = G.Stage.op_image_atomic_umin
    del G.Stage.op_image_atomic_umin
    try:
        refused(WORDS, 'image_atomic_umin')
    finally:
        G.Stage.op_image_atomic_umin = lowering
    print('control: with op_image_atomic_umin removed the program is refused')


def check_binary(llvm_mc):
    built = assemble(llvm_mc, SOURCE)
    if built != WORDS:
        raise AssertionError('assembled %s, recorded %s' % (built.hex(), WORDS.hex()))
    print('fixture words match llvm-mc')


def mismatch_line(stderr):
    return next(line for line in stderr.splitlines() if 'texel_mismatch ' in line)


def check_draw(dxc, renderer, adapter):
    source = translate(WORDS).hlsl_ps()
    expected = '%08x' % EXPECTED
    untouched = '%08x' % INITIAL
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', directory)

        def draw(text, label, want=expected):
            ps = compile_hlsl(dxc, 'dxc', text, label, 'ps_6_0', directory)
            return subprocess.run(
                [renderer, adapter, '--startup-image-atomic-umin', str(vs), str(ps), '1,2', want],
                capture_output=True, text=True)

        run = draw(source, 'umin')
        sys.stdout.write(run.stdout)
        if run.returncode == 77:
            sys.exit(77)
        if run.returncode:
            raise AssertionError('umin draw failed\n%s\n%s' % (run.stdout, run.stderr))
        print('draw: texel (1, 2) = %s, all other texels untouched' % expected)

        mutant = source.replace('InterlockedMin(', 'InterlockedMax(')
        assert mutant != source
        run = draw(mutant, 'max_for_min', want=expected)
        needle = 'texel_mismatch x=1 y=2 expected=%s actual=%s' % (expected, untouched)
        if run.returncode == 0 or needle not in run.stderr:
            raise AssertionError('control Max-for-Min was not detected\n%s\n%s' % (run.stdout, run.stderr))
        print('control Max-for-Min: fails as expected (%s)' % mismatch_line(run.stderr))

        on, off = translate(EXEC_ON).hlsl_ps(), translate(EXEC_OFF).hlsl_ps()
        for label, text, want in (('exec on', on, expected), ('exec off', off, untouched)):
            run = draw(text, label.replace(' ', '_'), want)
            if run.returncode:
                raise AssertionError('%s draw failed\n%s\n%s' % (label, run.stdout, run.stderr))
            print('draw %s: texel (1, 2) = %s, all other texels untouched' % (label, want))
        unguarded = off.replace(GUARDED_LINE, ATOMIC_LINE)
        assert unguarded != off
        run = draw(unguarded, 'exec_off_unguarded', untouched)
        needle = 'texel_mismatch x=1 y=2 expected=%s actual=%s' % (untouched, expected)
        if run.returncode == 0 or needle not in run.stderr:
            raise AssertionError('control exec guard removed was not detected\n%s\n%s' % (run.stdout, run.stderr))
        print('control exec guard removed: fails as expected (%s)' % mismatch_line(run.stderr))


def main():
    controls()
    if len(sys.argv) == 1:
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
