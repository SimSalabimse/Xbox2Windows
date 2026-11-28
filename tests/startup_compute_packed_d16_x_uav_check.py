"""Publish a source-built one-component D16 buffer store and check the written halfwords.

No game data. The program is a gfx1013 `buffer_store_format` with the D16 bit and one
component (the direct compiler's `packed_d16_x_uav`). `compile_dxil --requests` must
publish `uav 0 bufd16x table 1 32 bytes 2`. The stand-in then dispatches that translation.
The reference halfwords are the low half of `v_xor_b32` of the literal `0xc0003c00` with
the lane id, which for lanes 0..12 is `0x3c00 + lane`. An R32_UINT view (two halfwords
per element), a stride other than 2, and a descriptor with swizzle flags set must refuse
by name. Before publication the dispatch stays `no translation`.

usage:
  startup_compute_packed_d16_x_uav_check.py
      reference only (no compiler, no device)
  startup_compute_packed_d16_x_uav_check.py compile <compile_dxil> <dxc-bin-dir>
  startup_compute_packed_d16_x_uav_check.py <compile_dxil> <startup_stream_consumer>
      <offscreen_check> <dxc-bin-dir> <warp|hardware>
"""

from pathlib import Path
import struct
import subprocess
import sys

SENTINEL = 0xc5c5c5c5
WORDS = 320
OFFSET = 8
RECORDS = 13
STRIDE = 2
# gfx10 unified BUF_FMT_16_FLOAT is 13; BUF_FMT_32_UINT is 20. RESOURCE_LEVEL is bit 24
# and 0xfac is identity DST_SEL. A 32-bit element holds two halfwords, so it is not this store.
FLOAT16 = 13
UINT32 = 20
WORD3_BASE = 0x01000fac
INCOMPATIBLE = 'formatted D16 write has an incompatible typed buffer UAV'
MODE = 'dispatch-records32-output-offset'
MANIFEST_LINE = f'uav 0 bufd16x table 1 32 bytes {STRIDE}'


def vop2(op, dst, src0, src1):
    return op << 25 | dst << 17 | src1 << 9 | src0


def scalar(op, dst, a, b):
    return 0x80000000 | op << 23 | dst << 16 | b << 8 | a


def sop1(op, dst, src):
    return 0xbe800000 | dst << 16 | op << 8 | src


def program_words():
    # Same index and payload shape as the one-channel half-buffer store. Opcode 0x84 is
    # buffer_store_format_d16_x: idxen, v1, v8, s[32:35], offset 0.
    store = 0xe0002000 | (0x84 << 18)
    return [
        sop1(3, 20, 6), sop1(3, 21, 128),
        0xf408080a, 0xfa000020,
        0xbf8cc07f,
        scalar(0x26, 8, 8, 131),
        scalar(0x26, 9, 9, 134),
        scalar(0, 11, 7, 8), scalar(0, 11, 11, 9),
        scalar(0x1e, 11, 11, 133),
        vop2(0x25, 8, 11, 0),
        vop2(0x1d, 1, 255, 0), 0xc0003c00,
        0x7e0402ff, 0xbc003800,
        store, 0x80080108,
        0xbf810000,
    ]


def program():
    words = program_words()
    return struct.pack('<%dI' % len(words), *words)


def halfword(lane):
    """Low halfword of v1. Lanes below 16 have no bits in common with 0x3c00, so XOR adds."""
    return (0xc0003c00 ^ lane) & 0xffff


def word3(fmt):
    return WORD3_BASE | (fmt << 12)


def expected():
    data = bytearray(struct.pack('<%dI' % WORDS, *([SENTINEL] * WORDS)))
    for lane in range(RECORDS):
        struct.pack_into('<H', data, OFFSET + lane * STRIDE, halfword(lane))
    return list(struct.unpack('<%dI' % WORDS, data))


def check_words(got, want, label):
    first = next(((i, a, b) for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    if first is not None:
        word, actual, expected_word = first
        raise AssertionError(f'{label}: word {word} expected {expected_word:08x}, read {actual:08x}')


def check_reference():
    words = program_words()
    store, store_b = words[-3], words[-2]
    assert store == 0xe2102000, hex(store)
    assert (store >> 18) & 0xff == 0x84, 'opcode is buffer_store_format_d16_x'
    assert store & 0x3ffff == 0x2000, 'idxen, offset 0'
    assert (store_b >> 8) & 0xff == 1 and (store_b & 0xff) == 8, 'data v1, index v8'
    assert ((store_b >> 16) & 31) * 4 == 32, 'descriptor s[32:35]'
    assert vop2(0x1d, 1, 255, 0) in words and 0xc0003c00 in words
    for lane in range(RECORDS):
        assert halfword(lane) == 0x3c00 + lane, lane
    image = expected()
    raw = struct.pack('<%dI' % WORDS, *image)
    for lane in range(RECORDS):
        got, = struct.unpack_from('<H', raw, OFFSET + lane * STRIDE)
        assert got == 0x3c00 + lane, (lane, hex(got))
    assert raw[:OFFSET] == bytes([0xc5]) * OFFSET
    tail = OFFSET + RECORDS * STRIDE
    assert raw[tail:] == bytes([0xc5]) * (len(raw) - tail)
    accepted = (STRIDE, word3(FLOAT16))
    assert accepted != (4, word3(UINT32))
    assert accepted != (STRIDE + 2, word3(FLOAT16))
    assert accepted != (STRIDE | 0x8000, word3(FLOAT16))
    assert (word3(UINT32) >> 12) & 127 == UINT32
    print(f'reference: {RECORDS} halfwords at byte {OFFSET}, guards intact, three negative descriptors differ')


def publish(compile_dxil, sdk, cache, key):
    result = subprocess.run([compile_dxil, sdk, cache.parent / 'native', 'gfx1013-direct-cs@0',
                             '--requests', cache, cache], capture_output=True, text=True, timeout=30)
    manifest = cache / f'{key}.txt'
    assert result.returncode == 0 and manifest.is_file(), (result.returncode, result.stdout, result.stderr)
    text = manifest.read_text()
    assert MANIFEST_LINE in text.splitlines(), text
    assert not (cache / f'{key}.cs.hlsl').exists(), 'expected native publication'


def check_compiler(compile_dxil, sdk):
    from tempfile import TemporaryDirectory

    from startup_compute_translate_check import state_a, write_request
    from startup_gfx2hlsl_compute_check import TGID_X, TGID_Y, TGID_Z, USER7, root_blob

    state = state_a((32, 1, 1), USER7 | TGID_X | TGID_Y | TGID_Z)
    with TemporaryDirectory() as directory:
        cache = Path(directory)
        key = write_request(cache, program(), state, root=root_blob(table_range=True))
        publish(compile_dxil, sdk, cache, f'{key:016x}')
    print('compile: published', MANIFEST_LINE)


def check(compile_dxil, consumer, sdk, work, adapter):
    from startup_compute_dispatch_check import dispatch, lines, prepare_case
    from startup_compute_translate_check import state_a
    from startup_gfx2hlsl_compute_check import TGID_X, TGID_Y, TGID_Z, USER7

    cache, env = prepare_case(work, adapter)
    env.pop('X2W_NATIVE_SHADERS', None)
    env.pop('X2W_NATIVE_SHADERS_ONLY', None)
    state = state_a((32, 1, 1), USER7 | TGID_X | TGID_Y | TGID_Z)
    (work / 'program').write_bytes(program())
    (work / 'state').write_bytes(state)
    fields = work / 'fields.bin'
    env['X2W_TEST_DISPATCH_FIELDS'] = str(fields)
    accepted_word3 = word3(FLOAT16)
    untouched = [SENTINEL] * WORDS

    def run(view_stride=STRIDE, records=RECORDS, descriptor_word=accepted_word3, *, probe=False):
        fields.write_bytes(struct.pack('<6I', 4, WORDS, 0, view_stride, records, descriptor_word))
        return dispatch(work, consumer, env, MODE, probe=probe)

    got, result = run(probe=True)
    check_words(got, untouched, 'before translation')
    assert lines(result, 'Dispatch(3, 2, 2) refused: no translation'), result.stderr[-3000:]
    [request] = cache.glob('req_*.c67')
    key = request.stem.removeprefix('req_')
    publish(compile_dxil, sdk, cache, key)
    got, result = run()
    check_words(got, expected(), 'packed D16 x')
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused:'), result.stderr[-3000:]
    assert not [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
    layers = lines(result, 'x2w-d3d12 debug layer: ')
    layer = 'enabled' if layers and '0x00000000' in layers[0] else 'unavailable'
    print(f'{RECORDS} halfwords at byte {OFFSET} and all guards match on {adapter} (debug layer {layer})')

    for label, options in (
        ('R32_UINT view', {'view_stride': 4, 'descriptor_word': word3(UINT32)}),
        ('wrong stride', {'view_stride': STRIDE + 2}),
        ('flags set', {'view_stride': STRIDE | 0x8000}),
    ):
        got, result = run(**options)
        check_words(got, untouched, label)
        assert len(lines(result, 'Dispatch(3, 2, 2) refused: ' + INCOMPATIBLE)) == 1, (label, result.stderr[-3000:])
        print(f'{label}: refused ({INCOMPATIBLE}), output untouched')


def main(compile_dxil, consumer, renderer, sdk, adapter):
    from tempfile import TemporaryDirectory

    from native_gather_lod_check import render

    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, wave.stdout + wave.stderr
    with TemporaryDirectory() as directory:
        check(compile_dxil, consumer, sdk, Path(directory), adapter)


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] == 'reference':
        check_reference()
    elif sys.argv[1] == 'compile':
        check_compiler(*sys.argv[2:])
    elif len(sys.argv) == 6 and sys.argv[5] in ('warp', 'hardware'):
        main(*sys.argv[1:])
    else:
        sys.exit(__doc__)
