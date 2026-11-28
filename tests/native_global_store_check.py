"""DWORD GLOBAL stores through a root UAV in the direct C++ compute path.

  native_global_store_check.py encodings
  native_global_store_check.py <compile_dxil> <native_store_replay> <sdk bin dir>

ADDR = SADDR + zext(VADDR) + sext(OFFSET[11:0]) (RDNA 2 ISA, FLAT/GLOBAL). SADDR is the unadvanced root UAV.
The VGPR is a 32-bit byte offset. The direct frontend lowers global_store_dword, dwordx2, dwordx3 and dwordx4
on that form, and a global_load_dword from the same UAV. A 64-bit VGPR address (saddr off), flat, scratch,
LDS, atomics, sub-DWORD ops, a non-UAV root, an unaligned immediate and a data register that runs past v255
stay refused.

Words are built from the FLAT bit layout in tools/disasm_rdna2.py. `encodings` checks those words with the
disassembler and checks that a swapped x2 reference is not the one the readback uses. The dispatch needs
shader model 6.6 with wave32 (native_store_replay returns 77 otherwise).

Two groups of 32 lanes. Lane id N = group * 32 + lane. The root UAV is initially 0x69696969. Each lane's
48-byte record at byte 48*N is:

  +0  dword          tag
  +4  dword          tag+1
  +8  dwordx2        tag+2, tag+3
  +16 dwordx3        tag+4, tag+5, tag+6
  +28 untouched      0x69696969
  +32 dwordx4        tag, tag+1, tag+2, tag+3

tag = 0xA1000000 | N. Other regions, still inside the view:

  byte 3584+4*N     the address itself (data register aliases the address)
  byte 4092+4*N     tag+1 with vaddr 4096+4*N and offset -4
  byte 5120+4*N     tag on even lanes; odd lanes stay 0x69696969 (EXEC low bits 0x55555555)
  byte 5888         0, from a load of vaddr 0xfffffff4 offset 16 (the wrap guard)
  byte 6400+4*N     the tag loaded back from byte 48*N after the stores and s_waitcnt

An x4 store of 0xEEEEEEEE at vaddr 0xffffffe8 offset 16 must not land. Its first DWORD is at 0xfffffff8, still
inside the 32-bit range, and its last DWORD starts at 0x100000004. The upper guard is VADDR <= 2^32-1-(offset+12),
which is 4294967267. Narrowing that to `last_offset = offset` accepts this VADDR and drops 4294967267 from the
DXIL. The readback includes the 8-dword guard native_store_replay places in front of the view.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools'))

import disasm_rdna2 as D  # noqa: E402
from native_compute_clear_check import write_request
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1, vmove


SENTINEL = 0x69696969
TAG = 0xA1000000
POISON = 0xEEEEEEEE
GUARD = 8
VIEW_DWORDS = 2048
READBACK = (VIEW_DWORDS + 2 * GUARD)
RECORD = 48
ALIAS = 3584
NEG_BASE = 4096
SPARSE = 5120
LOAD_SLOT = 5888
RELOAD = 6400
X4_WRAP = 0xffffffe8
# s[4:5] root UAV, s6 = TGID_X. Wave32, one wave per group.
SA = struct.pack('<7I', 0, 0, 0, 32 | 1 << 16, 1 | (1 << 20), 0, 0x80 | 6 << 1)
# s[4:5] and s[6:7] root SRVs, s[8:9] root UAV. No group id.
PAIR_SA = struct.pack('<7I', 0, 0, 0, 32 | (1 << 16), 1 | (1 << 20), 0, 10 << 1)


def encode_global(op, vdst, addr, saddr, data=0, offset=0, glc=0, slc=0, dlc=0, lds=0, seg=2):
    w0 = (0xdc000000 | ((op & 0x7f) << 18) | ((slc & 1) << 17) | ((glc & 1) << 16)
          | ((seg & 3) << 14) | ((lds & 1) << 13) | ((dlc & 1) << 12) | (offset & 0xfff))
    w1 = ((vdst & 0xff) << 24) | ((saddr & 0x7f) << 16) | ((data & 0xff) << 8) | (addr & 0xff)
    return [w0, w1]


def program():
    code = [
        scalar(0x1e, 9, 6, 128 + 5),          # s_lshl_b32 s9, s6, 5
        vop2(0x25, 9, 9, 0),                  # v_add_nc_u32 v9, s9, v0
        vop2(0x1a, 1, 128 + 5, 9),            # v_lshlrev_b32 v1, 5, v9
        vop2(0x1a, 18, 128 + 4, 9),           # v_lshlrev_b32 v18, 4, v9
        vop2(0x25, 1, 256 + 18, 1),           # v_add_nc_u32 v1, v18, v1
        vop2(0x1c, 10, 9, 0),                 # v_or_b32 v10, s9, v0
        vop2(0x1c, 10, 255, 10), TAG,         # v_or_b32 v10, tag, v10
    ]
    for n in range(1, 8):
        code.append(vop2(0x25, 10 + n, 128 + n, 10))  # v_add_nc_u32 v(10+n), n, v10
    code += [
        vmove(2, 255), 0xfffffff4,            # v_mov_b32 v2, 0xfffffff4
        vmove(3, 255), POISON,
        vmove(5, 255), LOAD_SLOT,
        vmove(24, 255), X4_WRAP,
        vmove(20, 255), POISON,
        vmove(21, 255), POISON,
        vmove(22, 255), POISON,
        vmove(23, 255), POISON,
        vop2(0x1a, 19, 128 + 2, 9),           # v_lshlrev_b32 v19, 2, v9
        vop2(0x25, 6, 255, 19), NEG_BASE,     # v6 = 4096 + 4*N
        vop2(0x25, 7, 255, 19), ALIAS,
        vop2(0x25, 8, 255, 19), SPARSE,
        vop2(0x25, 25, 255, 19), RELOAD,
    ]
    code += encode_global(0x0c, 4, 2, 4, offset=16)  # global_load_dword v4, v2, s[4:5] offset:16
    code.append(0xbf8c0070)  # s_waitcnt vmcnt(0) lgkmcnt(0)
    code += encode_global(0x1c, 0, 5, 4, data=4)  # global_store_dword v5, v4, s[4:5]
    code += encode_global(0x1c, 0, 1, 4, data=10, glc=1, slc=1, dlc=1)
    code += encode_global(0x1c, 0, 1, 4, data=11, offset=4)
    code += encode_global(0x1d, 0, 1, 4, data=12, offset=8)
    code += encode_global(0x1f, 0, 1, 4, data=14, offset=16)
    code += encode_global(0x1e, 0, 1, 4, data=10, offset=32)
    code += encode_global(0x1c, 0, 6, 4, data=11, offset=-4)
    code += encode_global(0x1c, 0, 7, 4, data=7)  # address register is the data
    code += encode_global(0x1c, 0, 2, 4, data=3, offset=16)  # poison, must not store
    code += encode_global(0x1e, 0, 24, 4, data=20, offset=16)  # x4 whose last dword wraps
    code += [sop1(3, 126, 255), 0x55555555]
    code += encode_global(0x1c, 0, 8, 4, data=10)
    code += [sop1(3, 126, 193)]  # s_mov_b32 exec_lo, -1
    code.append(0xbf8c0070)  # s_waitcnt vmcnt(0) lgkmcnt(0)
    code += encode_global(0x0c, 4, 1, 4)  # global_load_dword v4, v1, s[4:5]
    code += encode_global(0x1c, 0, 25, 4, data=4)  # store the loaded tag
    code.append(0xbf810000)
    return code


def extent_program():
    return encode_global(0x1e, 0, 1, 4, data=10, offset=2044) + [0xbf810000]


def register_program():
    # s[4:5] and s[6:7] are root SRVs, s[8:9] is a root UAV. Two SRVs make srv++ observable.
    return (encode_global(0x0c, 1, 0, 4) + encode_global(0x0c, 2, 0, 6)
            + encode_global(0x1c, 0, 0, 8, data=1) + [0xbf810000])


def reference():
    expected = [SENTINEL] * READBACK
    writes = {}

    def put(byte, value):
        if byte % 4 or byte < 0 or byte // 4 >= VIEW_DWORDS:
            raise AssertionError(hex(byte))
        index = GUARD + byte // 4
        if index in writes and writes[index] != value:
            raise AssertionError((byte, hex(writes[index]), hex(value)))
        writes[index] = value
        expected[index] = value

    for lane in range(64):
        tag = TAG | lane
        base = lane * RECORD
        values = [tag + n for n in range(8)]
        for dword, value in enumerate((values[0], values[1], values[2], values[3], values[4], values[5], values[6])):
            put(base + 4 * dword, value)
        for dword, value in enumerate(values[:4]):
            put(base + 32 + 4 * dword, value)
        put(NEG_BASE + 4 * lane - 4, values[1])
        put(ALIAS + 4 * lane, ALIAS + 4 * lane)
        if lane % 2 == 0:
            put(SPARSE + 4 * lane, tag)
        put(RELOAD + 4 * lane, tag)
    put(LOAD_SLOT, 0)
    swapped = expected[:]
    # The x2 pair is dwords 2 and 3 of the record. Swapping them must change the buffer.
    for lane in range(64):
        index = GUARD + (lane * RECORD) // 4
        swapped[index + 2], swapped[index + 3] = swapped[index + 3], swapped[index + 2]
    if swapped == expected:
        raise AssertionError('swapped x2 reference matches')
    if POISON in expected:
        raise AssertionError('poison is in the reference')
    return expected


def encodings():
    text = []
    blob = struct.pack('<' + 'I' * len(program()), *program())
    at = 0
    while at < len(blob):
        size, name, asm, _ = D.decode_instruction(blob, at)
        text.append(asm)
        at += size
    joined = '\n'.join(text)
    for needle in (
        'global_load_dword v4, v2, s[4:5] offset:16',
        'global_store_dword v5, v4, s[4:5]',
        'global_store_dword v1, v10, s[4:5] glc slc dlc',
        'global_store_dword v1, v11, s[4:5] offset:4',
        'global_store_dwordx2 v1, v[12:13], s[4:5] offset:8',
        'global_store_dwordx3 v1, v[14:16], s[4:5] offset:16',
        'global_store_dwordx4 v1, v[10:13], s[4:5] offset:32',
        'global_store_dword v6, v11, s[4:5] offset:-4',
        'global_store_dword v7, v7, s[4:5]',
        'global_store_dword v2, v3, s[4:5] offset:16',
        'global_store_dwordx4 v24, v[20:23], s[4:5] offset:16',
        'global_store_dword v8, v10, s[4:5]',
        'global_load_dword v4, v1, s[4:5]',
        'global_store_dword v25, v4, s[4:5]',
        's_waitcnt vmcnt(0) lgkmcnt(0)',
        's_endpgm',
    ):
        if needle not in joined:
            raise AssertionError(needle + '\n' + joined)
    negatives = {
        'global_store_dword v[1:2], v10, off': encode_global(0x1c, 0, 1, 0x7d, data=10),
        'global_store_dword v1, v10, s[4:5] offset:2': encode_global(0x1c, 0, 1, 4, data=10, offset=2),
        'global_store_dwordx4 v1, v[253:256], s[4:5]': encode_global(0x1e, 0, 1, 4, data=253),
        'global_store_dwordx3 v1, v[254:256], s[4:5]': encode_global(0x1f, 0, 1, 4, data=254),
        'global_store_dword v1, v10, s5': encode_global(0x1c, 0, 1, 5, data=10),
        'global_store_dword v1, v10, exec_hi': encode_global(0x1c, 0, 1, 127, data=10),
        'global_store_byte v1, v10, s[4:5]': encode_global(0x18, 0, 1, 4, data=10),
        'global_atomic_add v1, v10, s[4:5]': encode_global(0x32, 0, 1, 4, data=10),
        'global_load_dword v1, s[4:5] lds': encode_global(0x0c, 4, 1, 4, lds=1),
        'flat_store_dword v[1:2], v10': encode_global(0x1c, 0, 1, 4, data=10, seg=0),
        'scratch_store_dword off, v10, s4': encode_global(0x1c, 0, 1, 4, data=10, seg=1),
        'global_load_dword v1, v0, s[4:5]': encode_global(0x0c, 1, 0, 4),
        'global_load_dword v2, v0, s[6:7]': encode_global(0x0c, 2, 0, 6),
        'global_store_dword v0, v1, s[8:9]': encode_global(0x1c, 0, 0, 8, data=1),
    }
    for asm, words in negatives.items():
        blob = struct.pack('<2I', *words)
        got = D.decode_instruction(blob, 0)[2]
        if got != asm:
            raise AssertionError((asm, got))
    reference()
    print('encodings: dword, x2, x3 and x4 stores, a UAV load, alias, negative offset and the wrap poison')


def main(compiler, replay, sdk):
    encodings()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        def root_sig(text):
            source, binary = work / 'root.hlsl', work / 'root.bin'
            source.write_text('#define SIG "%s"' % text)
            subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', binary, source], check=True)
            return binary.read_bytes()

        uav = root_sig('UAV(u0)')
        srv = root_sig('SRV(t0)')
        both = root_sig('SRV(t0), SRV(t1), UAV(u0)')

        def compile_words(code, root, sa=SA):
            marker = write_request(work, code, root, sa)
            return marker, subprocess.run([compiler, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker,
                                           marker.with_suffix('.cso')], capture_output=True, text=True)

        negatives = [
            (encode_global(0x1c, 0, 1, 0x7d, data=10), 'VGPR address pair is not a root buffer offset'),
            (encode_global(0x1c, 0, 1, 4, data=10, offset=2), 'unaligned immediate offset'),
            (encode_global(0x1e, 0, 1, 4, data=253), 'global store data register range'),
            (encode_global(0x1f, 0, 1, 4, data=254), 'global store data register range'),
            (encode_global(0x1c, 0, 1, 5, data=10), 'aligned SGPR pair'),
            (encode_global(0x1c, 0, 1, 127, data=10), 'scalar address is not an SGPR pair'),
            (encode_global(0x18, 0, 1, 4, data=10), 'only DWORD loads and stores with a scalar address pair'),
            (encode_global(0x32, 0, 1, 4, data=10), 'only DWORD loads and stores with a scalar address pair'),
            (encode_global(0x0c, 4, 1, 4, lds=1), 'does not lower an LDS result'),
            (encode_global(0x1c, 0, 1, 4, data=10, seg=0), 'flat or scratch memory is not a global root access'),
            (encode_global(0x1c, 0, 1, 4, data=10, seg=1), 'flat or scratch memory is not a global root access'),
            ([sop1(3, 5, 128)] + encode_global(0x1c, 0, 1, 4, data=10), 'advanced or mismatched root address'),
            ([scalar(0, 4, 4, 128 + 16)] + encode_global(0x1c, 0, 1, 4, data=10), 'unbound numeric SGPR s4'),
        ]
        body = program()
        for code, message in negatives:
            _, run = compile_words(code + [0xbf810000], uav)
            if run.returncode == 0 or message not in (run.stdout + run.stderr):
                raise AssertionError((message, run.returncode, run.stdout, run.stderr))
        _, srv_run = compile_words(body, srv)
        if srv_run.returncode == 0 or 'global store base is not a root UAV' not in (srv_run.stdout + srv_run.stderr):
            raise AssertionError((srv_run.returncode, srv_run.stdout, srv_run.stderr))
        print('refused: VGPR address pair, unaligned offset, data past v255, an odd base, saddr 127, '
              'a byte store, an atomic, LDS, flat, scratch, a replaced high half, an advanced pointer, an SRV root')

        _, extent = compile_words(extent_program(), uav)
        if extent.returncode != 0 or 'u0 root=0 bytes=0 min_size=2060' not in extent.stdout:
            raise AssertionError((extent.stdout, extent.stderr))
        marker, run = compile_words(body, uav)
        if run.returncode != 0 or 'u0 root=0 bytes=0 min_size=48' not in run.stdout:
            raise AssertionError((run.stdout, run.stderr))
        # x4 at offset 16 guards the last dword, 12 bytes past the first. `last_offset = offset`
        # would emit 4294967279, the same bound as the single-dword poison store.
        dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', marker.with_suffix('.cso')],
                              capture_output=True, text=True, check=True).stdout
        if '4294967267' not in dump:
            raise AssertionError('x4 wrap guard does not cover the last dword')
        _, regs = compile_words(register_program(), both, PAIR_SA)
        for line in ('t0 root=0 bytes=0 min_size=4', 't1 root=1 bytes=0 min_size=4', 'u0 root=2 bytes=0 min_size=4'):
            if regs.returncode != 0 or line not in regs.stdout:
                raise AssertionError((line, regs.stdout, regs.stderr))
        print('module: one root UAV, min_size 48, x4 wrap bound 4294967267, registers t0 t1 u0')

        dest = work / 'readback.bin'
        result = subprocess.run([replay, sdk, marker, work / 'cache', '0', dest, '--direct'],
                                capture_output=True, text=True)
        if result.returncode == 77:
            print(result.stdout + result.stderr)
            sys.exit(77)
        if result.returncode != 0:
            raise AssertionError((result.stdout, result.stderr))
        actual = list(struct.unpack('<%dI' % READBACK, dest.read_bytes()))
        expected = reference()
        diff = [(i, hex(a), hex(b)) for i, (a, b) in enumerate(zip(actual, expected, strict=True)) if a != b]
        if diff:
            raise AssertionError(diff[:12])
        if POISON in actual:
            raise AssertionError('poison store became visible')
        print('global_store_dword/x2/x3/x4: record, alias, negative offset, sparse EXEC, UAV load wrap guard; '
              'poison store absent')


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'encodings':
        encodings()
    else:
        main(*sys.argv[1:])
