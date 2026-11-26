#!/usr/bin/env python3
"""Probe of v_cvt_pk_u8_f32 (VOP3A opcode 350) on RDNA 2/3 hardware.

The RDNA ISA guides give  D = (S2 & ~(0xff << 8*S1[1:0])) | ((flt32_to_uint8(S0) & 0xff) << 8*S1[1:0])
but leave flt32_to_uint8 undefined (rounding, clamping, NaN).  This tool builds small HSA kernels that run
the instruction on a table of inputs, launches them through HIP (ctypes only) and compares the readback with
independent candidate models.

Usage:
  python3 tools/rdna_cvt_pk_u8.py            run on the first HIP device (every device is listed); writes
                                             out/rdna_cvt_pk_u8.tsv and compares it with the stored rows
  ... --device 1                             HIP device 1 (e.g. the discrete card when an iGPU is device 0)
  ... --arch gfx1032                         override the target (default: from the device)
  ... --asm-only --arch gfx1030 --keep DIR   build only (no GPU needed)
  ... --build asm                            hand-written assembly (llvm-mc + ld.lld) instead of clang
  ... --self-test                            check the models and the merge logic (no GPU needed)
  ... --out FILE / --compare FILE / --json FILE   result file, reference to compare with, optional verbose JSON
Python 3 standard library only; needs clang (ROCm / HIP SDK ship it); --build asm needs llvm-mc and ld.lld.

Stored result: tests/data/rdna_cvt_pk_u8_measured.tsv (see "Measured on RDNA 2" in docs/experiments.md).
The result file is compact: header lines starting with '#', then one tab separated row per (input, select):
label, input word, select word, then the output word of the default, rtz and flush variants.
A rerun is comparable with the stored file when the rows (lines not starting with '#') are equal; the
probe prints that verdict itself.
"""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

SENTINEL = 0xA5C3E17B
SELECTS = [0, 1, 2, 3, 0xFFFFFFFD, 0xFFFFFFFC, 0x80000002, 0x00000103]
BLOCK = 64
DEFAULT_REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'data',
                                 'rdna_cvt_pk_u8_measured.tsv')

# name -> (description, mode lines executed before the conversion)
VARIANTS = {
    'default': ('MODE left at the launch default', []),
    # s_round_mode simm4: [1:0] f32 rounding, [3:2] f64/f16; 3 = toward zero.
    'rtz': ('s_round_mode 0xf (round toward zero)', ['s_round_mode 0xf']),
    # s_denorm_mode simm4: [0] f32 input denormals, [1] f32 output denormals, [3:2] f64/f16; 0 = flush.
    'flush': ('s_denorm_mode 0 (denormals flushed)', ['s_denorm_mode 0']),
}


# ---------------------------------------------------------------- inputs

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def val(b):
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]


def input_table():
    """Return a list of (label, f32 bit pattern), without duplicates."""
    rows = []

    def add(label, b):
        rows.append((label, b & 0xFFFFFFFF))

    add('+0', 0x00000000)
    add('-0', 0x80000000)
    for i in list(range(1, 17)) + [31, 32, 63, 64, 100, 127, 128, 129, 200, 253, 254, 255]:
        add('%d' % i, bits(float(i)))
    for t in (0.5, 1.5, 2.5, 3.5, 127.5, 254.5, 255.5):
        b = bits(t)
        add('%g' % t, b)
        add('%g-ulp' % t, b - 1)
        add('%g+ulp' % t, b + 1)
    for t in (0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 100.25, 100.75, 254.25, 254.75, 0.1, 0.9, 0.4999999, 0.5000001):
        add('%.7g' % t, bits(f32(t)))
    for t in (-0.1, -0.4, -0.5, -0.6, -1.0, -1.5, -2.0, -255.0, -256.0):
        add('%g' % t, bits(t))
    add('-0.5-ulp', bits(-0.5) + 1)
    add('-0.5+ulp', bits(-0.5) - 1)
    add('255.4', bits(f32(255.4)))
    add('255.5+ulp', bits(255.5) + 1)
    for t in (255.9, 256.0, 256.5, 257.0, 511.0, 512.0, 1000.0, 65535.0, 65536.0, 16777216.0, 4294967296.0, 1e10):
        add('%g' % t, bits(f32(t)))
    add('+inf', 0x7F800000)
    add('-inf', 0xFF800000)
    for lab, b in (('qNaN', 0x7FC00000), ('-qNaN', 0xFFC00000), ('sNaN', 0x7FA00000), ('-sNaN', 0xFFA00000),
                   ('qNaN+payload', 0x7FC00001), ('sNaN+1', 0x7F800001), ('NaN-all-ones', 0x7FFFFFFF),
                   ('-NaN-all-ones', 0xFFFFFFFF), ('qNaN-low-payload-ff', 0x7FC000FF)):
        add(lab, b)
    add('min-subnormal', 0x00000001)
    add('-min-subnormal', 0x80000001)
    add('max-subnormal', 0x007FFFFF)
    add('min-normal', 0x00800000)
    add('max-float', 0x7F7FFFFF)
    add('-max-float', 0xFF7FFFFF)
    # Typical x * 255.0 computed in binary32 (the product of two f32 is exact in binary64, so one rounding).
    for den in (255, 256, 64, 100):
        for i in range(den + 1):
            x = f32(i / den)
            p = f32(x * 255.0)
            add('(%d/%d)*255' % (i, den), bits(p))
    seen, out = set(), []
    for lab, b in rows:
        if b not in seen:
            seen.add(b)
            out.append((lab, b))
    return out


def build_rows():
    """Each input under every select. Returns list of (label, in_bits, sel)."""
    return [(lab, b, s) for lab, b in input_table() for s in SELECTS]


# ---------------------------------------------------------------- models

def _sat(n):
    return 0 if n < 0 else 255 if n > 255 else n


def _is_nan(x):
    return x != x


def model_rne_sat(x):
    if _is_nan(x):
        return 0
    if math.isinf(x):
        return 255 if x > 0 else 0
    return _sat(int(round(x)))  # Python round() is half-to-even on floats


def model_trunc_sat(x):
    if _is_nan(x):
        return 0
    if math.isinf(x):
        return 255 if x > 0 else 0
    return _sat(int(x))


def model_half_away_sat(x):
    if _is_nan(x):
        return 0
    if math.isinf(x):
        return 255 if x > 0 else 0
    f = math.floor(abs(x))
    if abs(x) - f >= 0.5:
        f += 1
    return _sat(int(math.copysign(f, x)))


def model_trunc_wrap(x):
    # No saturation: low 8 bits of the truncated integer; NaN and infinities give 0 (arbitrary choice).
    if _is_nan(x) or math.isinf(x):
        return 0
    return int(x) & 0xFF


MODELS = {
    'rne_sat_nan0': model_rne_sat,
    'trunc_sat_nan0': model_trunc_sat,
    'half_away_sat_nan0': model_half_away_sat,
    'trunc_wrap8': model_trunc_wrap,
}


def merge(byte, sel, s2=SENTINEL):
    sh = 8 * (sel & 3)
    return ((s2 & ~(0xFF << sh)) | ((byte & 0xFF) << sh)) & 0xFFFFFFFF


def model_word(model, in_bits, sel):
    return merge(MODELS[model](val(in_bits)), sel)


def extract_byte(word, sel):
    return (word >> (8 * (sel & 3))) & 0xFF


def merge_ok(word, sel):
    """The bytes other than the selected one must equal S2's."""
    mask = ~(0xFF << (8 * (sel & 3))) & 0xFFFFFFFF
    return (word & mask) == (SENTINEL & mask)


# ---------------------------------------------------------------- assembly

def gen_asm(arch, variant, cov=5):
    name = 'cvt_pk_u8_' + variant
    mode = '\n'.join('  ' + l for l in VARIANTS[variant][1])
    if mode:
        mode += '\n  s_nop 0'
    meta_ver = '[1, 2]' if cov >= 5 else '[1, 1]'
    return f""".amdgcn_target "amdgcn-amd-amdhsa--{arch}"
.amdhsa_code_object_version {cov}
.text
.globl {name}
.p2align 8
.type {name},@function
{name}:
  ; kernarg: in (u64 pairs of {{f32 word, select word}}), out (u64), n (u32)
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  s_load_dwordx2 s[6:7], s[0:1], 0x8
  s_load_dword s8, s[0:1], 0x10
  s_waitcnt lgkmcnt(0)
  s_lshl_b32 s9, s2, 6
  v_add_nc_u32 v1, s9, v0
  v_cmp_gt_u32 vcc_lo, s8, v1
  s_and_saveexec_b32 s10, vcc_lo
  s_cbranch_execz .Ldone_{variant}
  v_lshlrev_b32 v2, 3, v1
  global_load_dwordx2 v[3:4], v2, s[4:5]
  s_waitcnt vmcnt(0)
{mode}
  v_mov_b32 v5, 0x{SENTINEL:08x}
  v_cvt_pk_u8_f32 v6, v3, v4, v5
  v_lshlrev_b32 v7, 2, v1
  global_store_dword v7, v6, s[6:7]
.Ldone_{variant}:
  s_endpgm
.Lend_{variant}:
.size {name}, .Lend_{variant}-{name}

.rodata
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 16
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: {meta_ver}
amdhsa.target: amdgcn-amd-amdhsa--{arch}
amdhsa.kernels:
  - .name: {name}
    .symbol: {name}.kd
    .language: Assembler
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .sgpr_count: 16
    .vgpr_count: 8
    .max_flat_workgroup_size: {BLOCK}
    .wavefront_size: 32
    .args:
      - {{.name: in, .offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global}}
      - {{.name: out, .offset: 8, .size: 8, .value_kind: global_buffer, .address_space: global}}
      - {{.name: n, .offset: 16, .size: 4, .value_kind: by_value}}
...
.end_amdgpu_metadata
"""


def gen_hip(arch, variant):
    """Device-only HIP source: the compiler emits the code object metadata, only the asm block is ours."""
    name = 'cvt_pk_u8_' + variant
    mode = ''.join(l + '\\n\\t' for l in VARIANTS[variant][1])
    if mode:
        mode += 's_nop 0\\n\\t'
    return f"""// target {arch}; no HIP headers needed (-nogpuinc -nogpulib)
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1, {BLOCK})))
void {name}(const unsigned* in, unsigned* out, unsigned n) {{
  unsigned i = __builtin_amdgcn_workgroup_id_x() * {BLOCK}u + __builtin_amdgcn_workitem_id_x();
  if (i >= n) return;
  unsigned x = in[2 * i], sel = in[2 * i + 1], d;
  asm volatile("{mode}v_cvt_pk_u8_f32 %0, %1, %2, %3"
               : "=v"(d) : "v"(x), "v"(sel), "v"(0x{SENTINEL:08x}u));
  out[i] = d;
}}
"""


def find_tool(names, override=None):
    if override:
        return override
    dirs = [None]
    for env in ('ROCM_PATH', 'HIP_PATH'):
        if os.environ.get(env):
            dirs += [os.path.join(os.environ[env], 'llvm', 'bin'), os.path.join(os.environ[env], 'bin')]
    dirs += ['/opt/rocm/llvm/bin', '/opt/rocm/bin']
    for d in dirs:
        for n in names:
            p = shutil.which(n) if d is None else shutil.which(n, path=d)
            if p:
                return p
    sys.exit('error: none of %s found (install ROCm / the HIP SDK or pass --clang / --llvm-mc / --lld)' % ', '.join(names))


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode:
        sys.exit('error: %s failed (%d):\n%s' % (' '.join(cmd), r.returncode, r.stdout))
    return r.stdout


def build(arch, variant, workdir, args):
    base = os.path.join(workdir, '%s_%s' % (variant, arch))
    if args.build == 'asm':
        mc = find_tool(['llvm-mc', 'llvm-mc.exe'], args.llvm_mc)
        lld = find_tool(['ld.lld', 'ld.lld.exe', 'lld'], args.lld)
        with open(base + '.s', 'w') as f:
            f.write(gen_asm(arch, variant, args.cov))
        run([mc, '-triple=amdgcn-amd-amdhsa', '-mcpu=' + arch, '-filetype=obj', '-o', base + '.o', base + '.s'])
        run([lld, '-shared', '-o', base + '.hsaco', base + '.o'])
    else:
        clang = find_tool(['clang', 'clang.exe', 'clang++', 'clang++.exe'], args.clang)
        with open(base + '.hip', 'w') as f:
            f.write(gen_hip(arch, variant))
        run([clang, '-x', 'hip', '--cuda-device-only', '--offload-arch=' + arch, '-nogpuinc', '-nogpulib',
             '-O1', '--no-gpu-bundle-output', '-o', base + '.hsaco', base + '.hip'])
    with open(base + '.hsaco', 'rb') as f:
        return f.read(), base + '.hsaco'


# ---------------------------------------------------------------- HIP

class Hip:
    def __init__(self):
        names = ['amdhip64_7.dll', 'amdhip64_6.dll', 'amdhip64.dll'] \
            if os.name == 'nt' else ['libamdhip64.so', 'libamdhip64.so.7', 'libamdhip64.so.6', 'libamdhip64.so.5']
        dirs = ['']
        for env in ('HIP_PATH', 'ROCM_PATH'):
            if os.environ.get(env):
                dirs.append(os.path.join(os.environ[env], 'bin' if os.name == 'nt' else 'lib'))
        if os.name != 'nt':
            dirs.append('/opt/rocm/lib')
        self.lib, err = None, []
        for d in dirs:
            for n in names:
                try:
                    self.lib = ctypes.CDLL(os.path.join(d, n) if d else n)
                    break
                except OSError as e:
                    err.append(str(e))
            if self.lib:
                break
        if not self.lib:
            raise RuntimeError('cannot load the HIP runtime (%s); tried: %s' % (names, '; '.join(err[-3:])))
        self.lib.hipGetErrorString.restype = ctypes.c_char_p
        self.lib.hipGetErrorString.argtypes = [ctypes.c_int]

    def check(self, code, what):
        if code != 0:
            raise RuntimeError('%s failed: HIP error %d (%s)' % (
                what, code, (self.lib.hipGetErrorString(code) or b'?').decode()))

    def call(self, fn, *a):
        self.check(getattr(self.lib, fn)(*a), fn)

    device = 0

    def count(self):
        self.call('hipInit', 0)
        n = ctypes.c_int()
        self.call('hipGetDeviceCount', ctypes.byref(n))
        return n.value

    def arch(self, device=None):
        """gcnArchName without relying on the struct layout: scan the property block for 'gfxNNNN'."""
        self.call('hipInit', 0)
        buf = ctypes.create_string_buffer(16384)
        fn = 'hipGetDevicePropertiesR0600' if hasattr(self.lib, 'hipGetDevicePropertiesR0600') \
            else 'hipGetDeviceProperties'
        self.call(fn, buf, self.device if device is None else device)
        m = re.search(rb'gfx[0-9a-f]{3,4}', buf.raw)
        if not m:
            raise RuntimeError('gcnArchName not found in the device properties; pass --arch')
        return m.group(0).decode()

    def run_kernel(self, hsaco, kname, words):
        """words: list of (in_bits, sel). Returns the list of output words."""
        lib, n = self.lib, len(words)
        self.call('hipSetDevice', self.device)
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        img = ctypes.create_string_buffer(hsaco, len(hsaco))
        self.call('hipModuleLoadData', ctypes.byref(mod), img)
        self.call('hipModuleGetFunction', ctypes.byref(fn), mod, kname.encode())
        din, dout = ctypes.c_void_p(), ctypes.c_void_p()
        self.call('hipMalloc', ctypes.byref(din), ctypes.c_size_t(8 * n))
        self.call('hipMalloc', ctypes.byref(dout), ctypes.c_size_t(4 * n))
        host = (ctypes.c_uint32 * (2 * n))(*[w for pair in words for w in pair])
        self.call('hipMemcpy', din, host, ctypes.c_size_t(8 * n), 1)   # hipMemcpyHostToDevice
        zero = (ctypes.c_uint32 * n)()
        self.call('hipMemcpy', dout, zero, ctypes.c_size_t(4 * n), 1)
        a_in, a_out, a_n = ctypes.c_void_p(din.value), ctypes.c_void_p(dout.value), ctypes.c_uint32(n)
        params = (ctypes.c_void_p * 3)(ctypes.addressof(a_in), ctypes.addressof(a_out), ctypes.addressof(a_n))
        blocks = (n + BLOCK - 1) // BLOCK
        self.call('hipModuleLaunchKernel', fn, blocks, 1, 1, BLOCK, 1, 1, 0, None, params, None)
        self.call('hipDeviceSynchronize')
        res = (ctypes.c_uint32 * n)()
        self.call('hipMemcpy', res, dout, ctypes.c_size_t(4 * n), 2)   # hipMemcpyDeviceToHost
        out = list(res)
        self.call('hipFree', din)
        self.call('hipFree', dout)
        self.call('hipModuleUnload', mod)
        return out


# ---------------------------------------------------------------- reporting

def analyse(rows, results, show):
    """results: variant -> list of words. Returns per-variant summary dict."""
    summary = {}
    for var, words in results.items():
        info = {'merge_ok_rows': sum(merge_ok(w, r[2]) for w, r in zip(words, rows)), 'rows': len(rows), 'models': {}}
        for m in MODELS:
            bad = [i for i, (w, r) in enumerate(zip(words, rows)) if w != model_word(m, r[1], r[2])]
            info['models'][m] = {'mismatches': len(bad), 'first': bad[:show]}
        info['matching_models'] = [m for m, v in info['models'].items() if v['mismatches'] == 0]
        summary[var] = info
    return summary


def fmt_row(rows, words, i):
    lab, b, s = rows[i]
    return 'in=%08x (%s) sel=%08x -> hw=%08x' % (b, lab, s, words[i])


def print_report(rows, results, summary, arch, show):
    print('\n== device/arch: %s, %d rows (%d inputs x %d selects) ==' % (arch, len(rows), len(rows) // len(SELECTS),
                                                                         len(SELECTS)))
    for var, info in summary.items():
        print('\n-- variant %s: %s' % (var, VARIANTS[var][0]))
        print('   S2 merge (other bytes == sentinel): %d/%d rows' % (info['merge_ok_rows'], info['rows']))
        for m, v in info['models'].items():
            print('   model %-20s mismatches %5d/%d %s' % (m, v['mismatches'], info['rows'],
                                                         'MATCH' if v['mismatches'] == 0 else ''))
        print('   models matching all rows: %s' % (info['matching_models'] or 'none'))
        for m, v in info['models'].items():
            if v['mismatches']:
                print('   first mismatches vs %s:' % m)
                for i in v['first']:
                    print('     %s  model=%08x' % (fmt_row(rows, results[var], i), model_word(m, rows[i][1], rows[i][2])))
    # compact table: select 0 only, byte value per variant and per model
    print('\n-- compact table (sel=0, selected byte): input | %s | %s' % (
        ' '.join('%-7s' % v for v in results), ' '.join(m.split('_')[0][:7].ljust(7) for m in MODELS)))
    shown = 0
    for i, (lab, b, s) in enumerate(rows):
        if s != 0:
            continue
        hw = ['%7d' % (results[v][i] & 0xFF) for v in results]
        md = ['%7d' % MODELS[m](val(b)) for m in MODELS]
        flag = '' if len({results[v][i] & 0xFF for v in results} | {MODELS['rne_sat_nan0'](val(b))}) == 1 else ' *'
        if lab.startswith('(') and not flag and shown > 60:
            continue
        shown += 1
        print('%-22s %08x | %s | %s%s' % (lab[:22], b, ' '.join(hw), ' '.join(md), flag))
    print('(* = a variant disagrees with the RNE model; (x/y)*255 rows are elided when all agree)')


# ---------------------------------------------------------------- self test

def self_test():
    ok = True

    def check(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print('FAIL %s: got %r want %r' % (name, got, want))

    b = lambda x: bits(x)
    # worked rows: (value, rne, trunc, half-away, wrap)
    table = [
        (2.5, 2, 2, 3, 2), (1.5, 2, 1, 2, 1), (0.5, 0, 0, 1, 0), (127.5, 128, 127, 128, 127),
        (254.5, 254, 254, 255, 254), (255.5, 255, 255, 255, 255), (256.0, 255, 255, 255, 0),
        (-0.4, 0, 0, 0, 0), (-0.5, 0, 0, 0, 0), (-1.0, 0, 0, 0, 255), (-255.0, 0, 0, 0, 1),
        (255.4, 255, 255, 255, 255), (1e10, 255, 255, 255, int(f32(1e10)) & 0xFF),
        (math.inf, 255, 255, 255, 0), (-math.inf, 0, 0, 0, 0), (math.nan, 0, 0, 0, 0),
        (0.0, 0, 0, 0, 0), (3.0, 3, 3, 3, 3), (254.75, 255, 254, 255, 254),
    ]
    for x, *want in table:
        got = [MODELS[m](x) for m in MODELS]
        check('models(%r)' % x, got, want)
    # merge: sel selects the byte, high bits ignored
    check('merge sel0', merge(0x11, 0), 0xA5C3E111)
    check('merge sel1', merge(0x11, 1), 0xA5C3117B)
    check('merge sel2', merge(0x11, 2), 0xA511E17B)
    check('merge sel3', merge(0x11, 3), 0x11C3E17B)
    check('merge sel high bits', merge(0x11, 0xFFFFFFFD), 0xA5C3117B)
    check('merge byte masked', merge(0x1FF, 0), 0xA5C3E1FF)
    check('extract', extract_byte(0xA5C3117B, 1), 0x11)
    check('merge_ok good', merge_ok(0xA5C3117B, 1), True)
    check('merge_ok bad', merge_ok(0xA5C3117C, 1), False)
    # input table sanity
    t = input_table()
    check('unique bits', len({x for _, x in t}), len(t))
    for must in (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7FA00000, 0x00000001, 0x7F7FFFFF, bits(2.5)):
        check('input has %08x' % must, must in {x for _, x in t}, True)
    check('nextafter 2.5', val(bits(2.5) + 1) > 2.5, True)
    check('rows count', len(build_rows()), len(t) * len(SELECTS))
    # models must differ on the interesting rows so the probe discriminates
    rows = build_rows()
    for m in MODELS:
        check('model %s distinct' % m, any(model_word(m, r[1], r[2]) != model_word('rne_sat_nan0', r[1], r[2])
                                            for r in rows), m != 'rne_sat_nan0')
    # asm text mentions the expected instructions
    a = gen_asm('gfx1030', 'rtz')
    check('asm rtz', 's_round_mode 0xf' in a and 'v_cvt_pk_u8_f32 v6, v3, v4, v5' in a, True)
    check('asm flush', 's_denorm_mode 0' in gen_asm('gfx1030', 'flush'), True)
    h = gen_hip('gfx1030', 'rtz')
    check('hip rtz', 's_round_mode 0xf\\n\\ts_nop 0\\n\\tv_cvt_pk_u8_f32 %0, %1, %2, %3' in h and 'cvt_pk_u8_rtz(' in h
          and '0xa5c3e17bu' in h, True)
    check('hip flush', 's_denorm_mode 0' in gen_hip('gfx1030', 'flush'), True)
    check('hip default has no mode', any(m in gen_hip('gfx1030', 'default') for m in ('s_round_mode', 's_denorm_mode', 's_nop')), False)
    check('asm default has no mode', 's_round_mode' in gen_asm('gfx1030', 'default'), False)
    print('self-test %s' % ('OK' if ok else 'FAILED'))
    return 0 if ok else 1


# ---------------------------------------------------------------- main

def _same_path(left, right):
    left_p = Path(left)
    right_p = Path(right)
    if os.path.normcase(str(left_p.resolve())) == os.path.normcase(str(right_p.resolve())):
        return True
    return left_p.exists() and right_p.exists() and left_p.samefile(right_p)


def _validate_output_path(flag, raw_path):
    if not raw_path or not raw_path.strip():
        raise ValueError('%s path must not be empty' % flag)
    if raw_path.endswith(('/', '\\')):
        raise ValueError('%s path is a directory path: %s' % (flag, raw_path))
    p = Path(raw_path)
    if p.is_symlink():
        raise ValueError('%s path must not be a symlink: %s' % (flag, raw_path))
    if p.is_dir():
        raise ValueError('%s path is a directory: %s' % (flag, raw_path))
    if p.exists() and not p.is_file():
        raise ValueError('%s path is not a regular file: %s' % (flag, raw_path))
    for parent in p.parents:
        if parent.is_symlink() and not parent.is_dir():
            raise ValueError('%s parent path is a symlink to non-directory: %s' % (flag, parent))
        if parent.exists() and not parent.is_dir():
            raise ValueError('%s parent path is not a directory: %s' % (flag, parent))
    return p


def _validate_args(args):
    if args.show < 0:
        raise ValueError('--show must be >= 0, got %d' % args.show)

    keep_path = None
    if args.keep is not None:
        if not args.keep.strip():
            raise ValueError('--keep path must not be empty')
        keep_path = Path(args.keep)
        if keep_path.exists() and not keep_path.is_dir():
            raise ValueError('--keep path is not a directory: %s' % args.keep)
        if keep_path.is_symlink() and not keep_path.is_dir():
            raise ValueError('--keep path is a broken symlink: %s' % args.keep)
        for parent in keep_path.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError('--keep parent path is not a directory: %s' % parent)

    cmp_path = None
    if args.compare:
        cmp_path = Path(args.compare)
        if cmp_path.is_dir():
            raise ValueError('--compare path is a directory: %s' % args.compare)

    out_path = _validate_output_path('--out', args.out)
    json_path = _validate_output_path('--json', args.json) if args.json is not None else None

    if cmp_path is not None and _same_path(out_path, cmp_path):
        raise ValueError('--out path collides with --compare: %s' % args.out)
    if json_path is not None and cmp_path is not None and _same_path(json_path, cmp_path):
        raise ValueError('--json path collides with --compare: %s' % args.json)
    if json_path is not None and _same_path(out_path, json_path):
        raise ValueError('--out and --json paths must not collide: %s' % args.out)
    if keep_path is not None:
        if _same_path(keep_path, out_path):
            raise ValueError('--keep and --out paths must not collide: %s' % args.out)
        if json_path is not None and _same_path(keep_path, json_path):
            raise ValueError('--keep and --json paths must not collide: %s' % args.json)
        if cmp_path is not None and _same_path(keep_path, cmp_path):
            raise ValueError('--keep and --compare paths must not collide: %s' % args.compare)


def _write_atomic(path, content):
    """Write content to a sibling temp file and atomically replace path, cleaning up on failure."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            newline='\n',
            dir=target.parent,
            prefix='.%s.' % target.name,
            suffix='.tmp',
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', type=int, default=0, help='HIP device index (default 0; every device is listed)')
    ap.add_argument('--arch', help='target, e.g. gfx1030, gfx1032, gfx1100 (default: from the HIP device)')
    ap.add_argument('--asm-only', action='store_true', help='build only, do not launch (no GPU needed)')
    ap.add_argument('--self-test', action='store_true', help='check the models without any device')
    ap.add_argument('--keep', help='directory to keep .hip/.s/.o/.hsaco files (default: temporary)')
    ap.add_argument('--out', default=os.path.join('out', 'rdna_cvt_pk_u8.tsv'), help='compact result file')
    ap.add_argument('--compare', default=DEFAULT_REFERENCE, help='stored result file to compare the rows with')
    ap.add_argument('--json', help='also write the verbose JSON (rows and model summary, about 860 KB)')
    ap.add_argument('--show', type=int, default=8, help='first mismatching rows listed per model')
    ap.add_argument('--cov', type=int, default=5, choices=(4, 5), help='HSA code object version')
    ap.add_argument('--build', choices=('hip', 'asm'), default='hip',
                    help='hip: compile a device-only HIP source with clang (default); asm: hand-written assembly')
    ap.add_argument('--clang', help='clang executable (--build hip)')
    ap.add_argument('--llvm-mc', help='llvm-mc executable (--build asm)')
    ap.add_argument('--lld', help='ld.lld executable (--build asm)')
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    temp_workdir = None
    try:
        _validate_args(args)

        hip, arch = None, args.arch
        if not args.asm_only:
            try:
                hip = Hip()
                hip.device = args.device
                for i in range(hip.count()):
                    print('HIP device %d: %s%s' % (i, hip.arch(i), '  <- used' if i == args.device else ''))
                if not arch:
                    arch = hip.arch()
            except RuntimeError as e:
                print('error: %s' % e, file=sys.stderr)
                return 2
        if not arch:
            print('error: --asm-only needs --arch', file=sys.stderr)
            return 2
        arch = arch.split(':')[0]

        if args.keep:
            workdir = args.keep
            os.makedirs(workdir, exist_ok=True)
        else:
            temp_workdir = tempfile.mkdtemp(prefix='cvt_pk_u8_')
            workdir = temp_workdir

        images = {}
        for var in VARIANTS:
            images[var] = build(arch, var, workdir, args)
            print('built %s (%d bytes): %s' % (var, len(images[var][0]), images[var][1]))
        if args.asm_only:
            return 0

        rows = build_rows()
        words = [(r[1], r[2]) for r in rows]
        results = {}
        try:
            for var in VARIANTS:
                results[var] = hip.run_kernel(images[var][0], 'cvt_pk_u8_' + var, words)
        except RuntimeError as e:
            print('error: %s' % e, file=sys.stderr)
            return 3
        summary = analyse(rows, results, args.show)
        print_report(rows, results, summary, arch, args.show)
        header = ['# rdna_cvt_pk_u8 v1: v_cvt_pk_u8_f32 vD, vIn, vSel, vS2 measured by tools/rdna_cvt_pk_u8.py',
                  '# sentinel (S2) %08x' % SENTINEL, '# device %s' % arch,
                  '# columns (tab separated, words in hex): label in sel default rtz flush',
                  '# default = MODE at launch default, rtz = s_round_mode 0xf, flush = s_denorm_mode 0']
        lines = ['\t'.join([r[0], '%08x' % r[1], '%08x' % r[2]] + ['%08x' % results[v][i] for v in VARIANTS])
                 for i, r in enumerate(rows)]
        _write_atomic(args.out, '\n'.join(header + lines) + '\n')
        print('\nresults written to %s' % os.path.abspath(args.out))
        if args.json:
            payload = json.dumps(
                {'arch': arch, 'sentinel': SENTINEL,
                 'rows': [{'label': r[0], 'in': '%08x' % r[1], 'sel': '%08x' % r[2],
                           **{v: '%08x' % results[v][i] for v in results}} for i, r in enumerate(rows)],
                 'summary': summary},
                indent=1,
            )
            _write_atomic(args.json, payload)
            print('verbose JSON written to %s' % os.path.abspath(args.json))
        if args.compare and os.path.exists(args.compare):
            with open(args.compare, encoding='utf-8') as f:
                stored = [l.rstrip('\n') for l in f if not l.startswith('#')]
            if stored == lines:
                print('rows identical to %s' % args.compare)
            else:
                print('rows DIFFER from %s (%d of %d rows)' % (args.compare, sum(a != b for a, b in zip(stored, lines)), len(lines)))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print('error: %s' % e, file=sys.stderr)
        return 1
    finally:
        if temp_workdir is not None:
            shutil.rmtree(temp_workdir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
