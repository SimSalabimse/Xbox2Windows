# Published manifest for a one-component packed D16 buffer store

A direct compute `buffer_store_format` with the D16 bit and one component is
`packed_d16_x_uav`. It already compiles, and the in-process route already binds it
(`slot.packed_d16_x`, `native_buffer_store_view`). `format_compute_runtime_manifest`
published no line for it, so a program that compiled still had no runtime manifest.
`view_kind_name` also stopped at `float3_texture_uav`, so both `packed_d16_x_uav` and
`indexed_raw_srv` were printed as `unknown`.

The publisher now emits `uav <register> bufd16x table <parameter> <byte> bytes 2` when
the binding is one register and the payload is two bytes. The dispatch reader accepts
that line next to `bufhalf`. The view is still checked with `native_buffer_store_view`.
A format, stride or flag the check rejects keeps the reason
`formatted D16 write has an incompatible typed buffer UAV`. An `R32_UINT` element is
four bytes, two halfwords, and the store would zero-extend a 16-bit payload across both
of them, so the published route refuses that view as well.

The source-built fixture is `tests/startup_compute_packed_d16_x_uav_check.py`. It has no
title data. Thirteen lanes write `0x3c00 + lane` at byte 8 of a sentinel buffer; the
reference is that addition, which matches the low half of the program's `v_xor_b32` for
those lanes. The R32_UINT view, a stride other than 2, and a descriptor with the swizzle
flag set must refuse by name and leave the buffer untouched. Before the manifest exists
the dispatch is `no translation`.

On Linux the reference half of the fixture runs with no compiler and no device:

```bash
./out/pe-env/bin/python -B tests/startup_compute_packed_d16_x_uav_check.py
```

The compile and dispatch halves need Windows, DXC and the stand-in. WARP is the CI
adapter. A hardware adapter is selected the same way as the other `startup_compute_*`
dispatch tests and returns 77 when none is available.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_packed_d16_x_uav_' -V
```

A 900 s start-up run of a retail title on the Windows test PC, runtime built from `main`
`aec2e08`, counted 23 shaders and 86,436 untranslated dispatches (5.7% of that run's
untranslated compute dispatches) whose published-route refusal was
`unsupported startup runtime manifest bindings: UAV kind unknown` for this binding.
Those counts are from that run. This note does not include a Windows dispatch result for
the new fixture; that measurement is still to be taken on the Windows test PC.
