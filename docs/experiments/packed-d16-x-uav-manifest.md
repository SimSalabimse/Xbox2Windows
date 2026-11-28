# Published manifest names for D16 buffer stores

`view_kind_name` names `packed_d16_x_uav` and `indexed_raw_srv`. A refusal that
mentions either kind prints that name.

`format_compute_runtime_manifest` emits
`uav <register> bufd16x table <parameter> <byte> bytes 2` only when the binding
kind is `packed_d16_x_uav`, the binding is one register, and the payload is two
bytes. The dispatch reader accepts that line next to `bufhalf` and still checks
the view with `native_buffer_store_view`. A format, stride or flag that check
rejects keeps the reason `formatted D16 write has an incompatible typed buffer UAV`.

A one-component `buffer_store_format` with the D16 bit is the half-buffer UAV.
It publishes `uav <register> bufhalf table <parameter> <byte> bytes 2`, the same
line the half-buffer fixture already requires. It does not publish `bufd16x`.

The source-built fixture is `tests/startup_compute_packed_d16_x_uav_check.py`.
It has no title data. It checks that this store's manifest line is the half-buffer
line and not `bufd16x`. Thirteen lanes write `0x3c00 + lane` at byte 8 of a
sentinel buffer. An R32_UINT view, a stride other than 2, and a descriptor with
the swizzle flag set must refuse with
`a half buffer store requires matching format, stride and structured bounds`.
Before the manifest exists the dispatch is `no translation`.

On Linux the reference half of the fixture runs with no compiler and no device:

```bash
./out/pe-env/bin/python -B tests/startup_compute_packed_d16_x_uav_check.py
```

The compile and dispatch halves need Windows, DXC and the stand-in. WARP is the CI
adapter. A hardware adapter is selected the same way as the other `startup_compute_*`
dispatch tests and returns 77 when none is available.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_(packed_d16_x_uav_|half_buffer_uav_)' -V
```

A 900 s start-up run of a retail title on the Windows test PC, runtime built from `main`
`aec2e08`, counted 23 shaders and 86,436 untranslated dispatches (5.7% of that run's
untranslated compute dispatches) whose published-route refusal was
`unsupported startup runtime manifest bindings: UAV kind unknown` for a one-component
D16 store. Those counts are from that run. Windows results for this correction are not
in this note.
