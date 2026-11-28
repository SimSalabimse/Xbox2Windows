# Root-UAV global stores in the direct C++ compute path

The direct compute frontend (`src/gfx1013_compute_native.cpp`, `src/gfx1013_direct_compute.inl`) lowers
`global_store_dword`, `global_store_dwordx2`, `global_store_dwordx3` and `global_store_dwordx4` in the `saddr`
form. The same root UAV can be read with `global_load_dword` and the wider loads that the root-SRV path already
accepts. A root CBV or SRV stays load-only.

ADDR = SADDR + zext(VADDR) + sext(OFFSET[11:0]). SADDR is the unadvanced root UAV pair. The 32-bit VGPR is the
byte offset. The immediate must be a multiple of 4. The low two address bits are dropped. A store whose first or
last DWORD would wrap a 32-bit offset is skipped. A load in that case returns 0. `glc`, `slc` and `dlc` are
cache hints and are accepted. The binding's minimum size is the largest positive immediate plus the transfer
width.

Still refused: the `off` form (the address is a VGPR pair, not a root plus a 32-bit offset), FLAT and scratch,
an LDS result, byte and short ops, atomics, `saddr` 127, a store through a root that is not a UAV, an unaligned
immediate, a data range past v255, and a scalar address that is not the matching root pair.

The view has to cover every byte the program writes. An address outside that view does not reach the guest
memory that would have been there.

This is a direct-translation lowering. The startup runtime manifest and the native dispatch path have no
case for `root_buffer_uav`, as they have none for `root_buffer_srv`. A program that reaches those paths is
refused with `UAV kind root_buffer_uav` rather than dispatched.

## Fixture

`tests/native_global_store_check.py`

```powershell
ctest --preset dev --no-tests=error -R '^native_global_store_encodings$' -V
ctest --preset dev --no-tests=error -R '^native_global_store_hardware$' -V
```

`encodings` only disassembles the source-built words. `hardware` needs `compile_dxil`, `native_store_replay` and
shader model 6.6 with wave32. A missing wave32 adapter is a skip (77), not a pass.

The readback checks one root UAV across two groups: the four store widths, an address that aliases its data, a
negative immediate, even lanes under a narrowed EXEC, a guarded load of vaddr 0xfffffff4 offset 16, and a
poison store at that address which must not appear. An x4 at vaddr 0xffffffe8 offset 16 has its first DWORD
inside the 32-bit range and its last DWORD past it; the whole store is skipped. The DXIL upper bound for that
store is 4294967267 (`2^32 - 1 - (16 + 12)`). Dropping `(count - 1) * 4` from `last_offset` removes that
constant, so the hardware test fails. After the stores and `s_waitcnt`, each lane loads the DWORD it wrote at
the start of its record and writes it to a free slot. A second program binds two root SRVs and one root UAV
and must report `t0`, `t1` and `u0`. Swapping the two dwords of the x2 reference changes the expected buffer.
The x4 at immediate 2044 asks for a 2060-byte view and is not dispatched.
