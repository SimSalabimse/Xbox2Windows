# Compute input view tokens

A translated compute dispatch binds a view token as a buffer input only when the token is a
constant-buffer view, or a buffer shader-resource view of a desktop resource. Every other token
used to share one census sentence, `an input view token is not a buffer view of a desktop resource`.
`indirect_table_buffer` now refuses that token with the record it resolved to, in this order:

1. `an input view token is unknown`
2. `an input view token is a sampler`
3. `an input view token is an unordered access view`
4. `an input view token is a render or depth view`
5. `an input view token has a truncated constant buffer view`
6. `an input view token has an unrecognized slot`
7. `an input view token has a truncated shader resource view`
8. `an input view token is not a buffer view of a desktop resource` (a shader resource view whose
   dimension is not a buffer; a texture token keeps this historical sentence)
9. `an input view token names no resource`
10. `an input view token does not name a resource`
11. `an input view token's placed or copied resource has no desktop backing`
12. `an input view token has no desktop resource`

The census still keeps the first reason of each key. A later range check is unchanged: a view that
passes this gate can still be refused because its element size, alignment, alias, or span fails.

A typed or raw buffer view whose resource has a desktop object passes this gate. The fixture binds
two of those on the placed input buffer that aliases title memory, an `R32_UINT` view and a raw
`R32_TYPELESS` view, and reads all 320 words of case A back. A placed buffer the desktop cannot
alias is still refused with `an input is in a placed buffer that does not alias title memory`. A
placed or copied resource with no desktop object stays refused; the fixture does not construct one,
and the Linux reference is what checks that sentence.

`tests/startup_compute_dispatch_input_check.py` still expects the historical sentence for
`dispatch-input-token`. This note does not change that test.

```bash
./out/pe-env/bin/python -B tests/startup_compute_input_view_token_check.py
```

That command runs the reference and the controls. It does not dispatch.

On the Windows test PC:

```powershell
ctest --preset dev --no-tests=error -R "^startup_compute_(input_view_token_|dispatch_input_)" -V
```

`startup_compute_input_view_token_warp` and `_hardware` skip with 77 when the adapter is not
available. A skip is not a pass. This note has no Windows readback. Nothing here shows that a
title starts or runs.

The decision table is [input_view_token.h](../../src/startup_stubs/input_view_token.h). The fixture
is [startup_compute_input_view_token_check.py](../../tests/startup_compute_input_view_token_check.py).
