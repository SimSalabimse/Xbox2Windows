#!/usr/bin/env python3
"""Name the shape of a compute input view token, and bind a buffer view of a desktop resource.

usage:
  startup_compute_input_view_token_check.py
  startup_compute_input_view_token_check.py reference
  startup_compute_input_view_token_check.py controls
  startup_compute_input_view_token_check.py <startup_stream_consumer> <dxc> <warp|hardware>

reference and controls need no device. With no arguments both run. reference checks the decision
table against src/startup_stubs/input_view_token.h. controls checks that the runtime calls that
table, that a non-buffer shader resource view still uses the historical sentence, and that
tests/startup_compute_dispatch_input_check.py still expects that sentence for a texture token.

warp and hardware dispatch case A. A typed R32_UINT view and a raw R32_TYPELESS view of the placed
input buffer, which aliases title memory, match all 320 words. A texture token, an unknown token,
an unordered-access token, a sampler token, a typed view of a buffer the desktop cannot alias, and
a typed view that runs past a custom-heap buffer are refused. That past view is recorded and not
created on the desktop. The output stays at the sentinel. Exit 77 when the adapter is not available.
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src" / "startup_stubs" / "input_view_token.h"
DRAW = ROOT / "src" / "startup_stubs" / "d3d12_draw.inl"
DEVICE = ROOT / "src" / "startup_stubs" / "d3d12_xs_device.cpp"
CONSUMER = ROOT / "tests" / "startup_stream_consumer.cpp"
EXISTING = ROOT / "tests" / "startup_compute_dispatch_input_check.py"

HISTORICAL = "an input view token is not a buffer view of a desktop resource"
BUFFER = 1


def facts(**overrides):
    base = {
        "known": True,
        "slot": 18,
        "desc_bytes": 40,
        "view_dimension": BUFFER,
        "resource_pointer": True,
        "wrapper": True,
        "resource": True,
        "desktop": True,
        "placement": 1,
    }
    base.update(overrides)
    return base


def classify(item):
    """Same order as input_view_token_refusal. None means the token passes this gate."""
    if not item["known"]:
        return "an input view token is unknown"
    if item["slot"] == 22:
        return "an input view token is a sampler"
    if item["slot"] == 19:
        return "an input view token is an unordered access view"
    if item["slot"] in (20, 21):
        return "an input view token is a render or depth view"
    if item["slot"] == 17:
        if item["desc_bytes"] < 12:
            return "an input view token has a truncated constant buffer view"
        return None
    if item["slot"] != 18:
        return "an input view token has an unrecognized slot"
    if item["desc_bytes"] < 40:
        return "an input view token has a truncated shader resource view"
    if item["view_dimension"] != BUFFER:
        return HISTORICAL
    if not item["resource_pointer"] or not item["wrapper"]:
        return "an input view token names no resource"
    if not item["resource"]:
        return "an input view token does not name a resource"
    if not item["desktop"]:
        if item["placement"] in (1, 2):
            return "an input view token's placed or copied resource has no desktop backing"
        return "an input view token has no desktop resource"
    return None


CASES = (
    ("unknown", facts(known=False, slot=0, desc_bytes=0, resource_pointer=False, wrapper=False, resource=False, desktop=False),
     "an input view token is unknown"),
    ("sampler", facts(slot=22), "an input view token is a sampler"),
    ("uav", facts(slot=19), "an input view token is an unordered access view"),
    ("rtv", facts(slot=20), "an input view token is a render or depth view"),
    ("dsv", facts(slot=21), "an input view token is a render or depth view"),
    ("short-cbv", facts(slot=17, desc_bytes=11), "an input view token has a truncated constant buffer view"),
    ("cbv", facts(slot=17, desc_bytes=12), None),
    ("other-slot", facts(slot=16), "an input view token has an unrecognized slot"),
    ("short-srv", facts(desc_bytes=39), "an input view token has a truncated shader resource view"),
    ("texture", facts(view_dimension=4), HISTORICAL),
    ("texture-without-desktop", facts(view_dimension=4, desktop=False), HISTORICAL),
    ("null-resource", facts(resource_pointer=False, wrapper=False, resource=False, desktop=False),
     "an input view token names no resource"),
    ("not-a-wrapper", facts(wrapper=False, resource=False, desktop=False), "an input view token names no resource"),
    ("wrong-kind", facts(resource=False, desktop=False), "an input view token does not name a resource"),
    ("placed-without-desktop", facts(desktop=False, placement=1),
     "an input view token's placed or copied resource has no desktop backing"),
    ("copied-without-desktop", facts(desktop=False, placement=2),
     "an input view token's placed or copied resource has no desktop backing"),
    ("no-desktop", facts(desktop=False, placement=0), "an input view token has no desktop resource"),
    ("committed-buffer", facts(placement=0), None),
    ("aliased-buffer", facts(placement=1), None),
    ("copied-buffer-gate", facts(placement=2), None),
)


def header_reasons():
    text = HEADER.read_text(encoding="utf-8")
    return re.findall(r'return "([^"]+)"', text)


def reference():
    produced = []
    for name, item, expected in CASES:
        got = classify(item)
        assert got == expected, (name, got, expected)
        if got:
            produced.append(got)
    declared = header_reasons()
    assert declared, "input_view_token.h has no refusal sentences"
    assert set(produced) == set(declared), (set(produced) ^ set(declared))
    assert HISTORICAL in declared
    print(f"startup_compute_input_view_token reference: {len(CASES)} shapes, {len(set(declared))} refusal sentences")


RANGE = "an input range is outside its desktop buffer"
HEAP = "an input buffer view token's resource is not on a default or upload heap"


def binding_body(draw):
    start = draw.index("IndirectBufferView indirect_table_buffer")
    end = draw.index("\nstruct FormattedBufferDescriptor", start)
    return draw[start:end]


def first_backing_reason(body, applicable):
    for reason in re.findall(r'return refuse\("([^"]+)"\)', body):
        if applicable.get(reason):
            return reason
    return None


def backing_order(draw):
    """A recorded past-the-end view is refused for its range before its heap type."""
    body = binding_body(draw)
    past = {RANGE: True, HEAP: True}
    fitting = {RANGE: False, HEAP: True}
    assert first_backing_reason(body, past) == RANGE
    assert first_backing_reason(body, fitting) == HEAP, "a fitting view of a non-default heap must keep its heap reason"
    mutant = body.replace('return refuse("%s")' % RANGE, "")
    try:
        assert first_backing_reason(mutant, past) == RANGE
    except AssertionError:
        got = first_backing_reason(mutant, past)
        assert got == HEAP, got
        print("past range check removed: refused for heap type, as required")
    else:
        raise AssertionError("past negative still passed after the range check was removed")


def controls():
    reference()
    draw = DRAW.read_text(encoding="utf-8")
    assert "input_view_token_refusal" in draw
    assert 'return refuse("an input view token has an unrecognized slot")' in draw
    assert HISTORICAL not in draw, "the historical sentence now lives only in input_view_token.h"
    assert "an input is in a placed buffer that does not alias title memory" in draw
    assert RANGE in draw
    backing_order(draw)
    device = DEVICE.read_text(encoding="utf-8")
    assert 'skip = "a buffer shader resource view extends past the resource"' in device
    consumer = CONSUMER.read_text(encoding="utf-8")
    assert 'mode == "dispatch-input-token-past"' in consumer
    assert "D3D12_HEAP_TYPE_CUSTOM" in consumer
    existing = EXISTING.read_text(encoding="utf-8")
    assert "('dispatch-input-token', 'an input view token is not a buffer view of a desktop resource')" in existing
    print("startup_compute_input_view_token controls: runtime uses the table; the texture token sentence is unchanged")


def adapter_run(consumer, dxc, adapter):
    import os
    import subprocess
    import startup_gfx2hlsl_compute_check as C
    from startup_compute_dispatch_check import dispatch, lines, prepare_case
    from startup_stream_check import gfx2hlsl as G

    refused = (
        ("dispatch-input-token", HISTORICAL),
        ("dispatch-input-token-unknown", "an input view token is unknown"),
        ("dispatch-input-token-uav", "an input view token is an unordered access view"),
        ("dispatch-input-token-sampler", "an input view token is a sampler"),
        ("dispatch-input-token-copied", "an input is in a placed buffer that does not alias title memory"),
        ("dispatch-input-token-past", "an input range is outside its desktop buffer"),
    )
    bound = ("dispatch-input-token-typed", "dispatch-input-token-raw")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, adapter)
        env.pop("X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT", None)
        env.pop("X2W_D3D12_RING_RELEASE_AT_SUBMIT", None)
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected("A")

        got, result = dispatch(directory, consumer, env, "dispatch", probe=True)
        print(result.stdout, end="")
        assert got == untouched and len(lines(result, "refused: no translation")) == 1, result.stderr[-3000:]
        published = lines(result, "x2w-draw compute request")
        assert published and " published " in published[0], result.stderr[-3000:]
        key = int(published[0].split()[3], 16)
        out = subprocess.run([sys.executable, "-B", G.__file__, "--pending", cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f"{key:016x}: ok" in out, out

        got, result = dispatch(directory, consumer, env, "dispatch")
        assert got == expected and not lines(result, "refused"), result.stderr[-3000:]

        for mode in bound:
            got, result = dispatch(directory, consumer, env, mode)
            assert got == expected and len(lines(result, "Dispatch(3, 2, 2) translated")) == 1, (mode, result.stderr[-3000:])
            assert not lines(result, "refused:"), (mode, result.stderr[-3000:])
            errors = [line for line in lines(result, "x2w-d3d12-debug ") if " sev=0 " in line or " sev=1 " in line]
            assert not errors, errors[0]
        for mode, reason in refused:
            got, result = dispatch(directory, consumer, env, mode)
            assert got == untouched, "%s wrote the output" % mode
            found = lines(result, "refused: ")
            census = lines(result, "x2w-render compute census")
            where = lines(result, f"x2w-draw compute {key:016x} first refused input 0 (table ")
            assert len(found) == 1 and found[0].endswith("refused: " + reason), (mode, result.stderr[-3000:])
            assert len(census) == 3 and census[1].endswith(f'reason "{reason}": 1 keys, 1 dispatches'), (mode, census)
            assert len(where) == 1, (mode, result.stderr[-3000:])
        print(f"startup_compute_input_view_token: PASS on {adapter} "
              f"({len(bound)} buffer views bound, {len(refused)} tokens refused)")


def main(argv):
    if not argv:
        controls()
        return
    if argv == ["reference"]:
        reference()
        return
    if argv == ["controls"]:
        controls()
        return
    if len(argv) == 3 and argv[2] in ("warp", "hardware"):
        adapter_run(*argv)
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
