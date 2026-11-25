"""Check bounded runtime mesh specialization against the existing CPU oracle."""

from pathlib import Path
import struct
import sys
from unittest.mock import patch

from mesh_dynamic_geometry_check import CASES, CODE, MESH_LIMITS, decode, words

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mesh_dynamic_aot import enumerate_variants, lower


def select(variants, geometry):
    matches = [variant for variant in variants
               if all((geometry[index] == value) == equal for index, value, equal in variant.conditions)]
    assert len(matches) == 1, (geometry, matches)
    return matches[0]


def expect_refusal(decoded, geometry_word_count, text, limits=MESH_LIMITS):
    try:
        lower(decoded, geometry_word_count, limits)
    except ValueError as error:
        assert text in str(error), error
    else:
        raise AssertionError(f"Expected refusal: {text}")


def check():
    decoded = decode(CODE)
    variants = enumerate_variants(decoded, 2, MESH_LIMITS)
    for name, packed_counts, second_index, vertex_count, triangles in CASES:
        variant = select(variants, (packed_counts, second_index))
        assert variant.counts == (vertex_count, len(triangles)), (name, variant.counts)
        assert variant.indices == {i: list(indices) for i, indices in enumerate(triangles)}, name
        assert {k: tuple(v) for k, v in variant.positions.items()} == {
            i: words((2 * (i % 2) - 1, 2 * (i // 2) - 1, 0, 1)) for i in range(vertex_count)
        }, name
        assert {k: tuple(v) for k, v in variant.attributes.items()} == {
            i: words((i % 2, i // 2, 0.5, 1)) for i in range(vertex_count)
        }, name

    # A different immediate in the source changes the accepted inputs and output.
    original = struct.pack("<I", 0x300402)
    changed = struct.pack("<I", 0x300401)
    assert CODE.count(original) == 1
    varied = decode(CODE.replace(original, changed))
    varied_variants = enumerate_variants(varied, 2, MESH_LIMITS)
    assert select(varied_variants, (0x2004, 0x300402)).counts == (0, 0)
    assert select(varied_variants, (0x2004, 0x300401)).indices[1] == [1, 1, 3]
    assert lower(decoded, 2, MESH_LIMITS) != lower(varied, 2, MESH_LIMITS)

    # An input reaching a non-equality predicate or unguarded export is unsupported.
    expect_refusal([(pc, text.replace("s_cmp_eq_u32 s15", "s_cmp_lt_u32 s15"))
                    for pc, text in decoded], 2, "Unproven runtime geometry word")
    expect_refusal([(pc, text.replace("v_mov_b32_e32 v3, 0", "v_mov_b32_e32 v3, s14"))
                    for pc, text in decoded], 2, "Unproven runtime geometry word")
    expect_refusal([(pc, text.replace("v_cndmask_b32_e64 v9, v9, s14", "v_cndmask_b32_e64 v9, s14, v9"))
                    for pc, text in decoded], 2, "Unproven runtime geometry word")
    expect_refusal([(pc, "s_branch -1" if text == "s_cbranch_scc0 1" else text)
                    for pc, text in decoded], 2, "Expected a forward mesh branch")
    expect_refusal(decoded, 1, "Geometry read outside binding")
    expect_refusal(decoded, 2, "Conflicting mesh counts", limits=(3, 1))
    with patch("mesh_dynamic_aot.MAX_VARIANTS", 1):
        expect_refusal(decoded, 2, "exceeds 1 input cases")

    hlsl = lower(decoded, 2, MESH_LIMITS)
    assert hlsl.count("SetMeshOutputCounts(") == 1
    assert "ByteAddressBuffer geometryData : register(t0);" in hlsl
    assert "geometryData.Load(0)" in hlsl and "geometryData.Load(4)" in hlsl
    assert "out vertices Vertex vertices[4], out indices uint3 primitives[2]" in hlsl
    print("Runtime mesh specializations matched the CPU cases and rejected unproven uses")


if __name__ == "__main__":
    check()
