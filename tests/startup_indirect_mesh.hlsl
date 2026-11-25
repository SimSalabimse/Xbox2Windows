struct Vertex {
    float4 position : SV_Position;
    nointerpolation uint side : SIDE;
};

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh(uint3 group : SV_GroupID, out vertices Vertex verts[3], out indices uint3 triangles[1]) {
    SetMeshOutputCounts(3, 1);
    verts[0].position = float4(-1, -1, 0, 1);
    verts[1].position = float4(-1, 3, 0, 1);
    verts[2].position = float4(3, -1, 0, 1);
    verts[0].side = group.x;
    verts[1].side = group.x;
    verts[2].side = group.x;
    triangles[0] = uint3(0, 1, 2);
}

float4 pixel(Vertex input) : SV_Target {
    if (input.side == 0 && input.position.x < 2) return float4(1, 0, 0, 1);
    if (input.side == 1 && input.position.x >= 2) return float4(0, 1, 0, 1);
    discard;
    return float4(0, 0, 0, 1);
}
