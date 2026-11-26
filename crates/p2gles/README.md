# p2gles

A thin OpenGL ES 3.0 renderer for POSTAL 2 maps with UE2-style lighting. It has two users:

- the `p2gles` binary, a fly-through viewer of the static world, used for benchmarks;
- `p2game` with `P2GAME_GLES=1`, which embeds it in the game (`crates/p2game/src/gles/`).

Scene data comes from `p2view::scene::build_scene`; nothing is parsed twice.

## Layout

| Module | What |
|---|---|
| `renderer` | The static world: bake into one vertex buffer, 48 m cells, frustum and fog culling, four passes (opaque, alpha-tested, terrain splat, blended) |
| `dynamic` | Everything else, handed over each frame: `mesh` (vertex layout, upload), `pass` (skinned, instanced and blended meshes), `canvas` (HUD), `particles`, `grade` (brightness, contrast, gamma) |
| `shaders` | GLSL ES 3.0 sources |
| `program` | Compile and link |

## Design notes

- **Lighting** is done in display (gamma) space, as UE2 does it: BSP is texture x lightmap x 2, static
  meshes are texture x baked vertex light, terrain layers blend through alpha maps.
- **Fog** is computed from per-vertex distance and mixed in linear light. World triangles with an edge
  over 16 m are split during the bake, because per-vertex distance is badly wrong on huge triangles.
  Whether an edge is split depends only on its two ends, so neighbours stay watertight.
- **One index buffer per draw record.** Panfrost caches index min/max per buffer in a small cache; with
  one shared buffer it rescanned the ranges on the CPU every frame.
- **Mipmaps** missing from a package are generated on the GPU: minifying a large texture without them
  thrashes the texture cache.
- **Anisotropic filtering** is opt-in; it costs several texture fetches per pixel.
- **Alpha test** lives in a separate program, because a `discard` anywhere in a program disables early
  depth testing on Mali.
- **`dynamic`** keys GPU meshes and textures by ids chosen by the host, which uploads each once. Packing
  a mesh (`pack_mesh`) makes no GL calls, so it can happen on another thread than the upload.

## Viewer

```
p2gles --game <POSTAL2Complete> --map suburbs-3 [--res 800x480] [--fullscreen] [--vsync]
p2gles --map estates --bench 20               # ms/frame, fps, draws, triangles every 5 s
p2gles --map suburbs-3 --screenshot out.png
```
WASD to move, mouse to look, Q/E down and up, Shift to go faster, Esc to quit. Other options:
`--cam x,y,z,yaw,pitch` (Unreal units), `--fog start,end` (metres), `--chunk METRES`, `--no-cull`.
The viewer draws the static world only.

`examples/smoke.rs` compiles every shader, draws through each path of `dynamic` and checks for GL
errors without game files: `xvfb-run -a cargo run --release -p p2gles --example smoke`.

## Diagnostics

| Variable | Effect |
|---|---|
| `P2GLES_NO_BC=1` | Decode DXT textures to RGBA8 |
| `P2GLES_ANISO=N` | Anisotropic filtering |
| `P2GLES_NO_GENMIPS=1` | Do not generate missing mip levels |
| `P2GLES_SKIP=opaque,masked,splat,blend` | Leave world passes out |
| `P2GLES_ONE_IBO=1` | One shared index buffer |
| `P2GLES_ONE_PROG=1` | Use the program without alpha test for the alpha-tested pass |
| `P2GLES_NO_RANGE=1` | Do not use `glDrawRangeElements` |
