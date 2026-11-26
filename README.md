# zaman-ua fork: p2gles and performance work for weak ARM boards (Mali-G31)

This fork tries to make the port playable on a TV box: Amlogic S905X3 (4x Cortex-A55, **Mali-G31 MC1**),
Debian trixie, Mesa/Panfrost, OpenGL ES 3.1. The author's README follows below, unchanged; the fork's
`main` branch matches the upstream repository and the work is on `p2gles`.

> **Status.** Everything since the first world-only renderer has been developed and checked on an x86
> desktop (Intel UHD 730 and software GLES). **The current code has not been measured on the Mali
> board yet**, so no frame rate is claimed for it.

## What is different

1. **`P2GAME_GLES=1`**: the game logic stays on Bevy, but Bevy's renderer is not created; a thin
   OpenGL ES 3.0 renderer ([`crates/p2gles`](crates/p2gles/README.md)) draws the frame.
2. **Cheaper game logic**, for both renderers.
3. **Board support** for the stock Bevy renderer on Panfrost (an opt-in build feature), and lower memory
   use.

## Why a second renderer

On the board the Bevy port ran at about 3.5 fps (~285 ms per frame, 800x480). The measurements pointed
at neither shaders nor pixels: about 40 ms were Bevy's fixed overhead on its compatibility path even
with an empty view, the world took roughly 2400 draw calls, and game logic about 30 ms.

A first experiment that drew only the static world reached 95 fps (suburbs-3) and 125 fps (estates) on
the board with Mesa 26.1, which justified putting such a renderer into the game. Those two numbers are
for the standalone world viewer at that time: no characters, logic, HUD or sky, and none of the later
changes.

| Mali-G31 MC1, 800x480, spawn view (measured before this fork's later work) | Mesa 25.0 | Mesa 26.1 |
|---|---|---|
| p2gles world only, suburbs-3 | 57 ms | 10.5 ms |
| p2gles world only, estates | 47 ms | 8 ms |
| Bevy port, whole game, suburbs-3 | ~285 ms | ~282 ms |
| Original game (i386) through box64 | 17 fps | 17-19 fps |

## Running the game on p2gles

```
P2GAME_GLES=1 p2game --game <POSTAL2Complete folder>
P2GAME_GLES=1 p2game --game <folder> --map suburbs-3
```

| Variable | Effect |
|---|---|
| `P2GAME_GLES=1` | Draw with p2gles instead of Bevy's renderer |
| `P2GAME_GLES_SCALE=0.75` | Draw the 3D scene at a fraction of the window resolution (0.25-1); the HUD stays native. For frames limited by pixel fill: 0.75 shades 56% of the pixels, 0.5 a quarter |
| `P2GAME_GLES_THREAD=0` | Keep all GL calls on the main thread instead of the render thread |
| `P2GAME_NO_POSE_CULL=1` | Pose and draw characters even when no camera sees them |
| `P2GAME_SERIAL_POSE=1` | Compute poses on one thread |

Every 5 s the log has a `gles:` line. If `main thread waited` is well above zero, the frame is limited
by rendering (try `P2GAME_GLES_SCALE`; more CPU cores will not help). Near zero means it is limited by
the CPU. Total CPU load does not tell the two apart (30% of four cores can be one busy core), so look
per thread: `top -H -p $(pidof p2game)`.

## What was improved, and why it is done that way

### Rendering (`P2GAME_GLES=1`)

- **The static world is merged into a few hundred draws.** BSP, terrain and static meshes go into one
  vertex buffer, cut into 48 m cells and culled by frustum and fog.
- **The merge waits two script ticks after a level loads.** Level-start script destroys the meshes of
  other days and holidays, so only then is it known which actors stay. Excluding every grouped actor at
  load time instead had left 2407 of them as separate draws on suburbs-3; now 283 stay dynamic
  (estates: 770 -> 197). It is the same rule the Bevy path uses to merge meshes.
- **Fog is per vertex, and large triangles are split.** Per-pixel fog would cost fill rate, which this
  GPU does not have. But per-vertex distance is wrong on huge triangles: the main menu is a 625 m shaft
  made of 12 of them and came out black. World triangles longer than 16 m are split once, at load
  (suburbs-3: 20% more world vertices, 1.5% more triangles in the spawn view). Fog blends in linear
  light like Bevy (gamma 2.0 as a cheap stand-in for 2.2).
- **GL runs on its own thread.** While frame N is submitted and presented, the game already runs frame
  N+1, so the frame time approaches the longer of the two instead of their sum. One frame at most is in
  flight; the price is one more frame of input latency.
- **Only visible characters are posed.** The game used to compute the pose of every character and write
  one transform per bone each frame, including those behind the camera. Posing now runs after the
  cameras have moved and skips what no camera can see (ragdolls are always posed).
- **Less redundant GL work:** uniforms are sent only when they change, characters upload a skeleton
  once for all their parts, HUD and particle meshes reuse their buffers.
- **Each draw has its own index buffer.** With one shared buffer Panfrost rescanned index ranges on the
  CPU every frame (`u_vbuf_get_minmax_index`; its per-buffer cache is small).
- **DXT textures are uploaded as they are** (video memory 636 -> 256 MB in the world viewer).

### Game logic

- **Bone attachments no longer need a full pose.** Placing a head or a weapon evaluated the whole
  skeleton of every character on the map. Only the bone chains under the attachments are computed now,
  by the same code as the full pose.
- **Property handles.** Host code looked script properties up by name thousands of times per tick.
  `prop!("Location")` resolves a name to a slot once per class (`uscript::prop`). Only hot paths use it.
- **Work done only when needed:** a moving collider builds its BVH when something is first traced
  against it, not on every rotation; particle settings are re-read from script only after script wrote
  to the emitter.
- **Cheap hashing** for maps keyed by object ids (a side effect: headless runs of the same build now
  repeat each other apart from the wall clock).
- **Poses are computed on several cores** when at least 8 visible characters need one. The gain is
  small (see below).

### Measurements on x86

These show the direction and size of each change, not what the board will do.

| What | Before | After |
|---|---|---|
| Instructions per logic tick (headless suburbs-3, callgrind) | 15.1 M | 9.3 M |
| Logic tick, suburbs-3 (headless, before property handles) | 2.19 ms | 1.58 ms |
| Logic tick, estates (same) | 2.13 ms | 1.52 ms |
| Whole frame on p2gles, suburbs-3 (Intel, offscreen; effect of view culling and related changes) | 6.23 ms | 5.24 ms |
| Whole frame on p2gles, estates (same) | 5.28 ms | 4.47 ms |
| Posing every character, 4 cores: serial -> parallel (suburbs-3) | 0.66 ms | 0.50 ms |

With view culling on there is rarely enough posing left to split, and parallel posing makes no
measurable difference to the whole frame.

### How behaviour was checked

- After 900 headless ticks, every property of every script object was compared with the previous
  build. For the property handles only wall-clock fields differ, on estates and suburbs-3. For the
  engine optimisations before them the same held on estates; the previous build did not repeat itself
  on suburbs-3, and the differences stayed within its own run-to-run spread.
- Temporary cross-checks compared the new shortcut with the old computation on live data (partial
  poses bit for bit, property handles, skipped particle reads, culled characters never being visible).
- `p2game --smoke --windowed` (menu, New Game, intro, first day, walk, fire, level portal) passes on a
  virtual X display, and screenshots were compared between builds.

### Stock Bevy renderer on the board (`arm-board` build option)

Everything that changes the author's Bevy renderer is behind the cargo feature `arm-board` of
`p2game`, **off by default**:

- Bevy's WebGL2 path outside the browser. Bevy turns its `webgl` code on only for `wasm32`; the copies
  in `patches/` add a `native-webgl2` feature that does the same on native. Without it they build
  exactly as published. The mode is a compromise for weak GPUs (for example 4 KB uniform buffers and a
  single directional light), which is why it must not be on elsewhere.
- wgpu's OpenGL ES backend. Bevy does not accept Panfrost's Vulkan driver. On the board also set
  `WGPU_EGL_X11=1`, so wgpu opens the X11 display itself.
- MSAA off on all cameras (`P2_MSAA=1` restores it) and the camera's far plane at the end of the fog
  (`P2_NO_FOG_CULL=1`).

```
cargo build --release -p p2game --features arm-board
```

Without the feature, `bevy*`, `wgpu*` and `naga*` resolve to the same crates, versions and features as
upstream. `patches/` is still applied (cargo patches cannot depend on a feature). Besides the gated
code it holds three small fixes that decide at run time: opening the X11 display (only with
`WGPU_EGL_X11`), uniform buffers on GPUs without vertex storage buffers, and not crashing on level
change when Bevy's renderer is absent, **which `P2GAME_GLES=1` needs**.

`P2GAME_GLES=1` itself does not need `arm-board`.

### Memory and fixes (always on)

- Game packages are memory-mapped instead of read into the heap: the process had grown to about 3 GB
  and was swapping (`P2_PKG_NO_MMAP=1` restores reading).
- Frame time reports use wall-clock time (frames over 250 ms were reported as 250 ms).
- UTF-16BE `.int` files (some Steam localisations); `P2_TEX_SKIP_MIPS=N` for less texture memory.

## Known limitations

- Dynamic actors are lit by ambient light and one sun.
- A `bStatic` mesh that script hides or destroys after the world is merged stays visible (the Bevy
  path's merging has the same limitation).
- With the render thread, input lags one more frame.
- Not yet verified on the board: performance, the render thread on Panfrost, and the extra fragment
  cost of linear-light fog on fogged pixels.

## Building

You need your own copy of POSTAL 2 (Steam, `POSTAL2Complete`).

**Docker** (cross-builds for ARM and builds for x86):
```
tools/docker/build.sh arm p2game    # with arm-board -> target-docker/aarch64-unknown-linux-gnu/release/p2game
tools/docker/build.sh x86 p2game    # without it     -> target-docker/release/p2game
```
`P2_FEATURES` overrides the feature list for either target.
The `p2-builder` image is built from `tools/docker/Dockerfile` on first use. `P2_PROFILE=dist` builds
with fat LTO (about 3-7% faster logic on x86; slow to build). x86 binaries from the image need glibc
2.39 or newer.

**On the host:** current stable Rust and `libasound2-dev libudev-dev libwayland-dev libxkbcommon-dev`:
```
cargo build --release -p p2game
```

---

# Postal 2 Rust/Bevy

> **Work in progress.** This is an unfinished, unofficial engine port. The whole week (Monday to Friday plus Apocalypse Weekend) can be played start to finish, but expect visual differences, AI oddities, missing effects and bugs compared with the original game. Bug reports are welcome.

A clean-room Rust / [Bevy](https://bevyengine.org) engine that runs **POSTAL 2** (Monday to Friday plus Apocalypse Weekend) using the assets of **your own installed copy** of the game.

## No game files here

- This repository contains **no game assets** (maps, textures, sounds, meshes, scripts) and **no original or decompiled game code**.
- Everything is read at runtime from a legitimately owned install (for example the Steam "POSTAL 2" / POSTAL2Complete edition).
- POSTAL 2 is © Running With Scissors. This project is unaffiliated with them.

## Playing (release build)

1. Download the latest release zip from the [Releases](../../releases) page and unzip it anywhere.
2. Run **`Postal2Rust.exe`** (one exe: the game itself, with a console window for its messages; if it crashes the console stays open with the error).
3. If your **POSTAL 2 install folder** is not found automatically (the one containing `System`, `Maps`, `Textures`; on Steam usually `...\steamapps\common\POSTAL2Complete`), it asks for it. The choice is saved in `postal2dir.txt` next to the exe.

Options:

| Option | What it does |
|---|---|
| `--change-folder` | Ask for the POSTAL 2 folder again |
| `--windowed` | Play in a window instead of fullscreen |
| `--map suburbs-1` | Start directly on a map |

Requirements: Windows 10/11 64-bit, a GPU with Vulkan or DirectX 12, and a copy of POSTAL 2.

## Keybinds

The port reads your bindings from the game's own `User.ini` / `DefUser.ini`, so these are POSTAL 2's defaults and any rebinding in the Controls menu applies.

**Movement**

| Key | Action |
|---|---|
| W / A / S / D | Move forward / strafe left / move back / strafe right |
| Mouse | Look |
| Space | Jump (also skips cutscenes) |
| C or Ctrl | Crouch |
| Shift | Walk |

**Combat**

| Key | Action |
|---|---|
| Left mouse | Fire |
| Right mouse | Alt fire |
| Middle mouse | Reload |
| Mouse wheel | Next / previous weapon (zoom on scoped weapons) |
| 0 to 9 | Weapon slots |
| F | Holster (switch to hands) |
| Q | Kick |
| Z | Throw weapon |

**Items and actions**

| Key | Action |
|---|---|
| E or Enter | Use the selected inventory item / confirm |
| [ / ] | Previous / next inventory item |
| Tab | Inventory menu |
| T | Quick health |
| M | Map |
| X | Throw powerup |
| R | Unzip (piss) |
| B | Talk |
| G | Middle finger |
| H | Dance |
| J | Scream |
| Y | Get down (order people to the ground) |
| K | Suicide |
| L | Toggle inventory hints |

**Game**

| Key | Action |
|---|---|
| Esc | Pause menu |
| F5 / F8 | Quick save / quick load |
| F9 | Screenshot |
| Pause | Pause |
| ~ (tilde) | Console |

**Port extras (not in the original)**

| Key | Action |
|---|---|
| F1 | Print frame timing to the log |
| F3 | Debug overlay (FPS, map, player state, position) |
| F10 | Developer menu (cheats, spawn NPCs, teleport, time of day) |
| F11 | Scene editor (pauses the world; Space resumes, left click selects, right mouse looks) |

## What still needs to be finished or fixed

Known differences from the original, roughly by impact:

**Gameplay and AI**
- NPC pathfinding handles stairs and blocked spots much better, but busy queues (the EastMall food court) can still jostle.
- Some pawns use poses or idle animations that differ from the original.
- Wandering NPCs picking a new random destination now cost about 1 ms instead of 3-14 ms; small hitches can remain right after arriving on a map.

**Rendering**
- Lighting, sky and fog colours still differ in places. Night maps, MilitaryBase cells, VincesHouse glass, EasterArena grass and the suburbs trailer were brought closer to the original in v0.3.0; the MilitaryBase cell mattresses still look like a lattice instead of a solid sheet.
- Explosions and the grenade blast look different (yellow-grey cloud instead of an orange flash with brown smoke); tracers are unchecked.
- A few missing details (e.g. the US flag poster in the police station).

**Interface**
- No known differences left in the map screen; report anything you spot.

**Not started / out of scope**
- Multiplayer.
- POSTAL 2: Paradise Lost (out of scope).
- Linux and macOS builds are produced by CI but have had little testing so far; please report problems.

## Building from source

Windows, Rust stable with the `x86_64-pc-windows-gnullvm` target and [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (no Visual Studio needed). `tools/cargo.bat` wraps cargo with that toolchain; edit the paths at its top for your machine.

```
tools\cargo.bat build --release -p p2game
```

This gives `target\release\p2game.exe`; the release ships it renamed to `Postal2Rust.exe` (under any name other than `p2game` it defaults to fullscreen and asks for the install folder when needed). The app icon (`crates/p2game/res/app.ico`, original art from `tools/release/make_icon.ps1`) is embedded by `build.rs` with llvm-windres. For a shipping build use `--profile dist` instead of `--release` (fat LTO, about 10% faster, slow to build; output in `target\dist`). The game needs `libunwind.dll` from llvm-mingw next to it. You can also run the game directly:

```
target\release\p2game.exe --game "C:\path\to\POSTAL2Complete" --map suburbs-1
```

Regression modes used during development: `--smoke` (boots a set of maps) and `--progress 7` (plays every day's errands headlessly).

## What is in it

| Crate | What |
|---|---|
| `p2game` | The game (shipped as `Postal2Rust.exe`): install detection and folder prompt, boot, map actors, the VM host, AI movement, weapons, HUD, menus, save/load, windowed play |
| `upkg` | Unreal Engine 2 package reader: tables, tagged properties, textures, static/skeletal meshes, animations, BSP, lightmaps, terrain, sounds |
| `uscript` | UnrealScript bytecode interpreter: object graph, states, latent functions, iterators, navigation natives |
| `p2view` | Bevy level renderer: BSP with baked lightmaps, static meshes, terrain, sky, projectors |
| `p2physics` / `p2collide` | UE2 actor physics (walking, falling, swimming, ladders, movers, volumes) and swept collision |
| `p2anim` | Skeletal animation and GPU skinning |
| `p2fx` | Particle emitters and decals |
| `p2sound` / `p2audio` | UE2-style positional audio, music |
| `p2canvas` / `p2input` | UE2 Canvas (HUD, menus, fonts) and input bindings |
| `p2matinee` / `p2movie` / `p2save` | Cutscenes, RoQ movie decoder, save games |
| `p2nav` | Path network loading |

## License

MIT OR Apache-2.0
