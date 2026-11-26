# Форк zaman-ua: p2gles и работа на слабом ARM (Mali-G31)

Ветка `p2gles` этого форка — эксперимент: заставить POSTAL 2 из этого порта работать на ТВ-приставке
(Amlogic S905X3: 4× Cortex-A55, GPU **Mali-G31 MC1**, Debian trixie, Mesa/Panfrost, OpenGL ES 3.1).
Оригинальный README автора — ниже, без изменений. Ветка `main` совпадает с исходным репозиторием.

## Что здесь сделано

**1. Патчи порта для Mali / GLES** (коммиты `x96plus: ...`):
- wgpu с бэкендом GLES и EGL на X11, WebGL2-путь Bevy на нативе (`patches/`, через `[patch.crates-io]`) —
  без них на Panfrost Bevy не стартует (panvk v7 он не принимает);
- пакеты игры читаются через mmap вместо копии в куче: процесс занимал ~3 ГБ и уходил в своп,
  теперь своп 0 (`P2_PKG_NO_MMAP=1` — по-старому);
- MSAA выключен на всех камерах (`P2_MSAA=1` — вернуть), дальняя плоскость камеры = конец тумана;
- отчёт `frame ... ms` считает по реальному времени (раньше любой кадр > 250 мс показывался как 250);
- `.int` в UTF-16BE (русская Steam-версия), `P2_TEX_SKIP_MIPS=N`.

Итог на приставке: Bevy-версия ~3.5 fps. Замеры показали, что упор не в шейдеры и не в пиксели,
а в накладные расходы Bevy в режиме совместимости (~40 мс пустого кадра), ~2400 отдельных вызовов
отрисовки и логику игры (~30 мс). Подробности — в истории коммитов.

**2. `crates/p2gles` — тонкий рендерер мира на OpenGL ES 3.0** (glow + glutin/EGL + winit):
- карта собирается тем же `p2view::scene::build_scene`, рисуется в стиле UE2: текстура × лайтмапа (BSP),
  текстура × запечённый цвет вершин (статик-меши), слои ландшафта по альфа-картам, туман;
- вся статика в одном вершинном буфере, нарезана на ячейки по 48 м, отсечение по пирамиде и туману:
  ~200 вызовов отрисовки вместо ~2400;
- DXT-текстуры уходят в GPU без распаковки (видеопамять 636 → 256 МБ);
- **только мир**: нет людей, логики, HUD и неба (фон — цвет тумана). Это этап 0 — проверка, стоит ли
  заменять рендер Bevy своим.

| Mali-G31 MC1, 800x480 | Mesa 25.0 | Mesa 26.1 |
|---|---|---|
| p2gles, suburbs-3 (спавн) | 57 мс (17 fps) | **10.5 мс (95 fps)** |
| p2gles, estates (спавн) | 47 мс (21 fps) | **8 мс (125 fps)** |
| Bevy-порт, suburbs-3 (вся игра) | ~285 мс | ~282 мс |
| Оригинал (i386) через box64 | 17 fps | 17–19 fps |

Mesa 26 для Debian trixie есть в `trixie-backports`.

**Ловушка Panfrost**: если много draw-вызовов берут индексы из одного общего индексного буфера,
драйвер каждый кадр заново сканирует индексы на CPU (`u_vbuf_get_minmax_index`, его кэш min/max на
буфер маленький; `glDrawRangeElements` не помогает). В p2gles у каждого вызова свой индексный буфер
(`P2GLES_ONE_IBO=1` — общий, для сравнения).

## Сборка

Нужна своя копия POSTAL 2 (Steam, `POSTAL2Complete`).

**В Docker (рекомендуется, кросс-сборка под ARM и сборка под x86):**
```
tools/docker/build.sh arm p2gles    # -> target-docker/aarch64-unknown-linux-gnu/release/p2gles
tools/docker/build.sh x86 p2gles    # -> target-docker/release/p2gles
```
Образ `p2-builder` собирается из `tools/docker/Dockerfile` при первом запуске (или `docker load -i p2-builder.tar`).
Бинарник x86 из образа собран на glibc 2.41 (Debian trixie).

**На хосте:** Rust stable (≥ 1.95 для Bevy 0.19), пакеты `libasound2-dev libudev-dev libwayland-dev libxkbcommon-dev`:
```
cargo build --release -p p2gles
```

## Запуск p2gles

```
p2gles --game <папка POSTAL2Complete> --map suburbs-3 [--res 800x480] [--fullscreen] [--vsync]
p2gles --map estates --bench 20          # замер: мс/кадр, fps, вызовы, треугольники каждые 5 с
p2gles --map suburbs-3 --screenshot out.png
```
Управление: WASD — движение, мышь — обзор, Q/E — вниз/вверх, Shift — быстрее, Esc — выход.
`--game` по умолчанию — `~/.local/share/Steam/steamapps/common/POSTAL2Complete`.
Ещё: `--cam x,y,z,yaw,pitch` (юниты Unreal), `--fog start,end`, `--chunk M`, `--no-cull`;
переменные `P2GLES_NO_BC`, `P2GLES_ANISO=N`, `P2GLES_SKIP=opaque,masked,splat,blend`.
Отчёт первого этапа и скриншоты: `crates/p2gles/REPORT.md`, `crates/p2gles/shots/`.

## Игра на p2gles (`P2GAME_GLES=1`)

```
P2GAME_GLES=1 p2game --map suburbs-3        # Bevy без рендера, кадр рисует p2gles
```
Bevy остаётся для логики (скрипты, физика, ИИ, звук, ECS), но рендер Bevy не создаётся вообще
(`WgpuSettings::backends = None`: нет RenderApp, extract/queue/prepare). GL-контекст (EGL, GLES 3.0)
создаётся на окне Bevy, всё рисуется в `Last` (`crates/p2game/src/gles/`: `bake` — статичный мир,
`collect` — чтение ECS в списки отрисовки, `assets` — копии мешей и текстур в GPU, `frame` — кадр,
`backend` — всё, что вызывает GL, `view` — что видит камера):

| что | как |
|---|---|
| BSP, статик-меши, ландшафт | запекаются один раз из `SceneData` (`p2gles::renderer`) через два тика скрипта после загрузки уровня (`bake_world`): к этому времени скрипт уже уничтожил меши других дней и праздников. Запекаются живые, видимые `bStatic` не-movers (то же правило, что в `merge_statics`); их сущности Bevy не рисуются |
| двери и подвижные объекты | общий проход (`p2gles::dynamic`), сущности Bevy |
| персонажи, оружие от первого лица, подбираемые предметы | общий проход: скиннинг в вершинном шейдере (до 64 костей), слои камер (`RenderLayers`) |
| небо | камера неба (порядок -1) → тот же общий проход, потом очистка глубины и мир |
| декорации (трава, камни) | общий проход, инстансинг (один draw на группу) |
| частицы и декали p2fx | отдельный шейдер со стилями отрисовки UE2 |
| HUD и меню p2canvas | порт `canvas.wgsl` на GLSL ES 3.0 |

### Ядра и разрешение

| переменная | что делает |
|---|---|
| (по умолчанию) | GL на отдельном потоке `p2gles-render`: пока кадр N уходит в GPU, игра считает кадр N+1. В полёте не больше одного кадра, задержка ввода больше на кадр |
| `P2GAME_GLES_THREAD=0` | всё на главном потоке, как раньше (для сравнения) |
| `P2GAME_GLES_SCALE=0.75` | 3D рисуется в доле разрешения окна и растягивается, HUD — в родном. 0.75 — 56% пикселей, 0.5 — четверть |

Персонажи вне кадра и за туманом не позируются и не рисуются (`P2GAME_NO_POSE_CULL=1` — позировать
всех, для проверки, если кто-то пропадает).

Как читать строку `gles:` в логе: `main thread waited` заметно больше нуля — кадр упирается в GPU (поможет
`P2GAME_GLES_SCALE`, ядра не помогут); около нуля — упор в процессор. Загрузку по ядрам показывает
`top -H -p $(pidof p2game)`: общие 30% на четырёх ядрах не отличают одно занятое ядро от ожидания GPU.

Известные пробелы: динамический свет на персонажах — ambient + одно солнце; статик-меш, который скрипт
спрячет или уничтожит позже (при том что он `bStatic`), останется виден — как и при слиянии в Bevy-пути.
Проверено с файлами игры на программном GLES (Xvfb): `--smoke --windowed` проходит все этапы, меню,
ходьба и пауза управляются настоящим вводом. На Mali эта версия ещё не проверялась.

## Дальше

Замер на приставке, профилирование логики.

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
