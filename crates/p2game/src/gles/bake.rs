//! The static world: BSP, terrain and the static mesh actors script no longer changes are merged
//! into p2gles' own buffers; their Bevy entities are then neither drawn nor driven.
//!
//! A level load calls [`queue_bake`]. [`classify_baked`] runs right after it and marks every
//! candidate entity, so the generic pass never uploads them. [`bake_world`] runs a couple of script
//! frames later and decides which actors really are static.

use super::Gles;
use crate::window::Sim;
use bevy::camera::visibility::{RenderLayers, VisibilityRange};
use bevy::prelude::*;
use p2view::scene::SceneData;
use std::collections::{HashMap, HashSet};
use uscript::vm::ObjId;

/// Marks an entity whose geometry is (or is about to be, until [`bake_world`] runs) in the world
/// buffers: the generic pass does not even visit it.
#[derive(Component)]
pub(crate) struct GlStatic;

/// What a level load hands over for baking.
pub(crate) struct LevelBake {
    /// The scene as loaded, terrain included.
    pub data: SceneData,
    /// Entities that existed before this level was spawned.
    pub before: HashSet<Entity>,
    /// Map export index -> its static mesh entities.
    pub actors: HashMap<usize, Vec<Entity>>,
    /// Map export index -> script actor.
    pub by_export: HashMap<usize, ObjId>,
    /// Exports of movers (doors, lifts): always dynamic.
    pub movers: HashSet<usize>,
}

/// The level waiting for its bake.
pub(crate) struct PendingBake {
    level: LevelBake,
    classified: bool,
    /// Script frame to bake at, set on the first run after loading.
    frame: Option<u64>,
}

/// Replaces a bake still pending from the previous level.
pub(crate) fn queue_bake(commands: &mut Commands, level: LevelBake) {
    commands.queue(move |w: &mut World| w.insert_non_send(Some(PendingBake { level, classified: false, frame: None })));
}

/// Runs right after `load_scene` in the Update chain, before anything else spawns meshes. Candidates:
/// BSP, terrain layers and static mesh instances on the default layer. Not candidates: movers, the
/// sky zone (own layer) and terrain decorations (distance-ranged entities that are no map actor).
#[allow(clippy::type_complexity)]
pub(crate) fn classify_baked(mut commands: Commands, pending: Option<NonSendMut<Option<PendingBake>>>, meshes: Query<(Entity, Option<&RenderLayers>, Has<VisibilityRange>), With<Mesh3d>>) {
    // A bake is only ever queued with P2GAME_GLES=1.
    let Some(mut pending) = pending else { return };
    let Some(p) = pending.as_mut() else { return };
    if std::mem::replace(&mut p.classified, true) {
        return;
    }
    let level = &p.level;
    let statics: HashSet<Entity> = level.actors.values().flatten().copied().collect();
    let movers: HashSet<Entity> = level.actors.iter().filter(|(export, _)| level.movers.contains(export)).flat_map(|(_, e)| e.iter().copied()).collect();
    for (e, layers, ranged) in &meshes {
        if !level.before.contains(&e) && !movers.contains(&e) && layers.is_none_or(|l| l.intersects(&RenderLayers::layer(0))) && (!ranged || statics.contains(&e)) {
            commands.entity(e).try_insert(GlStatic);
        }
    }
}

/// Bakes the level once level-start script has run, with the same rule as `merge_statics`:
/// PrepActorsByGroup destroys other days' and holidays' grouped meshes on the first ticks, so only
/// then is it known which actors stay. Baked: alive, visible, bStatic, not a mover. Every other
/// actor stays an ordinary entity under `sync_transforms`. The two frames are counted from this
/// level's load: `Sim::frames` runs across levels.
pub(crate) fn bake_world(mut commands: Commands, mut sim: NonSendMut<Sim>, pending: Option<NonSendMut<Option<PendingBake>>>, gles: Option<NonSendMut<Gles>>) {
    // Both exist only with P2GAME_GLES=1.
    let (Some(mut pending), Some(mut gles)) = (pending, gles) else { return };
    let Some(p) = pending.as_mut() else { return };
    if sim.frames < *p.frame.get_or_insert(sim.frames + 2) {
        return;
    }
    let Some(PendingBake { level, .. }) = pending.take() else { return };
    let vm = &sim.game.world.vm;
    let flag = |o: ObjId, name: &str| vm.get_prop(o, name).is_some_and(|v| v.as_bool());
    let keep = |export: usize| !level.movers.contains(&export) && level.by_export.get(&export).is_some_and(|&o| !vm.objects[o].deleted && flag(o, "bStatic") && !flag(o, "bHidden"));
    let skip: HashSet<usize> = level.data.instances.iter().map(|i| i.export).filter(|&e| !keep(e)).collect();
    let mut stay_baked = HashSet::new();
    let mut dynamic = 0;
    for (export, entities) in &level.actors {
        if skip.contains(export) {
            dynamic += entities.len();
            for &e in entities {
                if let Ok(mut entity) = commands.get_entity(e) {
                    entity.try_remove::<GlStatic>();
                }
            }
        } else {
            stay_baked.extend(entities.iter().copied());
        }
    }
    sim.stop_driving(&stay_baked);
    gles.load_world(level.data, skip);
    println!("gles: {} static mesh entities baked, {dynamic} stay dynamic", stay_baked.len());
}
