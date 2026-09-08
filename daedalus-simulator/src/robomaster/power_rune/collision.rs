use crate::components::{ProjectileCommandSeq, SimHitEvent, SimHitEventQueue};
use crate::robomaster::power_rune::common::RuneHitOutcome;
use crate::robomaster::power_rune::rotation::PowerRuneRotation;
use crate::robomaster::power_rune::rune::PowerRuneMechanism;
use avian3d::prelude::{CollisionEnd, CollisionEventsEnabled};
use bevy::prelude::{
    ChildOf, Commands, Component, Entity, EntityEvent, On, Query, ResMut, Resource, Update, With,
};
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Component)]
#[require(CollisionEventsEnabled)]
pub struct Projectile;

#[derive(Resource, Default)]
struct ConsumedRuneProjectiles(HashSet<Entity>);

#[derive(Component, Debug, Copy, Clone)]
pub struct RuneIndex {
    pub target: usize,
    pub rune: Entity,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct HitResult {
    pub outcome: RuneHitOutcome,
}

impl HitResult {
    pub const fn accurate(self) -> bool {
        self.outcome.is_accurate()
    }
}

#[derive(EntityEvent)]
pub struct RuneActivated {
    #[event_target]
    pub rune: Entity,
}

#[derive(EntityEvent)]
pub struct RuneHit {
    #[event_target]
    pub rune: Entity,
    pub result: HitResult,
}

fn handle_rune_collision(
    event: On<CollisionEnd>,
    mut commands: Commands,
    mut consumed_projectiles: ResMut<ConsumedRuneProjectiles>,
    mut runes: Query<(&mut PowerRuneMechanism, &mut PowerRuneRotation)>,
    targets: Query<&RuneIndex>,
    projectiles: Query<Entity, With<Projectile>>,
    projectile_commands: Query<&ProjectileCommandSeq, With<Projectile>>,
    child_of: Query<&ChildOf>,
    mut hit_events: ResMut<SimHitEventQueue>,
) {
    let projectile_body1 = event.body1.and_then(|body| projectiles.get(body).ok());
    let projectile_body2 = event.body2.and_then(|body| projectiles.get(body).ok());

    let (projectile_entity, target_collider) = match (projectile_body1, projectile_body2) {
        (Some(projectile), _) => (projectile, event.collider2),
        (_, Some(projectile)) => (projectile, event.collider1),
        _ => return,
    };

    let Some(target) = find_rune_target(target_collider, &targets, &child_of) else {
        return;
    };

    let Ok((mut mechanism, mut rotation)) = runes.get_mut(target.rune) else {
        return;
    };

    if !consumed_projectiles.0.insert(projectile_entity) {
        return;
    }

    commands
        .entity(projectile_entity)
        .remove::<CollisionEventsEnabled>();

    let mut rng = rand::rng();
    let outcome = mechanism.state_mut().hit(target.target, &mut rng);
    rotation.sync_activation(
        mechanism.state().mode(),
        mechanism.state().is_activating(),
        &mut rng,
    );

    commands.trigger(RuneHit {
        rune: target.rune,
        result: HitResult { outcome },
    });

    hit_events.0.push(SimHitEvent {
        command_seq: projectile_commands
            .get(projectile_entity)
            .map(|value| value.0)
            .unwrap_or(0),
        target_id: target.rune.to_bits() as u32,
        blade_id: target.target as i32,
        hit_type: 1,
        correct: outcome.is_accurate(),
        outcome: outcome as u8,
        timestamp_ns: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos() as u64)
            .unwrap_or(0),
    });

    if outcome.activates_rune() {
        commands.trigger(RuneActivated { rune: target.rune });
    }
}

fn cleanup_consumed_rune_projectiles(
    mut consumed_projectiles: ResMut<ConsumedRuneProjectiles>,
    projectiles: Query<(), With<Projectile>>,
) {
    consumed_projectiles
        .0
        .retain(|entity| projectiles.contains(*entity));
}

fn find_rune_target(
    entity: Entity,
    targets: &Query<&RuneIndex>,
    child_of: &Query<&ChildOf>,
) -> Option<RuneIndex> {
    if let Ok(target) = targets.get(entity) {
        return Some(*target);
    }

    child_of
        .iter_ancestors(entity)
        .find_map(|ancestor| targets.get(ancestor).ok().copied())
}

#[derive(Default)]
pub(super) struct PowerRuneCollisionPlugin;

impl bevy::app::Plugin for PowerRuneCollisionPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.init_resource::<ConsumedRuneProjectiles>()
            .add_systems(Update, cleanup_consumed_rune_projectiles)
            .add_observer(handle_rune_collision);
    }
}
