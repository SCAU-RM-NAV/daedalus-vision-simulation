use avian3d::prelude::{CollisionEventsEnabled, CollisionStart};
use bevy::prelude::{Commands, Entity, On, Plugin, Query, ResMut, Resource, Update, With};
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use super::construct::ArmorHitCollider;
use crate::components::{ProjectileCommandSeq, SimHitEvent, SimHitEventQueue};
use crate::robomaster::power_rune::prelude::Projectile;
use crate::statistic::ProjectileStatistics;

#[derive(Resource, Default)]
struct ConsumedArmorProjectiles(HashSet<Entity>);

fn handle_armor_collision(
    event: On<CollisionStart>,
    mut commands: Commands,
    mut stats: ResMut<ProjectileStatistics>,
    mut consumed_projectiles: ResMut<ConsumedArmorProjectiles>,
    projectiles: Query<Entity, With<Projectile>>,
    projectile_commands: Query<&ProjectileCommandSeq, With<Projectile>>,
    armor_hit_colliders: Query<&ArmorHitCollider>,
    mut hit_events: ResMut<SimHitEventQueue>,
) {
    let projectile_body1 = event.body1.and_then(|body| projectiles.get(body).ok());
    let projectile_body2 = event.body2.and_then(|body| projectiles.get(body).ok());

    let projectile_entity = match (projectile_body1, projectile_body2) {
        (Some(e), _) => e,
        (_, Some(e)) => e,
        _ => return,
    };

    let other_collider = if projectile_body1.is_some() {
        event.collider2
    } else {
        event.collider1
    };

    if let Ok(hit_collider) = armor_hit_colliders.get(other_collider) {
        if !consumed_projectiles.0.insert(projectile_entity) {
            return;
        }
        // Disable collision events for this projectile so it only counts once
        commands
            .entity(projectile_entity)
            .remove::<CollisionEventsEnabled>();
        stats.increase_accurate();
        hit_events.0.push(SimHitEvent {
            command_seq: projectile_commands
                .get(projectile_entity)
                .map(|value| value.0)
                .unwrap_or(0),
            target_id: hit_collider.root.to_bits() as u32,
            blade_id: -1,
            hit_type: 0,
            correct: true,
            outcome: 1,
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|value| value.as_nanos() as u64)
                .unwrap_or(0),
        });
    }
}

fn cleanup_consumed_armor_projectiles(
    mut consumed_projectiles: ResMut<ConsumedArmorProjectiles>,
    projectiles: Query<(), With<Projectile>>,
) {
    consumed_projectiles
        .0
        .retain(|entity| projectiles.contains(*entity));
}

#[derive(Default)]
pub(super) struct ArmorCollisionPlugin;

impl Plugin for ArmorCollisionPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.init_resource::<ConsumedArmorProjectiles>()
            .add_systems(Update, cleanup_consumed_armor_projectiles)
            .add_observer(handle_armor_collision);
    }
}
