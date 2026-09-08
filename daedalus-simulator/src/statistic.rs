use bevy::prelude::*;

#[derive(Resource, Default, Reflect)]
#[reflect(Resource)]
pub struct ProjectileStatistics {
    pub launch_count: u32,
    pub accurate_count: u32,
}

impl ProjectileStatistics {
    pub fn reset(&mut self) {
        self.launch_count = 0;
        self.accurate_count = 0;
    }

    pub fn increase_launch(&mut self) {
        self.launch_count += 1;
    }

    pub fn increase_accurate(&mut self) {
        if self.accurate_count < self.launch_count {
            self.accurate_count += 1;
        }
    }

    pub fn accurate_pct(&self) -> f32 {
        if self.launch_count == 0 {
            return 0.0;
        }
        (self.accurate_count as f32) / (self.launch_count as f32)
    }
}
