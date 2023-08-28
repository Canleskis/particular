#[derive(Clone, Copy)]
pub enum Spawnable {
    Massive {
        min_mass: f32,
        max_mass: f32,
        density: f32,
    },
    Massless {
        density: f32,
    },
}

impl Spawnable {
    pub fn min_mass(self) -> Option<f32> {
        match self {
            Spawnable::Massive { min_mass, .. } => Some(min_mass),
            Spawnable::Massless { .. } => None,
        }
    }

    pub fn max_mass(self) -> Option<f32> {
        match self {
            Spawnable::Massive { max_mass, .. } => Some(max_mass),
            Spawnable::Massless { .. } => None,
        }
    }

    pub fn mass_range(self) -> Option<(f32, f32)> {
        match self {
            Spawnable::Massive {
                min_mass, max_mass, ..
            } => Some((min_mass, max_mass)),
            Spawnable::Massless { .. } => None,
        }
    }

    pub fn density(self) -> f32 {
        match self {
            Spawnable::Massive { density, .. } => density,
            Spawnable::Massless { density } => density,
        }
    }

    pub fn is_massive(&self) -> bool {
        matches!(self, Spawnable::Massive { .. })
    }

    pub fn is_massless(&self) -> bool {
        matches!(self, Spawnable::Massless { .. })
    }
}
