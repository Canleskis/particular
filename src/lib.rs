mod particle;
mod particle_set;

pub use particle::*;
pub use particle_set::*;

pub mod prelude {
    pub use crate::{Particle, ParticleSet};
    pub use particular_derive::Particle;
}

#[cfg(test)]
mod tests {
    // TODO!
}
