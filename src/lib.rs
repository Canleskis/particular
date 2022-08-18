mod particle;
mod particle_set;

pub use particle_set::*;
pub use particle::*;

pub mod prelude {
    pub use crate::{ParticleSet, Particle};
    pub use particular_derive::Particle;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
