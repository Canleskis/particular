//! Particular is a crate providing a simple way to simulate N-body gravitational interaction of particles in Rust.
//!
//! The API to setup a simulation is straightforward:
//!
//! ## Implementing the [`Particle`] trait.
//!
//! #### Deriving:
//!
//! Used in most cases, when your type has fields named `position` and `mu`.
//! ```
//! # use particular::prelude::Particle;
//! # use glam::Vec3;
//! #
//! #[derive(Particle)]
//! pub struct Body {
//!     position: Vec3,
//!     mu: f32,
//! //  ...
//! }
//! ```
//! #### Manual implementation:
//!
//! Used when your type has more complex fields and cannot directly provide a position and a gravitational parameter.
//! ```
//! # const G: f32 = 1.0;
//! #
//! # use particular::Particle;
//! # use glam::Vec3;
//! #
//! struct Body {
//!     position: Vec3,
//!     mass: f32,
//! //  ...
//! }
//!
//! impl Particle for Body {
//!     fn position(&self) -> Vec3 {
//!         self.position
//!     }
//!     
//!     fn mu(&self) -> f32 {
//!         self.mass * G
//!     }
//! }
//! ```
//! ## Setting up the simulation.
//! Using your type implementing [`Particle`], you will need to create a [`ParticleSet`] that will contain the particles.
//!
//! Currently, it stores the particles in two different vectors depending on if the particle has mass or doesn't. This allows optimizations in the case of massless particles (which can represent objects that do not need to affect other objects, like a spaceship).
//! ```
//! # use particular::prelude::Particle;
//! # use particular::ParticleSet;
//! # use glam::Vec3;
//! #
//! # #[derive(Particle)]
//! # pub struct Body {
//! #     position: Vec3,
//! #     mu: f32,
//! # //  ...
//! # }
//! # let position = Vec3::ONE;
//! # let mu = 1E5;
//! #
//! let mut particle_set = ParticleSet::new();
//! // If the type cannot be inferred, use the turbofish syntax:
//! // let mut particle_set = ParticleSet::<Body>::new();
//!
//! particle_set.add(Body { position, mu });
//! ```
//! ## Computing and using the gravitational acceleration.
//! Finally, using the `result` method of [`ParticleSet`], you can iterate over the computed gravitational acceleration of each particle.
//! ```
//! # use particular::prelude::Particle;
//! # use particular::ParticleSet;
//! # use glam::Vec3;
//! #
//! # const DT: f32 = 1.0 / 60.0;
//! #
//! # #[derive(Particle)]
//! # pub struct Body {
//! #     position: Vec3,
//! #     velocity: Vec3,
//! #     mu: f32,
//! # }
//! # let mut particle_set = ParticleSet::<Body>::new();
//! for (particle, acceleration) in particle_set.result() {
//!     particle.velocity += acceleration * DT;
//!     particle.position += particle.velocity * DT;
//! }
//! ```
//! `particle` here being of the type you used for the [`ParticleSet`] that implements [`Particle`].

mod particle;
mod particle_set;

pub use particle::*;
pub use particle_set::*;

pub mod prelude {
    pub use crate::{Particle, ParticleSet};
    pub use particular_derive::Particle;
}