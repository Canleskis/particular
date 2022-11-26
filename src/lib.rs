//! # Particular
//!
//! Particular is a crate providing a simple way to simulate N-body gravitational interaction of particles in Rust.
//!
//! ## Goals
//! The main goal of this crate is to provide users with a simple API to setup N-body gravitational simulations that can easily be integrated into existing game and physics engines.
//! Thus it does not include numerical integration or other similar tools and instead only focuses on the acceleration calculations.
//!
//! Currently, acceleration calculations are computed naively by iterating over all the particles and summing the acceleration caused by all the `massive` particles.
//! In the future, I would like to implement other algorithms such as [Barnes-Hut algorithm](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) or even use compute shaders on the GPU for faster calculations.
//!
//! Particular can be used with a parallel implementation on the CPU thanks to the [rayon](https://github.com/rayon-rs/rayon) crate. Use the "parallel" feature to enable it, which can lead to huge performance improvements.
//!
//! # Using Particular
//!
//! The API to setup a simulation is straightforward:
//!
//! ## Implementing the [`Particle`](particle::Particle) trait
//!
//! #### Attribute macro or deriving:
//!
//! Used in most cases, when the type has fields named `position` and `mu`:
//! ```
//! # use particular::prelude::*;
//! # use glam::Vec3;
//! #
//! #[particle(3)]
//! pub struct Body {
//!     position: Vec3,
//!     mu: f32,
//! //  ...
//! }
//! ```
//! This is equivalent to:
//!
//! ```
//! # use particular::prelude::*;
//! # use glam::Vec3;
//! #
//! #[derive(Particle)]
//! #[dim(3)]
//! pub struct Body {
//!     position: Vec3,
//!     mu: f32,
//! //  ...
//! }
//! ```
//! #### Manual implementation:
//!
//! Used when the type has more complex fields and cannot directly provide a position and a gravitational parameter.
//! ```
//! # const G: f32 = 1.0;
//! #
//! # use particular::prelude::*;
//! # use glam::Vec3;
//! #
//! struct Body {
//!     position: Vec3,
//!     mass: f32,
//! //  ...
//! }
//!
//! impl Particle for Body {
//!     type Vector = VectorDescriptor<3, Vec3>;
//!
//!     fn position(&self) -> Vec3 {
//!         self.position
//!     }
//!     
//!     fn mu(&self) -> f32 {
//!         self.mass * G
//!     }
//! }
//! ```
//! ## Setting up the simulation
//! Using the type implementing [`Particle`](particle::Particle), create a [`ParticleSet`](particle_set::ParticleSet) that will contain the particles.
//!
//! Particles are stored in two vectors, `massive` or `massless`, depending on if they have mass or not.
//! This allows optimizations in the case of massless particles (which represents objects that do not need to affect other objects, like a spaceship).
//! ```
//! # use particular::prelude::*;
//! # use glam::Vec3;
//! #
//! # #[particle(3)]
//! # pub struct Body {
//! #     position: Vec3,
//! #     mu: f32,
//! # //  ...
//! # }
//! # let position = Vec3::ONE;
//! # let mu = 1E5;
//! #
//! // If the type cannot be inferred, use the turbofish syntax:
//! let mut particle_set = ParticleSet::<Body>::new();
//! // Otherwise:
//! let mut particle_set = ParticleSet::new();
//!
//! particle_set.add(Body { position, mu });
//! ```
//! ## Computing and using the gravitational acceleration
//! Finally, use the [`result`](particle_set::ParticleSet::result) method of [`ParticleSet`](particle_set::ParticleSet), which returns an iterator over a mutable reference to a `Particle` and its computed gravitational acceleration.
//! ```
//! # use particular::prelude::*;
//! # use glam::Vec3;
//! #
//! # const DT: f32 = 1.0 / 60.0;
//! #
//! # #[particle(3)]
//! # pub struct Body {
//! #     position: Vec3,
//! #     velocity: Vec3,
//! #     mu: f32,
//! # }
//! # let mut particle_set = ParticleSet::<Body>::new();
//! for (acceleration, particle) in particle_set.result() {
//!     particle.velocity += acceleration * DT;
//!     particle.position += particle.velocity * DT;
//! }
//! ```

pub mod particle;
pub mod particle_set;
pub mod vector;

pub mod particular_derive {
    pub use crate::vector::VectorDescriptor;
    pub use particular_derive::{particle, Particle};
}

pub mod prelude {
    pub use crate::particle::Particle;
    pub use crate::particle_set::ParticleSet;
    pub use crate::particular_derive::*;
    pub use crate::vector::{Descriptor, VectorDescriptor};
}
