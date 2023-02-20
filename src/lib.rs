//! # Particular
//!
//! Particular is a crate providing a simple way to simulate N-body gravitational interaction of particles in Rust.
//!
//! ## Goals
//!
//! The main goal of this crate is to provide users with a simple API to setup N-body gravitational simulations that can easily be integrated into existing game and physics engines.
//! Thus it does not include numerical integration or other similar tools and instead only focuses on the acceleration calculations.
//!
//! Currently, provided [`ComputeMethods`](compute_method::ComputeMethod) are naive and iterate over the particles and sum the acceleration caused by the `massive` particles.
//! I will likely provide algorithms such as e.g. [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) in the future.
//!
//! Particular can be used with a parallel implementation on the CPU thanks to [rayon](https://github.com/rayon-rs/rayon). Enable the "parallel" feature to access the available compute methods.
//!
//! Particular can also be used on the GPU thanks to [wgpu](https://github.com/gfx-rs/wgpu). Enable the "gpu" feature to access the available compute methods.
//!
//! # Using Particular
//!
//! The API to setup a simulation is straightforward:
//!
//! ## Implementing the [`Particle`](particle::Particle) trait
//!
//! #### Deriving:
//!
//! Used in most cases, when the type has fields named `position` and `mu`:
//!
//! ```
//! # use particular::prelude::*;
//! # use glam::Vec3;
//! #
//! #[derive(Particle)]
//! struct Body {
//!     position: Vec3,
//!     mu: f32,
//! //  ...
//! }
//! ```
//! #### Manual implementation:
//!
//! Used when the type has more complex fields and cannot directly provide a [position](Particle::position) and a [gravitational parameter](Particle::mu).
//!
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
//!     type Scalar = f32;
//!     type Vector = Vec3;
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
//!
//! Using the type implementing [`Particle`](particle::Particle), create a [`ParticleSet`](particle_set::ParticleSet) that will contain the particles.
//!
//! [`Particles`](particle::Particle) are stored in two vectors, `massive` or `massless`, depending on if they have mass or not.
//! This allows optimizations for objects that are affected by gravitational bodies but don't affect them back, e.g. a spaceship.
//!
//! ```
//! # use particular::prelude::*;
//! # use glam::Vec3;
//! #
//! # #[derive(Particle)]
//! # struct Body {
//! #     position: Vec3,
//! #     mu: f32,
//! # //  ...
//! # }
//! # let position = Vec3::ONE;
//! # let mu = 1E5;
//! #
//! let mut particle_set = ParticleSet::new();
//! particle_set.add(Body { position, mu });
//! ```
//!
//! ## Computing and using the gravitational acceleration
//!
//! Finally, use the [`result`](particle_set::ParticleSet::result) method of [`ParticleSet`](particle_set::ParticleSet).
//! It returns an iterator over a mutable reference to a [`Particle`](particle::Particle) and its computed gravitational acceleration using the provided [`ComputeMethod`](compute_method::ComputeMethod).
//!
//! ```
//! # use particular::prelude::*;
//! # use glam::Vec3;
//! #
//! # const DT: f32 = 1.0 / 60.0;
//! #
//! # #[derive(Particle)]
//! # struct Body {
//! #     position: Vec3,
//! #     velocity: Vec3,
//! #     mu: f32,
//! # }
//! # let mut particle_set = ParticleSet::<Body>::new();
//! let cm = &mut sequential::BruteForce;
//!
//! for (acceleration, particle) in particle_set.result(cm) {
//!     particle.velocity += acceleration * DT;
//!     particle.position += particle.velocity * DT;
//! }
//! ```

#![warn(missing_docs)]

/// Trait for computing accelerations and types implementing it for the user to choose from.
pub mod compute_method;

/// Trait to implement on types representing particles.
pub mod particle;

/// Storage for the particles.
pub mod particle_set;

/// Internal representation of vectors used for expensive computations.
pub mod vector;

/// Derive macro for types representing particles.
pub mod particular_derive {
    pub use particular_derive::Particle;
}

/// Everything needed to use the crate.
pub mod prelude {
    pub use crate::compute_method::*;
    pub use crate::particle::Particle;
    pub use crate::particle_set::ParticleSet;
    pub use crate::particular_derive::*;
}
