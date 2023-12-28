//! # Particular
//!
//! Particular is a crate providing a simple way to simulate N-body gravitational interaction of particles in Rust.
//!
//! ## Goals
//!
//! The main goal of this crate is to provide users with a simple API to set up N-body gravitational simulations that can easily be integrated into existing game and physics engines.
//! Thus it does not concern itself with numerical integration or other similar tools and instead only focuses on the acceleration calculations.
//!
//! Particular is also built with performance in mind and provides multiple ways of computing the acceleration between particles.
//!
//! ### Computation algorithms
//!
//! There are currently 2 algorithms used by the available compute methods: [BruteForce](https://en.wikipedia.org/wiki/N-body_problem#Simulation) and [BarnesHut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation).
//!
//! Generally speaking, the BruteForce algorithm is more accurate, but slower. The BarnesHut algorithm allows trading accuracy for speed by increasing the `theta` parameter.  
//! You can read more about their relative performance [here](#notes-on-performance).
//!
//! Particular uses [rayon](https://github.com/rayon-rs/rayon) for parallelization. Enable the "parallel" feature to access the available compute methods.  
//! Particular uses [wgpu](https://github.com/gfx-rs/wgpu) for GPU computation. Enable the "gpu" feature to access the available compute methods.
//!
//! ## Using Particular
//!
//! ### Implementing the [`Particle`](particle::Particle) trait
//!
//! When possible, it can be useful to implement [`Particle`](particle::Particle) on a type.
//!
//! #### Deriving:
//!
//! Used when the type has fields named `position` and `mu`:
//!
//! ```
//! # use particular::prelude::*;
//! # use ultraviolet::Vec3;
//! #
//! #[derive(Particle)]
//! #[dim(3)]
//! struct Body {
//!     position: Vec3,
//!     mu: f32,
//! //  ...
//! }
//! ```
//! #### Manual implementation:
//!
//! Used when the type cannot directly provide a position and a gravitational parameter.
//!
//! ```
//! # const G: f32 = 1.0;
//! #
//! # use particular::prelude::*;
//! # use ultraviolet::Vec3;
//! #
//! struct Body {
//!     position: Vec3,
//!     mass: f32,
//! //  ...
//! }
//!
//! impl Particle for Body {
//!     type Array = [f32; 3];
//!
//!     fn position(&self) -> [f32; 3] {
//!         self.position.into()
//!     }
//!     
//!     fn mu(&self) -> f32 {
//!         self.mass * G
//!     }
//! }
//! ```
//!
//! If you can't implement [`Particle`] on a type, you can use the fact that it is implemented for tuples of an array and its scalar type instead of creating an intermediate type.
//!
//! ```
//! # use particular::prelude::*;
//! let particle = ([1.0, 1.0, 0.0], 5.0);
//!
//! assert_eq!(particle.position(), [1.0, 1.0, 0.0]);
//! assert_eq!(particle.mu(), 5.0);
//! ```
//!
//! ### Computing and using the gravitational acceleration
//!
//! In order to compute the accelerations of your particles, you can use the [accelerations](particle::Accelerations::accelerations) method on iterators,
//! passing in a mutable reference to a [`ComputeMethod`](compute_method::ComputeMethod) of your choice.
//!
//! ### Examples
//!
//! #### When the iterated type doesn't implement [`Particle`](particle::Particle)
//! ```
//! # use particular::prelude::*;
//! # use ultraviolet::Vec3;
//! #
//! # const DT: f32 = 1.0 / 60.0;
//! # const G: f32 = 1.0;
//! # let mut cm = sequential::BruteForceScalar;
//! #
//! # let mut items = vec![
//! #     (Vec3::zero(), -Vec3::one(), 5.0),
//! #     (Vec3::zero(), Vec3::zero(), 3.0),
//! #     (Vec3::zero(), Vec3::one(), 10.0),
//! # ];
//! // Items are a tuple of a velocity, a position and a mass.
//! // We map them to a tuple of the positions as an array and the mu, since this implements `Particle`.
//! let accelerations = items
//!     .iter()
//!     .map(|(_, position, mass)| (*position.as_array(), *mass * G))
//!     .accelerations(&mut cm);
//!
//! for (acceleration, (velocity, position, _)) in accelerations.zip(&mut items) {
//!     *velocity += Vec3::from(acceleration) * DT;
//!     *position += *velocity * DT;
//! }
//! ```
//! #### When the iterated type implements [`Particle`](particle::Particle)
//! ```
//! # use particular::prelude::*;
//! # use ultraviolet::Vec3;
//! #
//! # const DT: f32 = 1.0 / 60.0;
//! # let mut cm = sequential::BruteForceScalar;
//! #
//! # #[derive(Particle)]
//! #[dim(3)]
//! # struct Body {
//! #     position: Vec3,
//! #     velocity: Vec3,
//! #     mu: f32,
//! # }
//! # let mut bodies = Vec::<Body>::new();
//! for (acceleration, body) in bodies.iter().accelerations(&mut cm).zip(&mut bodies) {
//!     body.velocity += Vec3::from(acceleration) * DT;
//!     body.position += body.velocity * DT;
//! }
//! ```
//!
//! ## Notes on performance
//!
//! A comparison between 7 available compute methods using an i9 9900KF and an RTX 3080 is available in the [README](https://crates.io/crates/particular).
//!
//! Depending on your needs and platform, you may opt for one compute method or another.
//! You can also implement the trait on your own type to use other algorithms or combine multiple compute methods and switch between them depending on certain conditions (e.g. the particle count).

#![warn(missing_docs)]

/// Implementation of algorithms to compute the acceleration of particles.
pub mod compute_method;
/// Traits for particle representation of objects and computing their acceleration.
pub mod particle;
/// Derive macro for the [`Particle`](crate::particle::Particle) trait.
pub mod particular_derive {
    pub use particular_derive::Particle;
}

pub use compute_method::*;

/// Most commonly used re-exported types.
pub mod prelude {
    #[doc(hidden)]
    pub use crate::{
        compute_method::*,
        particle::{Accelerations, IntoPointMass, Particle},
        particular_derive::Particle,
    };
}
