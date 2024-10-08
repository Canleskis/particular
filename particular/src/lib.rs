#![warn(missing_docs)]
//! # Particular
//!
//! Particular is a crate providing a simple way to compute N-body interaction of particles in
//! Rust.
//!
//! ## Goals
//!
//! The main goal of this crate is to provide users with a simple API to set up N-body simulations
//! that can easily be integrated into existing game and physics engines. Thus it does not include
//! anything related to numerical integration or other similar tools and instead only focuses on the
//! calculations of the interactions between particles.
//!
//! In order to do this, [`particular`](crate) provides multiple algorithms to compute the
//! interactions between particles, allowing users to choose the one that best fits their needs in
//! terms of performance and accuracy.
//!
//! ### Computation algorithms
//!
//! There are currently 2 algorithms:
//! [Brute-force](https://en.wikipedia.org/wiki/N-body_problem#Simulation) and
//! [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation).
//!
//! Generally speaking, the Brute-force algorithm is more accurate, but slower. The Barnes-Hut
//! algorithm allows trading accuracy for speed by increasing the `theta` parameter.  
//! You can see more about their relative performance [here](https://particular.rs/benchmarks/).
//!
//! Particular uses [rayon](https://github.com/rayon-rs/rayon) for parallelization and
//! [wgpu](https://github.com/gfx-rs/wgpu) for GPU computation.  
//! Enable the respective `parallel` and `gpu` features to access the relevant algorithms.
//!
//! ## Using Particular
//!
//! At its core, Particular is a simple set of traits designed to simplify the implementation of
//! different algorithms to compute interactions between particles. These interactions can be of any
//! nature, but a common example is the gravitational interaction between particles, which is
//! provided by the [`gravity`] module and the main focus of the library.  
//! This module provides implementations using popular vector math libraries such as `glam` and
//! `nalgebra`. You you will need to enable the corresponding features to use them.
//!
//! ### Getting started
//!
//! The [`Position`] and [`Mass`] traits provide methods to retrieve the position, mass and
//! gravitational parameter of a particle. Implementing these traits for a type will result in
//! multiple blanket implementations that will allow it to be used with all available algorithms for
//! computing different interactions.
//!
//! #### Implementing the [`Position`] and [`Mass`] trait
//!
//! When the type has fields named `position` and `mass` or `mu`, you can derive these traits. An
//! optional attribute `#[G = ...]` defining  the gravitational constant can be added on the field.
//! If the attribute is missing, the value of the gravitational constant is `1.0`.
//!
//! ```
//! use particular::prelude::*;
//! use glam::DVec3;
//!
//! #[derive(Position, Mass)]
//! struct Planet {
//!     velocity: DVec3,
//!     position: DVec3,
//!     #[G = 6.67430e-11]
//!     mass: f64,
//! }
//! ```
//!
//! If you can't implement these traits, you can use the fact that it is implemented for tuples of a
//! position and a mass or gravitational parameter instead of creating an intermediate type.
//!
//! ```
//! # use particular::prelude::*;
//! # use glam::Vec3;
//! let particle = (Vec3::new(-1.0, 0.0, 1.0), 5.0);
//!
//! assert_eq!(particle.position(), Vec3::new(-1.0, 0.0, 1.0));
//! assert_eq!(particle.mass(), 5.0);
//! assert_eq!(particle.mu(), 5.0);
//! ```
//!
//! #### Computing and using the gravitational interaction
//!
//! In order to compute the gravitational interaction between particles, you can use the various
//! methods implemented on storages of particles.  
//! Note that most algorithms return iterators that borrow the particles, and as such, rust's
//! borrowing rules will prevent you from mutating these particles as you iterate the computed
//! interactions, even if the interaction does not use the fields you are mutating. You can either
//! immediately collect the interactions or use separate collections for the properties you need to
//! mutate.
//!
//! ```
//! use particular::prelude::*;
//! use particular::gravity::newtonian::Acceleration;
//!
//! # use glam::Vec3;
//! # const DT: f32 = 1.0 / 60.0;
//! # #[derive(Position, Mass)]
//! # struct Planet {
//! #     position: Vec3,
//! #     velocity: Vec3,
//! #     mu: f32,
//! # }
//! # let mut planets = Vec::<Planet>::new();
//! let accelerations: Vec<_> = planets.brute_force(Acceleration::checked()).collect::<Vec<_>>();
//!
//! for (planet, acceleration) in planets.iter_mut().zip(accelerations) {
//!     planet.velocity += acceleration * DT;
//!     planet.position += planet.velocity * DT;
//! }
//! ```
//!
//! You can alternatively use indices, but note that this requires making sure the positions don't
//! change as the interaction is computed or you will get unexpected behaviour.
//!
//! ```
//! use particular::prelude::*;
//! use particular::gravity::newtonian::AccelerationSoftened;
//!
//! # use glam::Vec3;
//! # const DT: f32 = 1.0 / 60.0;
//! # #[derive(Position, Mass)]
//! # struct Planet {
//! #     position: Vec3,
//! #     velocity: Vec3,
//! #     mu: f32,
//! # }
//! # let mut planets = Vec::<Planet>::new();
//! for i in 0..planets.len() {
//!     let between = Between(&planets[i], planets.as_slice());
//!     let acceleration = between.brute_force(AccelerationSoftened::checked(1e-3));
//!     planets[i].velocity += acceleration * DT;
//! }
//!
//! for planet in planets.iter_mut() {
//!     planet.position += planet.velocity * DT;
//! }
//! ```
//!
//! <details>
//! <summary><h4>Advanced usage</h4></summary>
//!
//! #### Storages and built-in [`Interaction`] implementations
//!
//! Storages are containers that make it easy to apply certain optimisation or algorithms on
//! collections of particles when computing their gravitational interaction.
//!
//! [`Between`] is a tuple struct of two objects where the first one is conventionally the affected
//! object and the second one is the affecting object. These objects can be particles or storages
//! of particles. [`Between`] is the underlying type used to implement most algorithms allowing
//! trivial implementation of other storage types.
//!
//! The [`Reordered`] storage defines a slice of particles and stores a copy of them in an
//! [`Ordered`] storage. These two storages make it easy for algorithms to skip particles that do
//! not affect other particles when computing interactions. A [`RootedOrthtree`] is a tree structure
//! (equivalent to a quadtree in 2D, an octree in 3D etc.) that stores particles to be used in
//! Barnes-Hut algorithms.
//!
//! ##### Example
//!
//! ```
//! use particular::prelude::*;
//! use particular::gravity::newtonian::{Acceleration, TreeData};
//!
//! use glam::Vec3;
//!
//! // Particles here are simple gravitational fields.
//! let particles: Vec<GravitationalField<Vec3, f32>> = vec![
//!     // ...
//! #   GravitationalField::new(Vec3::new(-10.0, 0.0, 0.0), 5.0),
//! #   GravitationalField::new(Vec3::new(-5.0, 20.0, 0.0), 0.0),
//! #   GravitationalField::new(Vec3::new(-50.0, 0.0, 5.0), 0.0),
//! #   GravitationalField::new(Vec3::new(10.0, 5.0, 5.0), 10.0),
//! #   GravitationalField::new(Vec3::new(0.0, -5.0, 20.0), 0.0),
//! ];
//!
//! // Function to determine if a field will affect particles.
//! fn is_massive(p: &GravitationalField<Vec3, f32>) -> bool {
//!     p.m != 0.0
//! }
//!
//! // Create an `Ordered` storage to split massive and massless particles.
//! let ordered = Ordered::new(&particles, is_massive);
//!
//! // We can build a `RootedOrthtree` from just the massive particles. Note that this is done
//! // automatically when computing over an [`Ordered`] or [`Reordered`] storage for the Barnes-Hut
//! // algorithm.
//! let tree = RootedOrthtree::new(
//!     ordered.affecting(),
//!     |p| p.position().to_array(),
//!     |slice| GravitationalField::centre_of_mass(slice.iter().cloned()),
//! );
//!
//! // Do something with the tree.
//! for (node, data) in std::iter::zip(&tree.get().nodes, &tree.get().data) {
//!     // ...
//! }
//!
//! // Compute the acceleration the massive particles in a tree exert on the massless particles.
//! let between = Between(ordered.non_affecting(), &tree);
//! let accelerations = between.barnes_hut(0.5, Acceleration::checked()).collect::<Vec<_>>();
//! ```
//!
//! #### Custom [`Interaction`] implementations
//!
//! Implementing [`Interaction<Between<&YourParticle, &YourParticle>>`] for `YourInteraction` allows
//! it to be used with the CPU brute-force algorithms. Other algorithms may require you to implement
//! other traits to be used, namely [`SimdInteraction`], [`ReduceSimdInteraction`],
//! [`BarnesHutInteraction`] and [`TreeInteraction`]. Refer to the documentation of the specific
//! algorithms for more information.
//!
//! ##### Example
//!
//! ```
//! use particular::prelude::*;
//!
//! use glam::DVec3;
//!
//! #[derive(Clone, Copy)]
//! struct Body {
//!     position: DVec3,
//!     mass: f64,
//! }
//! # impl Body {
//! #     fn new(position: DVec3, mass: f64) -> Self {
//! #         Self { position, mass }
//! #     }
//! # }
//!
//! #[derive(Clone)]
//! struct GravitationalForce(pub f64);
//!
//! impl Interaction<Between<&Body, &Body>> for GravitationalForce {
//!    type Output = DVec3;
//!
//!     fn compute(&mut self, Between(affected, affecting): Between<&Body, &Body>) -> DVec3 {
//!         if affected.position == affecting.position {
//!             return DVec3::ZERO;
//!         }
//!
//!         let r = affecting.position - affected.position;
//!         let l = r.length_squared();
//!         let f = self.0 * affected.mass * affecting.mass / (l * l.sqrt());
//!         
//!         r * f
//!     }
//! }
//!
//! // Distance: AU, Mass: M☉ (solar mass)
//! const G: f64 = 4.0 * std::f64::consts::PI * std::f64::consts::PI;
//! let sun = Body::new(DVec3::ZERO, 1.0);
//! let earth = Body::new(DVec3::new(1.0, 0.0, 0.0), 3.0027e-6);
//! let jupiter = Body::new(DVec3::new(5.2, 0.0, 0.0), 0.000954588);
//!
//! let solar_system = [sun, earth, jupiter];
//! let mut forces = solar_system.brute_force(GravitationalForce(G));
//!
//! let sun_earth = GravitationalForce(G).compute(Between(&sun, &earth));
//! let sun_jupiter = GravitationalForce(G).compute(Between(&sun, &jupiter));
//! let earth_jupiter = GravitationalForce(G).compute(Between(&earth, &jupiter));
//! assert_eq!(forces.next(), Some(sun_earth + sun_jupiter));
//! assert_eq!(forces.next(), Some(-sun_earth + earth_jupiter));
//! assert_eq!(forces.next(), Some(-sun_jupiter - earth_jupiter));
//! ```
//! </details>
//!
//! [`Position`]: gravity::Position
//! [`Mass`]: gravity::Mass
//! [`Ordered`]: storage::Ordered
//! [`Reordered`]: storage::Reordered
//! [`RootedOrthtree`]: storage::RootedOrthtree
//! [`Between`]: Between
//! [`Interaction`]: Interaction
//! [`SimdInteraction`]: SimdInteraction
//! [`ReduceSimdInteraction`]: ReduceSimdInteraction
//! [`BarnesHutInteraction`]: BarnesHutInteraction
//! [`TreeInteraction`]: TreeInteraction
//! [`gravity`]: gravity

/// Algorithms that use the GPU.
#[cfg(feature = "gpu")]
pub mod gpu;
/// Implementation of the computation of the gravitational force between particles.
pub mod gravity;
/// Algorithms that use multiple CPU threads.
#[cfg(feature = "parallel")]
pub mod parallel;
/// Algorithms that use one CPU thread.
pub mod sequential;
/// Collections used to store particles and apply optimisations and algorithms on them.
pub mod storage;
/// Tree and space partitioning implementation.
pub mod tree;

pub use storage::*;

/// Represents a pair of objects, which can be particles or storages of particles, between which an
/// interaction is computed.
///
/// The first object is the one being affected by the second object.
#[derive(Clone, Copy, Debug)]
pub struct Between<S1, S2>(pub S1, pub S2);

/// Trait to compute an interaction between particles contained in a storage.
///
/// This is the main trait used throughout `particular` to implement the available algorithms, and
/// is also how interactions between particles are defined.  
/// For example, implementing [`Interaction<Between<&YourParticle, &YourParticle>>`] for
/// `YourInteraction` allows it to be used with the CPU brute-force algorithms. Other algorithms
/// may require you to implement other traits to be used. Refer to the documentation of the
/// specific algorithms for more information.
///
/// # Example
///
/// ```
/// # use particular::prelude::*;
/// use glam::DVec3;
///
/// #[derive(Clone, Copy)]
/// struct Body {
///     position: DVec3,
///     mass: f64,
/// }
///
/// # impl Body {
/// #     fn new(position: DVec3, mass: f64) -> Self {
/// #         Self { position, mass }
/// #     }
/// # }
/// #[derive(Clone)]
/// struct GravitationalForce(pub f64);
///
/// impl Interaction<Between<&Body, &Body>> for GravitationalForce {
///    type Output = DVec3;
///
///     fn compute(&mut self, Between(affected, affecting): Between<&Body, &Body>) -> DVec3 {
///         if affected.position == affecting.position {
///             return DVec3::ZERO;
///         }
///
///         let r = affecting.position - affected.position;
///         let l = r.length_squared();
///         let f = self.0 * affected.mass * affecting.mass / (l * l.sqrt());
///         
///         r * f
///     }
/// }
///
/// // Distance: AU, Mass: M☉ (solar mass)
/// const G: f64 = 4.0 * std::f64::consts::PI * std::f64::consts::PI;
/// let sun = Body::new(DVec3::ZERO, 1.0);
/// let earth = Body::new(DVec3::new(1.0, 0.0, 0.0), 3.0027e-6);
/// let jupiter = Body::new(DVec3::new(5.2, 0.0, 0.0), 0.000954588);
///
/// let solar_system = [sun, earth, jupiter];
/// let mut forces = solar_system.brute_force(GravitationalForce(G));
///
/// let sun_earth = GravitationalForce(G).compute(Between(&sun, &earth));
/// let sun_jupiter = GravitationalForce(G).compute(Between(&sun, &jupiter));
/// let earth_jupiter = GravitationalForce(G).compute(Between(&earth, &jupiter));
///
/// assert_eq!(forces.next(), Some(sun_earth + sun_jupiter));
/// assert_eq!(forces.next(), Some(-sun_earth + earth_jupiter));
/// assert_eq!(forces.next(), Some(-sun_jupiter - earth_jupiter));
/// ```
pub trait Interaction<Storage> {
    /// The computed interaction. This can be one or multiple values depending on the storage used.
    type Output;

    /// Returns the interaction between the particles in the storage.
    fn compute(&mut self, storage: Storage) -> Self::Output;
}

/// Trait to define how particles of type `P` are converted to SIMD particles for an interaction.
pub trait SimdInteraction<const L: usize, P> {
    /// The SIMD particle that the interaction is computed with.
    type Simd;

    /// Creates a SIMD value with all lanes set to the specified value.
    fn lanes_splat(value: &P) -> Self::Simd;

    /// Creates a SIMD value with its lanes set to the given values in the array.
    fn lanes_array(array: [P; L]) -> Self::Simd;

    /// Creates a SIMD value with its lanes set to the given values in the slice. If the slice is
    /// smaller than the number of lanes, the remaining lanes should be set to a default value.
    fn lanes_slice(slice: &[P]) -> Self::Simd;

    /// Creates a vector of SIMD values from the given slice of scalar values. If the slice is not
    /// a multiple of the number of lanes, the remaining lanes should be set to a default value.
    #[inline]
    fn values_to_simd_vec(slice: &[P]) -> Vec<Self::Simd> {
        slice.chunks(L).map(Self::lanes_slice).collect()
    }
}

/// Trait to reduce the lanes of a SIMD interaction that has been computed.
pub trait ReduceSimdInteraction<Storage>: Interaction<Storage> {
    /// The resulting type after applying the reduce operation.
    type Reduced;

    /// Sums the lanes of the SIMD interaction.
    fn reduce_sum(output: <Self as Interaction<Storage>>::Output) -> Self::Reduced;
}

/// Trait to define how particles of type `P` are used in a Barnes-Hut algorithm for an
/// interaction.
pub trait BarnesHutInteraction<P1, P2>: Interaction<Between<P1, P2>> {
    /// The type representing the distance between two particles.
    type Distance;

    /// Returns the squared distance between the two particles.
    fn distance_squared(lhs: P1, rhs: P2) -> Self::Distance;
}

/// Trait to define how particles of type `P` are used in the construction of a tree built for an
/// interaction.
pub trait TreeInteraction<P> {
    /// Coordinates type used in the construction of the tree.
    type Coordinates;

    /// The data stored in the built tree.
    type TreeData;

    /// Returns the coordinates of the particle.
    fn coordinates(p: &P) -> Self::Coordinates;

    /// Computes the data stored in a node of the tree from the given particles.
    fn compute_data(particles: &[P]) -> Self::TreeData;
}

/// Commonly used types, re-exported.
pub mod prelude {
    #[cfg(feature = "glam")]
    pub use crate::gravity::wide_glam;

    // Common traits and their derive macros.
    pub use crate::{
        gravity::{GravitationalField, Mass, Norm, Position},
        storage::{Ordered, Reordered, RootedOrthtree},
        Between, Interaction,
    };
    pub use particular_derive::{Mass, Position};

    #[cfg(feature = "gpu")]
    pub use crate::gpu::{GpuCompute, GpuResources, MemoryStrategy};
    pub use crate::sequential::SequentialCompute;
    #[cfg(feature = "parallel")]
    pub use {crate::parallel::ParallelCompute, rayon::prelude::*};
}
