# Particular

<div align="center">
    <img src="https://github.com/Canleskis/particular/blob/main/particular/particular-showcase.gif?raw=true" alt="showcase gif">
</div>

[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/canleskis/particular#license)
[![Crates.io](https://img.shields.io/crates/v/particular)](https://crates.io/crates/particular)
[![Docs](https://docs.rs/particular/badge.svg)](https://docs.rs/particular)

Particular is a crate providing a simple way to simulate N-body gravitational interaction of
particles in Rust.

## [Change log](https://github.com/Canleskis/particular/blob/main/particular/CHANGELOG.md)

## Goals

The main goal of this crate is to provide users with a simple API to set up N-body gravitational
simulations that can easily be integrated into existing game and physics engines. Thus it does
not concern itself with numerical integration or other similar tools and instead only focuses on
the acceleration calculations.

Particular is also built with performance in mind and provides multiple ways of computing the
acceleration between particles.

### Computation algorithms

There are currently 2 algorithms used by the available compute methods:
[Brute-force](https://en.wikipedia.org/wiki/N-body_problem#Simulation) and
[Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation).

Generally speaking, the Brute-force algorithm is more accurate, but slower. The Barnes-Hut
algorithm allows trading accuracy for speed by increasing the `theta` parameter.  
You can see more about their relative performance [here](https://particular.rs/benchmarks/).

Particular uses [rayon](https://github.com/rayon-rs/rayon) for parallelization and
[wgpu](https://github.com/gfx-rs/wgpu) for GPU computation.  
Enable the respective `parallel` and `gpu` features to access the available compute methods.

## Using Particular

Particular consists of two "modules", one that takes care of the abstraction of the computation of
the gravitational forces between bodies for different floating-point types and dimensions, and one
that facilitates usage of that abstraction for user-defined andnon-user-defined types. For most
simple use cases, the latter is all that you need to know about.

### Simple usage

The [`Particle`] trait provides the main abstraction layer between the internal representation
of the position and mass of an object in N-dimensional space and external types by defining
methods to retrieve a position and a gravitational parameter.  
These methods respectively return an array of scalars and a scalar, which are converted using
the [point_mass] method to interface with the underlying algorithm implementations.

#### Implementing the [`Particle`] trait

When possible, it can be useful to implement [`Particle`] on a type.

##### Deriving

Used when the type has fields named `position` and `mu`:

```rust
#[derive(Particle)]
#[dim(3)]
struct Body {
    position: Vec3,
    mu: f32,
//  ...
}
```

##### Manual implementation

Used when the type does not directly provide a position and a gravitational parameter.

```rust
struct Body {
    position: Vec3,
    mass: f32,
//  ...
}

impl Particle for Body {
    type Array = [f32; 3];

    fn position(&self) -> [f32; 3] {
        self.position.into()
    }

    fn mu(&self) -> f32 {
        self.mass * G
    }
}
```

If you can't implement [`Particle`] on a type, you can use the fact that it is implemented for
tuples of an array and its scalar type instead of creating an intermediate type.

```rust
let particle = ([1.0, 1.0, 0.0], 5.0);

assert_eq!(particle.position(), [1.0, 1.0, 0.0]);
assert_eq!(particle.mu(), 5.0);
```

#### Computing and using the gravitational acceleration

In order to compute the accelerations of your particles, you can use the [accelerations] method
on iterators, passing in a mutable reference to a [`ComputeMethod`] of your choice. It returns
the acceleration of each iterated item, preserving the original order.  
Because it collects the mapped particles in a [`ParticleReordered`] in order to optimise the
computation of forces of massless particles, this method call results in one additional
allocation. See the [advanced usage](#advanced-usage) section for information on how to opt out.

##### When the iterated type implements [`Particle`]

```rust
for (acceleration, body) in bodies.iter().accelerations(&mut cm).zip(&mut bodies) {
    body.velocity += Vec3::from(acceleration) * DT;
    body.position += body.velocity * DT;
}
```

##### When the iterated type doesn't implement [`Particle`]

```rust
// Items are a tuple of a velocity, a position and a mass.
// We map them to a tuple of the positions as an array and the mu,
// since this implements `Particle`.
let accelerations = items
    .iter()
    .map(|(_, position, mass)| (*position.as_array(), *mass * G))
    .accelerations(&mut cm);

for (acceleration, (velocity, position, _)) in accelerations.zip(&mut items) {
    *velocity += Vec3::from(acceleration) * DT;
    *position += *velocity * DT;
}
```

### Advanced usage

In some instances the iterator abstraction provided by particular might not be flexible enough.
For example, you might need to access the tree built from the particles for the Barnes-Hut
algorithm, want to compute the gravitational forces between two distinct collections of particles,
or both at the same time.

#### The [`PointMass`] type

The underlying type used in storages is the [`PointMass`], a simple representation in
N-dimensional space of a position and a gravitational parameter. Instead of going through a
[`ComputeMethod`], you can directly use the different generic methods available to compute the
gravitational forces between [`PointMass`]es, with variants optimised for scalar and simd types.

##### Example

```rust
use particular::math::Vec2;

use storage::PointMass;

let p1 = PointMass::new(Vec2::new(0.0, 1.0), 1.0);
let p2 = PointMass::new(Vec2::new(0.0, 0.0), 1.0);
let softening = 0.0;

assert_eq!(p1.force_scalar::<false>(p2.position, p2.mass, softening), Vec2::new(0.0, -1.0));
```

#### Storages and built-in [`ComputeMethod`] implementations

Storages are containers that make it easy to apply certain optimisation or algorithms on
collections of particles when computing their gravitational acceleration.

The [`ParticleSystem`] storage defines an `affected` slice of particles and a `massive` storage,
allowing algorithms to compute gravitational forces the particles in the `massive` storage exert
on the `affected` particles. It is used to implement most compute methods, and blanket
implementations with the other storages allow a [`ComputeMethod`] implemented with
[`ParticleSliceSystem`] or [`ParticleTreeSystem`] to also be implemented with the other
storages.

The [`ParticleReordered`] similarly defines a slice of particles, but stores a copy of them in a
[`ParticleOrdered`]. These two storages make it easy for algorithms to skip particles with no
mass when computing the gravitational forces of particles.

##### Example

```rust
use particular::math::Vec3;

let particles = vec![
    // ...
];

// Create a `ParticleOrdered` to split massive and massless particles.
let ordered = ParticleOrdered::from(&*particles);

// Build a `ParticleTree` from the massive particles.
let tree = ParticleTree::from(ordered.massive());

// Do something with the tree.
for (node, data) in std::iter::zip(&tree.get().nodes, &tree.get().data) {
    // ...
}

let bh = &mut sequential::BarnesHut { theta: 0.5 };
// The implementation computes the acceleration exerted on the particles in
// the `affected` slice.
// As such, this only computes the acceleration of the massless particles.
let accelerations = bh.compute(ParticleSystem {
    affected: ordered.massless(),
    massive: &tree,
});
```

#### Custom [`ComputeMethod`] implementations

In order to work with the highest number of cases, built-in compute method implementations may
not be the most appropriate or optimised for your specific use case. You can implement the
[`ComputeMethod`] trait on your own type to satisfy your specific requirements but also if you
want to implement other algorithms.

##### Example

```rust
use particular::math::Vec3;

struct MyComputeMethod;

impl ComputeMethod<ParticleReordered<'_, Vec3, f32>> for MyComputeMethod {
    type Output = Vec<Vec3>;

    #[inline]
    fn compute(&mut self, storage: ParticleReordered<Vec3, f32>) -> Self::Output {
        // Only return the accelerations of the massless particles.
        sequential::BruteForceScalar.compute(ParticleSystem {
            affected: storage.massless(),
            massive: storage.massive(),
        })
    }
}
```

## License

This project is licensed under either of [Apache License, Version 2.0](https://github.com/Canleskis/particular/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/Canleskis/particular/blob/main/LICENSE-MIT), at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache 2.0 license, shall be dual licensed as above, without any additional terms or conditions.

[accelerations]: https://docs.rs/particular/latest/particular/particle/trait.Accelerations.html#method.accelerations
[point_mass]: https://docs.rs/particular/latest/particular/particle/trait.IntoPointMass.html#method.point_mass
[`Particle`]: https://docs.rs/particular/latest/particular/particle/trait.Particle.html
[`ComputeMethod`]: https://docs.rs/particular/latest/particular/compute_method/trait.ComputeMethod.html
[`ParticleReordered`]: https://docs.rs/particular/latest/particular/compute_method/storage/struct.ParticleReordered.html
[`ParticleOrdered`]: https://docs.rs/particular/latest/particular/compute_method/storage/struct.ParticleOrdered.html
[`ParticleSystem`]: https://docs.rs/particular/latest/particular/compute_method/storage/struct.ParticleSystem.html
[`ParticleSliceSystem`]: https://docs.rs/particular/latest/particular/compute_method/storage/struct.ParticleSliceSystem.html
[`ParticleTreeSystem`]: https://docs.rs/particular/latest/particular/compute_method/storage/struct.ParticleTreeSystem.html
[`PointMass`]: https://docs.rs/particular/latest/particular/compute_method/storage/struct.PointMass.html
