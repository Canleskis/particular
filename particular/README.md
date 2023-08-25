# Particular

<div align="center">
    <img src="https://github.com/Canleskis/particular/blob/main/particular/particular-showcase.gif?raw=true">
</div>

[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/canleskis/particular#license)
[![Crates.io](https://img.shields.io/crates/v/particular)](https://crates.io/crates/particular)
[![Docs](https://docs.rs/particular/badge.svg)](https://docs.rs/particular)

Particular is a crate providing a simple way to simulate N-body gravitational interaction of particles in Rust.

## Goals

The main goal of this crate is to provide users with a simple API to set up N-body gravitational simulations that can easily be integrated into existing game and physics engines.
Thus it does not concern itself with numerical integration or other similar tools and instead only focuses on the acceleration calculations.

Particular is also built with performance in mind and provides multiple ways of computing the acceleration between particles.

### Computation algorithms

There are currently 2 algorithms used by the available compute methods: [BruteForce](https://en.wikipedia.org/wiki/N-body_problem#Simulation) and [BarnesHut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation).

Generally speaking, the BruteForce algorithm is more accurate, but slower. The BarnesHut algorithm allows trading accuracy for speed by increasing the `theta` parameter.  
You can read more about their relative performance [here](#notes-on-performance).

Particular uses [rayon](https://github.com/rayon-rs/rayon) for parallelization. Enable the "parallel" feature to access the available compute methods.  
Particular uses [wgpu](https://github.com/gfx-rs/wgpu) for GPU computation. Enable the "gpu" feature to access the available compute methods.

## Using Particular

### Implementing the [`Particle`] trait

When possible, it can be useful to implement [`Particle`] on a type.

#### Deriving

Used when the type has fields named `position` and `mu`:

```rust
#[derive(Particle)]
struct Body {
    position: Vec3,
    mu: f32,
//  ...
}
```

#### Manual implementation

Used when the type does not directly provide a position and a gravitational parameter.

```rust
struct Body {
    position: Vec3,
    mass: f32,
//  ...
}

impl Particle for Body {
    type Scalar = f32;

    type Vector = Vec3;
    
    fn position(&self) -> Vec3 {
        self.position
    }
    
    fn mu(&self) -> f32 {
        self.mass * G
    }
}
```

If you can't implement [`Particle`] on a type, you can almost certainly use the fact that it is implemented for tuples of a vector and its scalar type.

```rust
let particle = (Vec3::ONE, 5.0);

assert_eq!(particle.position(), Vec3::ONE);
assert_eq!(particle.mu(), 5.0);
```

### Computing and using the gravitational acceleration

In order to compute the accelerations of your particles, you can use the [accelerations] method on iterators, passing in a [`ComputeMethod`] of your choice.

### Examples

#### When the iterated type doesn't implement [`Particle`]

```rust
// Items are a tuple of a velocity, a position and a mass.
// We map them to a tuple of the position and the mu, since this implements `Particle`.
let accelerations = items
    .iter()
    .map(|(_, position, mass)| (*position, *mass * G))
    .accelerations(cm);

for (acceleration, (velocity, position, _)) in accelerations.zip(&mut items) {
    *velocity += acceleration * DT;
    *position += *velocity * DT;
}
```

#### When the iterated type implements [`Particle`]

```rust
for (acceleration, body) in bodies.iter().accelerations(cm).zip(&mut bodies) {
    body.velocity += acceleration * DT;
    body.position += body.velocity * DT;
}
```

## Notes on performance

Here is a comparison between 7 available compute methods using an i9 9900KF and an RTX 3080:

<div align="center">
    <img src="https://github.com/Canleskis/particular/blob/main/particular/particular-comparison.png?raw=true" alt="Performance chart" />
</div>

Depending on your needs and platform, you may opt for one compute method or another.
You can also implement the trait on your own type to use other algorithms or combine multiple compute methods and switch between them depending on certain conditions (e.g. the particle count).

## License

This project is licensed under either of [Apache License, Version 2.0](https://github.com/Canleskis/particular/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/Canleskis/particular/blob/main/LICENSE-MIT), at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache 2.0 license, shall be dual licensed as above, without any additional terms or conditions.

[`Particle`]: https://docs.rs/particular/latest/particular/particle/trait.Particle.html
[accelerations]: https://docs.rs/particular/latest/particular/particle/trait.Accelerations.html#method.accelerations
[`ComputeMethod`]: https://docs.rs/particular/latest/particular/compute_method/trait.ComputeMethod.html
