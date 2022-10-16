# Particular

<p align="center">
  <img src="./particular-showcase.gif">
</p>

[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/canleskis/particular#license)
[![Crates.io](https://img.shields.io/crates/v/particular)](https://crates.io/crates/particular)
[![Docs](https://docs.rs/particular/badge.svg)](https://docs.rs/particular)

Particular is a crate providing a simple way to simulate N-body gravitational interaction of particles in Rust.

## Goals

The main goal of this crate is to provide users with a simple API to setup N-body gravitational simulations that can easily be integrated in existing game and physics engines.
Thus, it does not include numerical integration or other similar tools and instead only focuses on the acceleration calculations.

Currently, acceleration calculations are computed naively by iterating over all the particles and summing the acceleration caused by all the `massive` particles.
In the future, I would like to implement other algorithms such as [Barnes-Hut algorithm](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) or even use compute shaders on the GPU for faster calculations.

Particular can be used with a parallel implementation on the CPU thanks to the [rayon](https://github.com/rayon-rs/rayon) crate. Use the "parallel" feature to enable it, which can lead to huge performance improvements.

# Using Particular

The API to setup a simulation is straightforward:

## Implementing the `Particle` trait

#### Attribute macro or deriving

Used in most cases, when the type has fields named `position` and `mu`:

```rust
#[particle(3)]
pub struct Body {
    position: Vec3,
    mu: f32,
//  ...
}
```

This is equivalent to:

```rust
#[derive(Particle)]
#[dim(3)]
pub struct Body {
    position: Vec3,
    mu: f32,
//  ...
}
```

#### Manual implementation

Used when the type has more complex fields and cannot directly provide a position and a gravitational parameter.

```rust
struct Body {
    position: Vec3,
    mass: f32,
//  ...
}

impl Particle for Body {
    type Vector = VectorDescriptor<3, Vec3>;
    
    fn position(&self) -> Vec3 {
        self.position
    }

    fn mu(&self) -> f32 {
        self.mass * G
    }
}
```

## Setting up the simulation

Using the type implementing `Particle`, create a `ParticleSet` that will contain the particles.

Particles are stored in two vectors, `massive` or `massless`, depending on if they have mass or not.
This allows optimizations in the case of massless particles (which represents objects that do not need to affect other objects, like a spaceship).

```rust
// If the type cannot be inferred, use the turbofish syntax:
let mut particle_set = ParticleSet::<Body>::new();
// Otherwise:
let mut particle_set = ParticleSet::new();

particle_set.add(Body { position, mu });
```

## Computing and using the gravitational acceleration

Finally, use the `result` method of `ParticleSet`, which returns an iterator over a mutable reference to the `Particle` and its computed gravitational acceleration.

```rust
for (acceleration, particle) in particle_set.result() {
    particle.velocity += acceleration * DT;
    particle.position += particle.velocity * DT;
}
```

## Contribution

PRs are welcome!
