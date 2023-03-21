# Particular

<div align="center"><img src="./particular-showcase.gif"></div>

[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/canleskis/particular#license)
[![Crates.io](https://img.shields.io/crates/v/particular)](https://crates.io/crates/particular)
[![Docs](https://docs.rs/particular/badge.svg)](https://docs.rs/particular)

Particular is a crate providing a simple way to simulate N-body gravitational interaction of particles in Rust.

## Goals

The main goal of this crate is to provide users with a simple API to set up N-body gravitational simulations that can easily be integrated into existing game and physics engines.
Thus it does not concern itself with numerical integration or other similar tools and instead only focuses on the acceleration calculations.

Multiple algorithms are available to compute the acceleration between particles as [`ComputeMethods`].

### Computation algorithms

| ComputeMethod       | [BruteForce] | [BarnesHut] |
| :------------------ | :----------- | :---------- |
| GPU                 | &check;      | &cross;     |
| CPU single-threaded | &check;      | &check;     |
| CPU multi-threaded  | &check;      | &check;     |

[BruteForce]: https://en.wikipedia.org/wiki/N-body_problem#Simulation
[BarnesHut]: https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation

Generally speaking, the BruteForce algoritm is more accurate, but slower. The BarnesHut algorithm allows trading accuracy for speed by increasing the `theta` parameter.  
You can read more about their relative performance [here](#notes-on-performance).

Particular uses [rayon](https://github.com/rayon-rs/rayon) for parallelization. Enable the "parallel" feature to access the available compute methods.

Particular uses [wgpu](https://github.com/gfx-rs/wgpu) for GPU computation. Enable the "gpu" feature to access the available compute methods.

## Using Particular

### Implementing the Particle trait

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

Used when the type cannot directly provide a position and a gravitational parameter.

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

If you can't implement Particle on a type, you can almost certainly use the fact that it is implemented for tuples of a vector and its scalar type.

```rust
let particle = (Vec3::ONE, 5.0);

assert_eq!(particle.position(), Vec3::ONE);
assert_eq!(particle.mu(), 5.0);
```

### Computing and using the gravitational acceleration

In order to get the acceleration of a type, you can use the [accelerations] or [map_accelerations] methods on iterators.  
These effectively return the original iterator zipped with the acceleration of its items.

As such, you can create an iterator from a collection and get the acceleration using either methods depending on if the items implement Particle.

Pass a mutable reference to a [`ComputeMethod`] when calling either method.

### Examples

#### When the iterated type doesn't implement Particle

```rust
// Items are a tuple of a velocity, a position and a mass.
// We map them to a tuple of the position and the mu, since this implements `Particle`.
let accelerations = items
    .iter_mut()
    .map_accelerations(|(_, position, mass)| (*position, *mass * G), &mut cm);

for ((velocity, position, _), acceleration) in accelerations {
    *velocity += acceleration * DT;
    *position += *velocity * DT;
}
```

#### When the iterated type implements Particle

```rust
for (body, acceleration) in bodies.iter_mut().accelerations(&mut cm) {
    body.velocity += acceleration * DT;
    body.position += body.velocity * DT;
}
```

## Notes on performance

Particular is built with performance in mind and uses multiple ways of computing the acceleration between particles in the form of [`ComputeMethods`].

Here is a comparison of the three current available compute methods on an i9 9900KF and an RTX 3080:

<div align="center"><img src="particular-comparison.png" alt="Performance chart" /></div>

Above 1,000 particles the parallel implementation is about 5x faster than the sequential one, whilst the GPU implementation ranges from 50x to 100x faster than the parallel implementation above 15,000 particles (250x to 500x faster than sequential).

Depending on your needs and platform, you may opt for one compute method or another. You can also implement the trait on your own type to combine multiple compute methods and switch between them depending on certain conditions (e.g. the particle count).

## Contribution

PRs are welcome!

[`Particle`]: https://docs.rs/particular/latest/particular/particle/trait.Particle.html
[`ComputeMethods`]: https://docs.rs/particular/latest/particular/compute_method/trait.ComputeMethod.html
[accelerations]: https://docs.rs/particular/latest/particular/iterators/trait.Compute.html#method.accelerations
[map_accelerations]: https://docs.rs/particular/latest/particular/iterators/trait.MapCompute.html#method.map_accelerations
