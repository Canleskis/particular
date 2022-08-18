# Particular
 
<p align="center">
  <img src="./particular_5000_bodies.gif">
</p>
 
Particular is a crate providing a simple way to simulate N-body gravitational interaction of particles in Rust.
The API to setup a simulation is straightforward:

## Implementing the `Particle` trait. 
It can be derived if your type contains both a `position` and a `mu` field.
```
#[derive(Particle)]
pub struct Body {
    position: Vec3,
    mu: f32,
    ...
}
```
It can also be implemented manually.
```
impl Particle for Body {
    fn position(&self) -> Vec3 {
        self.position
    }

    fn mu(&self) -> f32 {
        self.mass * G
    }
}
```
## Setting up the simulation.
Using your type implementing `Particle`, you will need to create a `ParticleSet` that will contain the particles. It will only handle the gravitational acceleration calculations.

Currently, it stores the particles in two different vectors depending on if the particle has mass or doesn't. This allows optimizations in the case of massless particles (which can represent objects that do not need to affect other objects, like a spaceship).
```
let mut particle_set = ParticleSet::new();
// If the type cannot be inferred, use the turbofish syntax: ParticleSet::<Body>::new().
particle_set.add(Body { position, mu });
```
## Computing and using the gravitational acceleration.
Finally, using the `result` method of `ParticleSet`, you can iterate over the computed gravitational acceleration of each particle.
```
for (particle, acceleration) in particle_set.result() {
    ...
}
```
`body` here being of the type you used for the `ParticleSet`.
