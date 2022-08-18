# Particular
 
Particular is crate that provides a simple way to simulate N-body gravitational interaction of particles in Rust.
The API to setup a simulation is straightforward:
1. First, you will need a type that implements the `Particle` trait. 
   - It can either be derived if your type contains both a `position` and a `mu` field.
   - It can also be implemented manually.
```
// Deriving.

#[derive(Particle)]
pub struct Body {
    position: Vec3,
    mu: f32,
    ...
}
```
```
// Manual implementation.

impl Particle for Body {
    fn position(&self) -> Vec3 {
        self.position
    }

    fn mu(&self) -> f32 {
        self.mass * G
    }
}
```

2. You will then need to use that type to create a `ParticleSet` that will contain the particles. It will only handle the gravitational acceleration calculations. Currently, it stores the particles in two different vectors depending on if the particle has mass or doesn't. This allows optimization in the case of massless particles (which can represent objects that do not need to affect other objects, like a spaceship).
```
let mut particle_set = ParticleSet::new();
// If the type cannot be inferred, use the turbofish syntax: ParticleSet::<Body>::new().
particle_set.add(Body { position, mu });
```

3. Finally, using the `result` method of `ParticleSet`, you can iterate over the computed gravitational acceleration of each particle.
```
for (body, gravity) in particle_set.result() {
    ...
}
```
