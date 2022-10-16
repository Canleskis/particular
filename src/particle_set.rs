use crate::{
    particle::{Particle, ToPointMass},
    vector::{Descriptor, FromVector, SIMD},
};

/// The structure used to store the particles and calculate their acceleration.
///
/// Particles are stored in two vectors, `massive` or `massless`, depending on if they have mass or not.
/// This allows optimizations in the case of massless particles (which represents objects that do not need to affect other objects, like a spaceship).
/// ```
/// # use particular::prelude::*;
/// # use glam::Vec3;
/// #
/// # #[particle(3)]
/// # pub struct Body {
/// #     position: Vec3,
/// #     mu: f32,
/// # //  ...
/// # }
/// # let position = Vec3::ONE;
/// # let mu = 1E5;
/// #
/// // If the type cannot be inferred, use the turbofish syntax:
/// let mut particle_set = ParticleSet::<Body>::new();
/// // Otherwise:
/// let mut particle_set = ParticleSet::new();
///
/// particle_set.add(Body { position, mu });
/// ```
///
/// If a particle needs to be removed from the [`ParticleSet`], it is preferrable to create a new one, as it is a cheap operation.
#[derive(Default)]
pub struct ParticleSet<P: Particle> {
    massive: Vec<P>,
    massless: Vec<P>,
}

impl<P: Particle> ParticleSet<P> {
    /// Creates an empty [`ParticleSet`].
    pub fn new() -> Self {
        Self {
            massive: Vec::new(),
            massless: Vec::new(),
        }
    }
}

impl<P: Particle> ParticleSet<P> {
    /// Adds a [`Particle`] to the [`ParticleSet`].
    ///
    /// This method adds the particle in the corresponding vector depending on its mass.
    pub fn add(&mut self, particle: P) {
        if particle.mu() != 0.0 {
            self.massive.push(particle);
        } else {
            self.massless.push(particle);
        }
    }

    /// Adds a [`Particle`] that has mass to the [`ParticleSet`].
    ///
    /// Panics if the particle doesn't have mass.
    pub fn add_massive(&mut self, particle: P) {
        assert!(particle.mu() != 0.0);
        self.massive.push(particle);
    }

    /// Adds a [`Particle`] that has no mass to the [`ParticleSet`].
    ///
    /// Panics if the particle has mass.
    pub fn add_massless(&mut self, particle: P) {
        assert!(particle.mu() == 0.0);
        self.massless.push(particle);
    }
}

impl<P: Particle> ParticleSet<P> {
    /// Converts the massive particules into their respective point-mass.
    #[inline]
    fn massive_particles(&self) -> Vec<(SIMD, f32)> {
        self.massive().map(P::point_mass).collect()
    }

    /// Converts the massless particules into their respective point-mass.
    #[inline]
    fn massless_particles(&self) -> Vec<(SIMD, f32)> {
        self.massless().map(P::point_mass).collect()
    }

    /// Computes the accelerations of the `massive` [particles](Particle) then the `massless` ones.
    #[inline]
    #[cfg(not(feature = "parallel"))]
    fn compute(&self) -> Vec<SIMD> {
        let massive = self.massive_particles();
        let massless = self.massless_particles();

        massive
            .iter()
            .chain(massless.iter())
            .map(|&(position1, _)| {
                massive
                    .iter()
                    .fold(SIMD::ZERO, |acceleration, &(position2, mass2)| {
                        let dir = position2 - position1;
                        let mag_2 = dir.length_squared();

                        let grav_acc = if mag_2 != 0.0 {
                            dir * mass2 / (mag_2 * mag_2.sqrt())
                        } else {
                            dir
                        };

                        acceleration + grav_acc
                    })
            })
            .collect()
    }

    /// Computes in parallel the accelerations of the `massive` [particles](Particle) then the `massless` ones.
    #[inline]
    #[cfg(feature = "parallel")]
    fn compute(&self) -> Vec<SIMD> {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

        let massive = self.massive_particles();
        let massless = self.massless_particles();

        massive
            .par_iter()
            .chain(massless.par_iter())
            .map(|&(position1, _)| {
                massive
                    .iter()
                    .fold(SIMD::ZERO, |acceleration, &(position2, mass2)| {
                        let dir = position2 - position1;
                        let mag_2 = dir.length_squared();

                        let grav_acc = if mag_2 != 0.0 {
                            dir * mass2 / (mag_2 * mag_2.sqrt())
                        } else {
                            dir
                        };

                        acceleration + grav_acc
                    })
            })
            .collect()
    }

    /// Iterates over the `massive` [particles](Particle).
    #[inline]
    pub fn massive(&self) -> impl Iterator<Item = &P> {
        self.massive.iter()
    }

    /// Mutably iterates over the `massive` [particles](Particle).
    #[inline]
    pub fn massive_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.massive.iter_mut()
    }

    /// Iterates over the `massless` [particles](Particle).
    #[inline]
    pub fn massless(&self) -> impl Iterator<Item = &P> {
        self.massless.iter()
    }

    /// Mutably iterates over the `massless` [particles](Particle).
    #[inline]
    pub fn massless_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.massless.iter_mut()
    }

    /// Iterates over the `massive` [particles](Particle), then the `massless` ones.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &P> {
        self.massive.iter().chain(self.massless.iter())
    }

    /// Mutably iterates over the `massive` [particles](Particle), then the `massless` ones.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.massive.iter_mut().chain(self.massless.iter_mut())
    }

    /// Iterates over the accelerations of the `massive` [particles](Particle) then the `massless` ones.
    ///
    /// When this method is called, the acceleration of all the particles in the [`ParticleSet`] will be computed.
    #[inline]
    pub fn accelerations(&self) -> impl Iterator<Item = <P::Vector as Descriptor>::Type> {
        self.compute().into_iter().map(P::Vector::from_simd)
    }

    /// Returns an iterator over a mutable reference to the [`Particle`] and its computed gravitational acceleration.
    ///
    /// Equivalent to:
    /// ```ignore
    /// particle_set
    ///     .accelerations()
    ///     .zip(particle_set.iter_mut());
    /// ```
    /// # Example
    /// ```
    /// # use particular::prelude::*;
    /// # use glam::Vec3;
    /// #
    /// # const DT: f32 = 1.0 / 60.0;
    /// #
    /// # #[particle(3)]
    /// # pub struct Body {
    /// #     position: Vec3,
    /// #     velocity: Vec3,
    /// #     mu: f32,
    /// # }
    /// # let mut particle_set = ParticleSet::<Body>::new();
    /// for (acceleration, particle) in particle_set.result() {
    ///     particle.velocity += acceleration * DT;
    ///     particle.position += particle.velocity * DT;
    /// }
    /// ```
    #[inline]
    pub fn result(&mut self) -> impl Iterator<Item = (<P::Vector as Descriptor>::Type, &mut P)> {
        self.accelerations().zip(self.iter_mut())
    }
}

#[cfg(test)]
pub mod tests {
    use crate::prelude::*;
    use glam::Vec3;
    use particular_derive::particle;

    #[particle(3)]
    pub struct Body {
        position: Vec3,
        mu: f32,
    }

    #[test]
    #[should_panic]
    fn add_massless_particle_as_massive() {
        let mut particle_set = ParticleSet::new();

        particle_set.add_massive(Body {
            position: Vec3::ZERO,
            mu: 0.0,
        });
    }

    #[test]
    #[should_panic]
    fn add_massive_particle_as_massless() {
        let mut particle_set = ParticleSet::new();

        particle_set.add_massless(Body {
            position: Vec3::ZERO,
            mu: 1.0,
        });
    }

    #[test]
    fn add_particles() {
        let mut particle_set = ParticleSet::new();

        particle_set.add_massive(Body {
            position: Vec3::ZERO,
            mu: 1.0,
        });

        particle_set.add_massless(Body {
            position: Vec3::ZERO,
            mu: 0.0,
        });
    }

    fn with_two_particles(p1: (Vec3, f32), p2: (Vec3, f32)) -> ParticleSet<Body> {
        let mut particle_set = ParticleSet::new();

        particle_set.add(Body {
            position: p1.0,
            mu: p1.1,
        });

        particle_set.add(Body {
            position: p2.0,
            mu: p2.1,
        });

        particle_set
    }

    #[test]
    fn add_particles_correspondence() {
        let p1 = (Vec3::splat(1.0), 0.0);
        let p2 = (Vec3::splat(-1.0), 8.0);

        let particle_set = with_two_particles(p1, p2);
        let mut iter = particle_set.iter();

        assert!(p2.1 == iter.next().unwrap().mu);
        assert!(p1.1 == iter.next().unwrap().mu);
    }

    const EPSILON: f32 = 1E-6;

    #[test]
    fn acceleration_calculation() {
        let p1 = (Vec3::default(), 2.0);
        let p2 = (Vec3::splat(1.0), 3.0);

        let dir = p2.0 - p1.0;
        let mag_2 = dir.length_squared();
        let grav_acc = dir / (mag_2 * mag_2.sqrt());

        for (acceleration, particle) in with_two_particles(p1, p2).result() {
            if particle.mu == p1.1 {
                assert!((acceleration - grav_acc * p2.1).length_squared() < EPSILON);
            } else {
                assert!((acceleration + grav_acc * p1.1).length_squared() < EPSILON);
            }
        }
    }
}
