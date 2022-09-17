use crate::{NormedVector, Particle, ToPointMass, Vector};

#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// The structure used to store the particles and calculate their acceleration.
///
/// Particles are stored in two vectors, `massive` or `massless`, depending on if they have mass or not.
/// This allows optimizations in the case of massless particles (which represents objects that do not need to affect other objects, like a spaceship).
/// ```
/// # use particular::prelude::Particle;
/// # use particular::ParticleSet;
/// # use glam::Vec3;
/// #
/// # #[derive(Particle)]
/// # pub struct Body {
/// #     position: Vec3,
/// #     mu: f32,
/// # //  ...
/// # }
/// # let position = Vec3::ONE;
/// # let mu = 1E5;
/// #
/// let mut particle_set = ParticleSet::new();
/// // If the type cannot be inferred, use the turbofish syntax:
/// // let mut particle_set = ParticleSet::<Body>::new();
///
/// particle_set.add(Body { position, mu });
/// ```
///
#[derive(Default)]
pub struct ParticleSet<P: Particle> {
    massive: Vec<P>,
    massless: Vec<P>,
}

impl<P: Particle> ParticleSet<P> {
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
    #[inline]
    #[cfg(not(feature = "parallel"))]
    fn get_accelerations(&self) -> Vec<P::Vector> {
        let massive = self.massive.iter().map(P::point_mass).collect::<Vec<_>>();
        let massless = self.massless.iter().map(P::point_mass).collect::<Vec<_>>();

        let accelerations = massive.iter().chain(&massless).map(|particle1| {
            massive
                .iter()
                .fold(P::Vector::zero_value(), |acceleration, particle2| {
                    let dir = particle2.0 - particle1.0;
                    let mag_2 = dir.norm_sq();

                    let grav_acc = if mag_2 != 0.0 {
                        dir * particle2.1 / (mag_2 * mag_2.sqrt())
                    } else {
                        dir
                    };

                    acceleration + grav_acc
                })
        });

        accelerations.collect()
    }

    #[inline]
    #[cfg(feature = "parallel")]
    fn get_accelerations(&self) -> Vec<P::Vector> {
        let massive = self.massive.iter().map(P::point_mass).collect::<Vec<_>>();
        let massless = self.massless.iter().map(P::point_mass).collect::<Vec<_>>();

        let accelerations = massive.par_iter().chain(&massless).map(|particle1| {
            massive
                .iter()
                .fold(P::Vector::zero_value(), |acceleration, particle2| {
                    let dir = particle2.0 - particle1.0;
                    let mag_2 = dir.norm_sq();

                    let grav_acc = if mag_2 != 0.0 {
                        dir * particle2.1 / (mag_2 * mag_2.sqrt())
                    } else {
                        dir
                    };

                    acceleration + grav_acc
                })
        });

        accelerations.collect()
    }

    /// Iterates over the `massive` [particles](Particle), then the `massless` ones.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &P> {
        self.massive.iter().chain(&self.massless)
    }

    /// Mutably iterates over the `massive` [particles](Particle), then the `massless` ones.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.massive.iter_mut().chain(&mut self.massless)
    }

    /// Returns an iterator over a mutable reference to the [`Particle`] and its computed gravitational acceleration.
    /// # Example
    /// ```
    /// # use particular::prelude::Particle;
    /// # use particular::ParticleSet;
    /// # use glam::Vec3;
    /// #
    /// # const DT: f32 = 1.0 / 60.0;
    /// #
    /// # #[derive(Particle)]
    /// # pub struct Body {
    /// #     position: Vec3,
    /// #     velocity: Vec3,
    /// #     mu: f32,
    /// # }
    /// # let mut particle_set = ParticleSet::<Body>::new();
    /// for (particle, acceleration) in particle_set.result() {
    ///     particle.velocity += acceleration * DT;
    ///     particle.position += particle.velocity * DT;
    /// }
    /// ```
    #[inline]
    pub fn result(&mut self) -> impl Iterator<Item = (&mut P, P::Vector)> {
        let accelerations = self.get_accelerations();
        let particles = self.iter_mut();
        particles.zip(accelerations)
    }
}

#[cfg(test)]
pub mod tests {
    use crate::{NormedVector, Particle, ParticleSet, ToPointMass};
    use glam::Vec3;
    use particular_derive::Particle;

    #[derive(Particle)]
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
        let p1 = (Vec3::new(1.0, 1.0, 1.0), 0.0);
        let p2 = (Vec3::new(-1.0, -1.0, -1.0), 8.0);

        let particle_set = with_two_particles(p1, p2);
        let mut iter = particle_set.iter();

        assert_eq!(p2, iter.next().unwrap().point_mass());
        assert_eq!(p1, iter.next().unwrap().point_mass());
    }

    const EPSILON: f32 = 1E-6;

    #[test]
    fn acceleration_calculation() {
        let p1 = (Vec3::default(), 0.0);
        let p2 = (Vec3::new(1.0, 1.0, 1.0), 3.0);

        let dir = p2.0 - p1.0;
        let mag_2 = dir.norm_sq();
        let grav_acc = dir / (mag_2 * mag_2.sqrt());

        for (particle, acceleration) in with_two_particles(p1, p2).result() {
            if particle.point_mass() == p1 {
                assert!((acceleration - grav_acc * p2.1).norm_sq() < EPSILON);
            } else {
                assert!((acceleration + grav_acc * p1.1).norm_sq() < EPSILON);
            }
        }
    }
}
