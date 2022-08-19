use glam::Vec3;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::particle::{Particle, ToPointMass};

#[derive(Default)]
pub struct ParticleSet<P: Particle + Sync> {
    massive: Vec<P>,
    massless: Vec<P>,
}

impl<P: Particle + Sync> ParticleSet<P> {
    pub fn new() -> Self {
        Self {
            massive: Vec::new(),
            massless: Vec::new(),
        }
    }
}

impl<P: Particle + Sync> ParticleSet<P> {
    pub fn add(&mut self, particle: P) {
        if particle.mu() == 0.0 {
            self.massless.push(particle);
        } else {
            self.massive.push(particle);
        }
    }
}

impl<P: Particle + Sync> ParticleSet<P> {
    fn get_accelerations(&self) -> Vec<Vec3> {
        let massive = self.massive.iter().map(P::to_point_mass).collect::<Vec<_>>();
        let massless = self.massless.iter().map(P::to_point_mass).collect::<Vec<_>>();

        let accelerations = massive.par_iter().chain(&massless).map(|particle1| {
            massive.iter().fold(Vec3::ZERO, |acceleration, particle2| {
                let dir = particle2.0 - particle1.0;
                let mag_2 = dir.length_squared();

                let grav_acc = if mag_2 != 0.0 {
                    particle2.1 * dir / (mag_2 * mag_2.sqrt())
                } else {
                    dir
                };

                acceleration + grav_acc
            })
        });

        accelerations.collect()
    }

    /// Iterates over the `massive` particles, then the `massless` ones.
    pub fn iter(&self) -> impl Iterator<Item = &P> {
        self.massive.iter().chain(&self.massless)
    }

    /// Mutably iterates over the `massive` particles, then the `massless` ones.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.massive.iter_mut().chain(&mut self.massless)
    }

    /// Returns a vector holding tuples of a mutable reference to a [`Particle`]-like object and its computed gravitational acceleration.
    /// ```
    /// # use particular::prelude::Particle;
    /// # use particular::ParticleSet;
    /// # use glam::Vec3;
    /// # 
    /// # const dt: f32 = 1.0 / 60.0;
    /// #
    /// # #[derive(Particle)]
    /// # pub struct Body {
    /// #     position: Vec3,
    /// #     velocity: Vec3,
    /// #     mu: f32,
    /// # }
    /// # let mut particle_set = ParticleSet::<Body>::new();
    /// for (particle, acceleration) in particle_set.result() {
    ///     particle.velocity += acceleration * dt;
    ///     particle.position += particle.velocity * dt;
    /// }
    /// ```
    pub fn result(&mut self) -> Vec<(&mut P, Vec3)> {
        let accelerations = self.get_accelerations();
        let particles = self.iter_mut();
        particles.zip(accelerations).collect()
    }
}
