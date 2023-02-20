use crate::{
    compute_method::ComputeMethod,
    particle::{Particle, ToPointMass, VectorInternal},
    vector::{Scalar, Vector},
};

/// The structure used to store the [`Particles`](Particle) and calculate their acceleration.
/// 
/// Particles are stored in two vectors, `massive` or `massless`, depending on if they have mass or not.
/// This allows optimizations for objects that are affected by gravitational bodies but don't affect them back, e.g. a spaceship.
///
/// If a particle needs to be removed, it is preferable to create a new [`ParticleSet`] as it is a cheap operation.
pub struct ParticleSet<P: Particle> {
    massive: Vec<P>,
    massless: Vec<P>,
}

impl<P> Default for ParticleSet<P>
where
    P: Particle,
{
    fn default() -> Self {
        Self {
            massive: Vec::new(),
            massless: Vec::new(),
        }
    }
}

impl<P> Clone for ParticleSet<P>
where
    P: Particle + Clone,
{
    fn clone(&self) -> Self {
        Self {
            massive: self.massive.clone(),
            massless: self.massless.clone(),
        }
    }
}

impl<P> ParticleSet<P>
where
    P: Particle,
{
    /// Creates an empty [`ParticleSet`].
    pub fn new() -> Self {
        Self {
            massive: Vec::new(),
            massless: Vec::new(),
        }
    }
}

impl<P> ParticleSet<P>
where
    P: Particle,
    P::Scalar: Scalar,
{
    /// Adds a particle to the [`ParticleSet`].
    ///
    /// This method adds the particle in the corresponding vector depending on its mass.
    pub fn add(&mut self, particle: P) {
        if particle.mu() != <P::Scalar>::default() {
            self.massive.push(particle);
        } else {
            self.massless.push(particle);
        }
    }

    /// Adds a particle that has mass to the [`ParticleSet`].
    ///
    /// Panics if the particle does not have mass.
    pub fn add_massive(&mut self, particle: P) {
        assert!(!(particle.mu() == <P::Scalar>::default()));
        self.massive.push(particle);
    }

    /// Adds a particle that has no mass to the [`ParticleSet`].
    ///
    /// Panics if the particle has mass.
    pub fn add_massless(&mut self, particle: P) {
        assert!(particle.mu() == <P::Scalar>::default());
        self.massless.push(particle);
    }
}

impl<P> ParticleSet<P>
where
    P: Particle,
{
    /// Iterates over the `massive` particles.
    #[inline]
    pub fn massive(&self) -> impl Iterator<Item = &P> {
        self.massive.iter()
    }

    /// Mutably iterates over the `massive` particles.
    #[inline]
    pub fn massive_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.massive.iter_mut()
    }

    /// Iterates over the `massless` particles.
    #[inline]
    pub fn massless(&self) -> impl Iterator<Item = &P> {
        self.massless.iter()
    }

    /// Mutably iterates over the `massless` particles.
    #[inline]
    pub fn massless_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.massless.iter_mut()
    }

    /// Iterates over the `massive` particles, then the `massless` ones.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &P> {
        self.massive.iter().chain(self.massless.iter())
    }

    /// Mutably iterates over the `massive` particles, then the `massless` ones.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.massive.iter_mut().chain(self.massless.iter_mut())
    }

    /// Iterates over the accelerations of the `massive` particles then the `massless` ones computed using the provided [`ComputeMethod`].
    ///
    /// When this method is called, the acceleration of all the particles in the [`ParticleSet`] is computed before iteration.
    #[inline]
    pub fn accelerations<const DIM: usize, C>(&self, cm: &mut C) -> impl Iterator<Item = P::Vector>
    where
        P::Scalar: Scalar,
        P::Vector: Vector<DIM, P::Scalar>,
        C: ComputeMethod<VectorInternal<DIM, P>, P::Scalar>,
    {
        self.compute(cm).into_iter().map(Vector::from_internal)
    }

    /// Returns an iterator over a reference to a particle and its computed gravitational acceleration using the provided [`ComputeMethod`].
    /// 
    /// Useful when you want to modify the state of an external object using the computed acceleration without modifying its particle representation at the same time.
    ///
    /// It is equivalent to:
    /// ```ignore
    /// particle_set
    ///     .accelerations(cm)
    ///     .zip(particle_set.iter());
    /// ```
    /// # Example
    /// ```
    /// # use particular::prelude::*;
    /// # use glam::Vec3;
    /// #
    /// # const DT: f32 = 1.0 / 60.0;
    /// #
    /// # #[derive(Particle)]
    /// # struct Body {
    /// #     id: u32,
    /// #     position: Vec3,
    /// #     velocity: Vec3,
    /// #     mu: f32,
    /// # }
    /// # let mut particle_set = ParticleSet::<Body>::new();
    /// # let mut game_query = std::collections::HashMap::<u32, Vec3>::new();
    /// let cm = &mut sequential::BruteForce;
    ///
    /// for (acceleration, particle) in particle_set.result(cm) {
    ///     if let Some(rb_acceleration) = game_query.get_mut(&particle.id) {
    ///         *rb_acceleration = acceleration;
    ///     }
    /// }
    /// ```
    pub fn result<const DIM: usize, C>(
        &mut self,
        cm: &mut C,
    ) -> impl Iterator<Item = (P::Vector, &P)>
    where
        P::Scalar: Scalar,
        P::Vector: Vector<DIM, P::Scalar>,
        C: ComputeMethod<VectorInternal<DIM, P>, P::Scalar>,
    {
        self.accelerations(cm).zip(self.iter())
    }

    /// Returns an iterator over a mutable reference to a particle and its computed gravitational acceleration using the provided [`ComputeMethod`].
    /// 
    /// Useful when you want to modify a particle using the computed acceleration.
    ///
    /// It is equivalent to:
    /// ```ignore
    /// particle_set
    ///     .accelerations(cm)
    ///     .zip(particle_set.iter_mut());
    /// ```
    /// # Example
    /// ```
    /// # use particular::prelude::*;
    /// # use glam::Vec3;
    /// #
    /// # const DT: f32 = 1.0 / 60.0;
    /// #
    /// # #[derive(Particle)]
    /// # struct Body {
    /// #     position: Vec3,
    /// #     velocity: Vec3,
    /// #     mu: f32,
    /// # }
    /// # let mut particle_set = ParticleSet::<Body>::new();
    /// let cm = &mut sequential::BruteForce;
    ///
    /// for (acceleration, particle) in particle_set.result_mut(cm) {
    ///     particle.velocity += acceleration * DT;
    ///     particle.position += particle.velocity * DT;
    /// }
    /// ```
    pub fn result_mut<const DIM: usize, C>(
        &mut self,
        cm: &mut C,
    ) -> impl Iterator<Item = (P::Vector, &mut P)>
    where
        P::Scalar: Scalar,
        P::Vector: Vector<DIM, P::Scalar>,
        C: ComputeMethod<VectorInternal<DIM, P>, P::Scalar>,
    {
        self.accelerations(cm).zip(self.iter_mut())
    }
}

trait ComputeParticleSet<const DIM: usize, P>
where
    P: Particle,
    P::Scalar: Scalar,
    P::Vector: Vector<DIM, P::Scalar>,
{
    fn massive_point_masses(&self) -> Vec<(VectorInternal<DIM, P>, P::Scalar)>;

    fn massless_point_masses(&self) -> Vec<(VectorInternal<DIM, P>, P::Scalar)>;

    #[inline]
    fn compute<C>(&self, cm: &mut C) -> Vec<VectorInternal<DIM, P>>
    where
        C: ComputeMethod<VectorInternal<DIM, P>, P::Scalar>,
    {
        cm.compute(self.massive_point_masses(), self.massless_point_masses())
    }
}

impl<const DIM: usize, P> ComputeParticleSet<DIM, P> for ParticleSet<P>
where
    P: Particle,
    P::Scalar: Scalar,
    P::Vector: Vector<DIM, P::Scalar>,
{
    #[inline]
    fn massive_point_masses(&self) -> Vec<(VectorInternal<DIM, P>, P::Scalar)> {
        self.massive().map(P::point_mass).collect()
    }

    #[inline]
    fn massless_point_masses(&self) -> Vec<(VectorInternal<DIM, P>, P::Scalar)> {
        self.massless().map(P::point_mass).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use glam::Vec3;

    // #[particle(3)]
    pub struct Body {
        position: Vec3,
        mu: f32,
    }

    impl Particle for Body {
        type Scalar = f32;
        type Vector = Vec3;

        fn position(&self) -> Vec3 {
            self.position
        }

        fn mu(&self) -> f32 {
            self.mu
        }
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

        assert_eq!(p2.1, iter.next().unwrap().mu);
        assert_eq!(p1.1, iter.next().unwrap().mu);
    }
}
