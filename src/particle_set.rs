use crate::{
    compute_method::ComputeMethod,
    particle::{Particle, ToPointMass, VectorInternal},
    vector::{Scalar, Vector},
};

/// The structure used to store the [`Particles`](Particle) and calculate their acceleration.
/// [`Particles`](Particle) are stored in two vectors, `massive` or `massless`, depending on if they have mass or not.
/// This allows optimizations for objects that are affected by gravitational bodies but don't affect them back, e.g. a spaceship.
/// 
/// If a [`Particle`] needs to be removed, it is preferable to create a new [`ParticleSet`] as it is a cheap operation.
#[derive(Default)]
pub struct ParticleSet<P: Particle> {
    massive: Vec<P>,
    massless: Vec<P>,
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
    /// Adds a [`Particle`] to the [`ParticleSet`].
    ///
    /// This method adds the [`Particle`] in the corresponding vector depending on its mass.
    pub fn add(&mut self, particle: P) {
        if particle.mu() != <P::Scalar>::default() {
            self.massive.push(particle);
        } else {
            self.massless.push(particle);
        }
    }

    /// Adds a [`Particle`] that has mass to the [`ParticleSet`].
    ///
    /// Panics if the [`Particle`] doesn't have mass.
    pub fn add_massive(&mut self, particle: P) {
        assert!(!(particle.mu() == <P::Scalar>::default()));
        self.massive.push(particle);
    }

    /// Adds a [`Particle`] that has no mass to the [`ParticleSet`].
    ///
    /// Panics if the [`Particle`] has mass.
    pub fn add_massless(&mut self, particle: P) {
        assert!(particle.mu() == <P::Scalar>::default());
        self.massless.push(particle);
    }
}

impl<P> ParticleSet<P>
where
    P: Particle,
{
    /// Iterates over the `massive` [`Particles`](Particle).
    #[inline]
    pub fn massive(&self) -> impl Iterator<Item = &P> {
        self.massive.iter()
    }

    /// Mutably iterates over the `massive` [`Particles`](Particle).
    #[inline]
    pub fn massive_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.massive.iter_mut()
    }

    /// Iterates over the `massless` [`Particles`](Particle).
    #[inline]
    pub fn massless(&self) -> impl Iterator<Item = &P> {
        self.massless.iter()
    }

    /// Mutably iterates over the `massless` [`Particles`](Particle).
    #[inline]
    pub fn massless_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.massless.iter_mut()
    }

    /// Iterates over the `massive` [`Particles`](Particle), then the `massless` ones.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &P> {
        self.massive.iter().chain(self.massless.iter())
    }

    /// Mutably iterates over the `massive` [`Particles`](Particle), then the `massless` ones.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.massive.iter_mut().chain(self.massless.iter_mut())
    }

    /// Iterates over the accelerations of the `massive` [`Particles`](Particle) then the `massless` ones computed using the provided [`ComputeMethod`].
    ///
    /// When this method is called, the acceleration of all the [`Particles`](Particle) in the [`ParticleSet`] is computed before iteration.
    #[inline]
    pub fn accelerations<const DIM: usize, C>(&self, cm: &mut C) -> impl Iterator<Item = P::Vector>
    where
        P::Scalar: Scalar,
        P::Vector: Vector<DIM, P::Scalar>,
        C: ComputeMethod<VectorInternal<DIM, P>, P::Scalar>,
    {
        self.compute(cm).into_iter().map(Vector::from_internal)
    }

    /// Returns an iterator over a mutable reference to a [`Particle`] and its computed gravitational acceleration using the provided [`ComputeMethod`].
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
    /// # pub struct Body {
    /// #     position: Vec3,
    /// #     velocity: Vec3,
    /// #     mu: f32,
    /// # }
    /// # let mut particle_set = ParticleSet::<Body>::new();
    /// let cm = &mut sequential::BruteForce;
    /// 
    /// for (acceleration, particle) in particle_set.result(cm) {
    ///     particle.velocity += acceleration * DT;
    ///     particle.position += particle.velocity * DT;
    /// }
    /// ```
    pub fn result<const DIM: usize, C>(
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
