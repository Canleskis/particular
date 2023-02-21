#[cfg(feature = "gpu")]
/// Compute methods that use the GPU.
pub mod gpu;

#[cfg(feature = "parallel")]
/// Compute methods that use multiple CPU threads.
pub mod parallel;

/// Compute methods that use one CPU thread.
pub mod sequential;

/// Trait for algorithms computing the gravitational forces between [`Particles`](crate::particle::Particle).
///
/// To implement it, specify an internal vector representation and its scalar type.
///
/// # Example
///
/// ```
/// # use particular::prelude::*;
/// # use glam::Vec3A;
/// struct AccelerationCalculator;
///
/// impl ComputeMethod<Vec3A, f32> for AccelerationCalculator {
///     fn compute(&mut self, massive: Vec<(Vec3A, f32)>, massless: Vec<(Vec3A, f32)>) -> Vec<Vec3A> {
///     // ...
/// #       Vec::new()
///     }
/// }
/// ```
pub trait ComputeMethod<V, U> {
    /// Computes the acceleration the massive particles exert on all the particles.
    ///
    /// The returning vector should contain the acceleration of the massive particles first, then the massless ones, in the same order they were input.
    fn compute(&mut self, massive: Vec<(V, U)>, massless: Vec<(V, U)>) -> Vec<V>;
}

trait Computable<V, U> {
    fn compute<F, G>(massive: Vec<(V, U)>, massless: Vec<(V, U)>, mag_sq: F, sqrt: G) -> Vec<V>
    where
        F: Fn(V) -> U + Sync,
        G: Fn(U) -> U + Sync;
}

macro_rules! computable {
    ($s: ty, $i: ty) => {
        impl<C> ComputeMethod<$i, $s> for C
        where
            C: Computable<$i, $s>,
        {
            #[inline]
            fn compute(&mut self, massive: Vec<($i, $s)>, massless: Vec<($i, $s)>) -> Vec<$i> {
                C::compute(massive, massless, <$i>::length_squared, <$s>::sqrt)
            }
        }
    };
}

computable!(f32, glam::Vec3A);
computable!(f32, glam::Vec4);

computable!(f64, glam::DVec2);
computable!(f64, glam::DVec3);
computable!(f64, glam::DVec4);

#[cfg(test)]
pub(crate) mod tests {
    use crate::prelude::*;
    use glam::Vec3A;

    pub fn acceleration_computation<C>(mut cm: C)
    where
        C: ComputeMethod<Vec3A, f32>,
    {
        let massive = vec![(Vec3A::splat(0.0), 2.0), (Vec3A::splat(1.0), 3.0)];
        let massless = vec![(Vec3A::splat(5.0), 0.0)];

        let computed = cm.compute(massive.clone(), massless.clone());

        for (&point_mass1, computed) in massive.iter().chain(massless.iter()).zip(computed) {
            let mut acceleration = Vec3A::ZERO;

            for &point_mass2 in massive.iter() {
                let dir = point_mass2.0 - point_mass1.0;
                let mag_2 = dir.length_squared();

                if mag_2 != 0.0 {
                    acceleration += dir * point_mass2.1 / (mag_2 * mag_2.sqrt());
                }
            }

            assert!((acceleration).abs_diff_eq(computed, f32::EPSILON))
        }
    }
}
