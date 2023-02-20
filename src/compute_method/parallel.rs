use crate::compute_method::ComputeMethod;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// A brute-force [`ComputeMethod`] using the CPU with [rayon](https://github.com/rayon-rs/rayon).
pub struct BruteForce;

macro_rules! impl_brute_force {
    ($scalar: ty, $internal: ty) => {
        impl ComputeMethod<$internal, $scalar> for BruteForce {
            #[inline]
            fn compute(&mut self, massive: Vec<($internal, $scalar)>, massless: Vec<($internal, $scalar)>) -> Vec<$internal> {
                let massive_len = massive.len();
        
                let particles = &[massive, massless].concat()[..];
                let len = particles.len();
        
                (0..len)
                    .into_par_iter()
                    .map(|i| {
                        let mut acceleration = <$internal>::ZERO;
        
                        for j in 0..massive_len {
                            if i != j {
                                let dir = particles[j].0 - particles[i].0;
                                let mag_2 = dir.length_squared();
        
                                acceleration += dir * particles[j].1 / (mag_2 * mag_2.sqrt())
                            }
                        }
        
                        acceleration
                    })
                    .collect()
            }
        }
    }
}

impl_brute_force!(f32, glam::Vec3A);
impl_brute_force!(f32, glam::Vec4);

impl_brute_force!(f64, glam::DVec2);
impl_brute_force!(f64, glam::DVec3);
impl_brute_force!(f64, glam::DVec4);

#[cfg(test)]
mod tests {
    use crate::compute_method::{parallel, tests};

    #[test]
    fn brute_force() {
        tests::acceleration_computation(parallel::BruteForce);
    }
}
