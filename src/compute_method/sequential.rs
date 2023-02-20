use crate::compute_method::ComputeMethod;

/// A brute-force [`ComputeMethod`] using the CPU.
pub struct BruteForce;

macro_rules! impl_brute_force {
    ($scalar: ty, $internal: ty) => {
        impl ComputeMethod<$internal, $scalar> for BruteForce {
            #[inline]
            fn compute(
                &mut self,
                massive: Vec<($internal, $scalar)>,
                massless: Vec<($internal, $scalar)>,
            ) -> Vec<$internal> {
                let massive_len = massive.len();

                let particles = &[massive, massless].concat()[..];
                let len = particles.len();

                let mut accelerations = vec![<$internal>::ZERO; len];

                for i in 0..massive_len {
                    let (pos1, mu1) = particles[i];
                    let mut acceleration = <$internal>::ZERO;

                    for j in (i + 1)..len {
                        let (pos2, mu2) = particles[j];

                        let dir = pos2 - pos1;
                        let mag_2 = dir.length_squared();

                        let f = dir / (mag_2 * mag_2.sqrt());

                        acceleration += f * mu2;
                        accelerations[j] -= f * mu1;
                    }

                    accelerations[i] += acceleration;
                }

                accelerations
            }
        }
    };
}

impl_brute_force!(f32, glam::Vec3A);
impl_brute_force!(f32, glam::Vec4);

impl_brute_force!(f64, glam::DVec2);
impl_brute_force!(f64, glam::DVec3);
impl_brute_force!(f64, glam::DVec4);

#[cfg(test)]
mod tests {
    use crate::compute_method::{sequential, tests};

    #[test]
    fn brute_force() {
        tests::acceleration_computation(sequential::BruteForce);
    }
}
