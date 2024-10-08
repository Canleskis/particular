/// Extension of the `glam` crate adding vectors that use SIMD types from the `wide` crate.
pub mod wide_glam {
    pub use glam::{DVec2, DVec3, DVec4, Vec2, Vec3, Vec3A, Vec4};
    pub use wide::{f32x4, f32x8, f64x2, f64x4};

    macro_rules! impl_from_lanes {
        ([$vector: ty; 2] => $simd: ty, { $($f: ident),* }) => {
            impl From<[$vector; 2]> for $simd {
                #[inline]
                fn from(vecs: [$vector; 2]) -> Self {
                    Self {
                        $($f: From::from([vecs[0].$f, vecs[1].$f]),)*
                    }
                }
            }
        };

        ([$vector: ty; 4] => $simd: ty, { $($f: ident),* }) => {
            impl From<[$vector; 4]> for $simd {
                #[inline]
                fn from(vecs: [$vector; 4]) -> Self {
                    Self {
                        $($f: From::from([vecs[0].$f, vecs[1].$f, vecs[2].$f, vecs[3].$f]),)*
                    }
                }
            }
        };
        ([$vector: ty; 8] => $simd: ty, { $($f: ident),* }) => {
            impl From<[$vector; 8]> for $simd {
                #[inline]
                fn from(vecs: [$vector; 8]) -> Self {
                    Self {
                        $($f: From::from([
                            vecs[0].$f, vecs[1].$f, vecs[2].$f, vecs[3].$f, vecs[4].$f,
                            vecs[5].$f, vecs[6].$f, vecs[7].$f
                        ]),)*
                    }
                }
            }
        };
    }

    macro_rules! vecNxL {
        ([$vector: ty; $lane: tt] => $name: ident, $scalar: ty, { $($f: ident),* }: $dim: literal) => {
            #[repr(C)]
            #[derive(Clone, Copy, Debug, Default, PartialEq)]
            #[doc = concat!("A vector of ", $dim, "x", $lane, " elements.")]
            #[allow(missing_docs)]
            pub struct $name {
                $(pub $f: $scalar,)*
            }

            impl $name {
                /// Creates a new vector.
                #[inline]
                pub const fn new($($f: $scalar,)*) -> Self {
                    Self { $($f,)* }
                }

                /// Creates a vector with all elements set to `v`.
                #[inline]
                pub fn splat(v: $vector) -> Self {
                    Self { $($f: <$scalar>::splat(v.$f),)* }
                }

                /// Computes the squared length of self.
                #[inline]
                pub fn length_squared(self) -> $scalar {
                    $(self.$f * self.$f +)* 0.0
                }
            }

            impl std::ops::Add for $name {
                type Output = Self;

                #[inline]
                fn add(self, rhs: Self) -> Self {
                    Self { $($f: self.$f + rhs.$f,)* }
                }
            }

            impl std::ops::AddAssign for $name {
                #[inline]
                fn add_assign(&mut self, rhs: Self) {
                    $(self.$f += rhs.$f;)*
                }
            }

            impl std::ops::Sub for $name {
                type Output = Self;

                #[inline]
                fn sub(self, rhs: Self) -> Self {
                    Self { $($f: self.$f - rhs.$f,)* }
                }
            }

            impl std::ops::SubAssign for $name {
                #[inline]
                fn sub_assign(&mut self, rhs: Self) {
                    $(self.$f -= rhs.$f;)*
                }
            }

            impl std::ops::Mul<$scalar> for $name {
                type Output = Self;

                #[inline]
                fn mul(self, rhs: $scalar) -> Self {
                    Self { $($f: self.$f * rhs,)* }
                }
            }

            impl std::ops::Div<$scalar> for $name {
                type Output = Self;

                #[inline]
                fn div(self, rhs: $scalar) -> Self {
                    Self { $($f: self.$f / rhs,)* }
                }
            }

            impl From<$name> for [$scalar; $dim] {
                #[inline]
                fn from(vec: $name) -> Self {
                    [$(vec.$f,)*]
                }
            }

            impl_from_lanes!([$vector; $lane] => $name, { $($f),* });
        }
    }

    vecNxL!([Vec2; 4] => Vec2x4, f32x4, { x, y }: 2);
    vecNxL!([Vec2; 8] => Vec2x8, f32x8, { x, y }: 2);
    vecNxL!([Vec3; 4] => Vec3x4, f32x4, { x, y, z }: 3);
    vecNxL!([Vec3; 8] => Vec3x8, f32x8, { x, y, z }: 3);
    // Vec3AxL needed to return Vec3A for SIMD algorithms even though it is identical to Vec3xL.
    vecNxL!([Vec3A; 4] => Vec3Ax4, f32x4, { x, y, z }: 3);
    vecNxL!([Vec3A; 8] => Vec3Ax8, f32x8, { x, y, z }: 3);
    vecNxL!([Vec4; 4] => Vec4x4, f32x4, { x, y, z, w }: 4);
    vecNxL!([Vec4; 8] => Vec4x8, f32x8, { x, y, z, w }: 4);
    vecNxL!([DVec2; 2] => DVec2x2, f64x2, { x, y }: 2);
    vecNxL!([DVec2; 4] => DVec2x4, f64x4, { x, y }: 2);
    vecNxL!([DVec3; 2] => DVec3x2, f64x2, { x, y, z }: 3);
    vecNxL!([DVec3; 4] => DVec3x4, f64x4, { x, y, z }: 3);
    vecNxL!([DVec4; 2] => DVec4x2, f64x2, { x, y, z, w }: 4);
    vecNxL!([DVec4; 4] => DVec4x4, f64x4, { x, y, z, w }: 4);
}

use wide_glam::{
    f32x4, f32x8, f64x2, f64x4, DVec2, DVec2x2, DVec2x4, DVec3, DVec3x2, DVec3x4, DVec4, DVec4x2,
    DVec4x4, Vec2, Vec2x4, Vec2x8, Vec3, Vec3A, Vec3Ax4, Vec3Ax8, Vec3x4, Vec3x8, Vec4, Vec4x4,
    Vec4x8,
};

crate::impl_into_array!([f32; 2], Vec2);
crate::impl_into_array!([f32; 3], Vec3);
crate::impl_into_array!([f32; 3], Vec3A);
crate::impl_into_array!([f32; 4], Vec4);
crate::impl_into_array!([f64; 2], DVec2);
crate::impl_into_array!([f64; 3], DVec3);
crate::impl_into_array!([f64; 4], DVec4);
crate::impl_into_array!([f32x4; 2], Vec2x4);
crate::impl_into_array!([f32x8; 2], Vec2x8);
crate::impl_into_array!([f32x4; 3], Vec3x4);
crate::impl_into_array!([f32x8; 3], Vec3x8);
crate::impl_into_array!([f32x4; 3], Vec3Ax4);
crate::impl_into_array!([f32x8; 3], Vec3Ax8);
crate::impl_into_array!([f32x4; 4], Vec4x4);
crate::impl_into_array!([f32x8; 4], Vec4x8);
crate::impl_into_array!([f64x2; 2], DVec2x2);
crate::impl_into_array!([f64x4; 2], DVec2x4);
crate::impl_into_array!([f64x2; 3], DVec3x2);
crate::impl_into_array!([f64x4; 3], DVec3x4);
crate::impl_into_array!([f64x2; 4], DVec4x2);
crate::impl_into_array!([f64x4; 4], DVec4x4);

crate::impl_norm!(Vec2, f32, Vec2::length_squared);
crate::impl_norm!(Vec3, f32, Vec3::length_squared);
crate::impl_norm!(Vec3A, f32, Vec3A::length_squared);
crate::impl_norm!(Vec4, f32, Vec4::length_squared);
crate::impl_norm!(DVec2, f64, DVec2::length_squared);
crate::impl_norm!(DVec3, f64, DVec3::length_squared);
crate::impl_norm!(DVec4, f64, DVec4::length_squared);
crate::impl_norm!(Vec2x4, f32x4, Vec2x4::length_squared);
crate::impl_norm!(Vec2x8, f32x8, Vec2x8::length_squared);
crate::impl_norm!(Vec3x4, f32x4, Vec3x4::length_squared);
crate::impl_norm!(Vec3x8, f32x8, Vec3x8::length_squared);
crate::impl_norm!(Vec3Ax4, f32x4, Vec3Ax4::length_squared);
crate::impl_norm!(Vec3Ax8, f32x8, Vec3Ax8::length_squared);
crate::impl_norm!(Vec4x4, f32x4, Vec4x4::length_squared);
crate::impl_norm!(Vec4x8, f32x8, Vec4x8::length_squared);
crate::impl_norm!(DVec2x2, f64x2, DVec2x2::length_squared);
crate::impl_norm!(DVec2x4, f64x4, DVec2x4::length_squared);
crate::impl_norm!(DVec3x2, f64x2, DVec3x2::length_squared);
crate::impl_norm!(DVec3x4, f64x4, DVec3x4::length_squared);
crate::impl_norm!(DVec4x2, f64x2, DVec4x2::length_squared);
crate::impl_norm!(DVec4x4, f64x4, DVec4x4::length_squared);

crate::impl_reduce!([<Vec2, f32>; 4], Vec2x4 => reduce_add(x, y));
crate::impl_reduce!([<Vec2, f32>; 8], Vec2x8 => reduce_add(x, y));
crate::impl_reduce!([<Vec3, f32>; 4], Vec3x4 => reduce_add(x, y, z));
crate::impl_reduce!([<Vec3, f32>; 8], Vec3x8 => reduce_add(x, y, z));
crate::impl_reduce!([<Vec3A, f32>; 4], Vec3Ax4 => reduce_add(x, y, z));
crate::impl_reduce!([<Vec3A, f32>; 8], Vec3Ax8 => reduce_add(x, y, z));
crate::impl_reduce!([<Vec4, f32>; 4], Vec4x4 => reduce_add(x, y, z, w));
crate::impl_reduce!([<Vec4, f32>; 8], Vec4x8 => reduce_add(x, y, z, w));
crate::impl_reduce!([<DVec2, f64>; 2], DVec2x2 => reduce_add(x, y));
crate::impl_reduce!([<DVec2, f64>; 4], DVec2x4 => reduce_add(x, y));
crate::impl_reduce!([<DVec3, f64>; 2], DVec3x2 => reduce_add(x, y, z));
crate::impl_reduce!([<DVec3, f64>; 4], DVec3x4 => reduce_add(x, y, z));
crate::impl_reduce!([<DVec4, f64>; 2], DVec4x2 => reduce_add(x, y, z, w));
crate::impl_reduce!([<DVec4, f64>; 4], DVec4x4 => reduce_add(x, y, z, w));

crate::impl_to_simd!([<Vec2, f32>; 4] => <Vec2x4, f32x4>);
crate::impl_to_simd!([<Vec2, f32>; 8] => <Vec2x8, f32x8>);
crate::impl_to_simd!([<Vec3, f32>; 4] => <Vec3x4, f32x4>);
crate::impl_to_simd!([<Vec3, f32>; 8] => <Vec3x8, f32x8>);
crate::impl_to_simd!([<Vec3A, f32>; 4] => <Vec3Ax4, f32x4>);
crate::impl_to_simd!([<Vec3A, f32>; 8] => <Vec3Ax8, f32x8>);
crate::impl_to_simd!([<Vec4, f32>; 4] => <Vec4x4, f32x4>);
crate::impl_to_simd!([<Vec4, f32>; 8] => <Vec4x8, f32x8>);
crate::impl_to_simd!([<DVec2, f64>; 2] => <DVec2x2, f64x2>);
crate::impl_to_simd!([<DVec2, f64>; 4] => <DVec2x4, f64x4>);
crate::impl_to_simd!([<DVec3, f64>; 2] => <DVec3x2, f64x2>);
crate::impl_to_simd!([<DVec3, f64>; 4] => <DVec3x4, f64x4>);
crate::impl_to_simd!([<DVec4, f64>; 2] => <DVec4x2, f64x2>);
crate::impl_to_simd!([<DVec4, f64>; 4] => <DVec4x4, f64x4>);

crate::impl_acceleration_cpu_scalar!(Vec2, f32);
crate::impl_acceleration_cpu_scalar!(Vec3, f32);
crate::impl_acceleration_cpu_scalar!(DVec2, f64);
crate::impl_acceleration_cpu_scalar!(DVec3, f64);
crate::impl_acceleration_cpu_scalar!(DVec4, f64);

#[cfg(not(any(
    target_arch = "aarch64",
    target_feature = "sse2",
    target_feature = "simd128"
)))]
mod scalar_vectors {
    use super::*;
    crate::impl_acceleration_cpu_scalar!(Vec3A, f32);
    crate::impl_acceleration_cpu_scalar!(Vec4, f32);
}

macro_rules! impl_acceleration_cpu_scalar {
    ($vector: ty, $scalar: ty) => {
        impl crate::gravity::newtonian::AccelerationAt
            for crate::gravity::GravitationalField<$vector, $scalar>
        {
            type Vector = $vector;

            type Softening = $scalar;

            type Output = $vector;

            #[inline]
            fn acceleration_at<const CHECKED: bool>(
                &self,
                at: &Self::Vector,
                softening: &Self::Softening,
            ) -> Self::Output {
                let d = self.position - *at;
                let norm = crate::gravity::Norm::norm_squared(d);

                // Branch removed by the compiler when `CHECKED` is false.
                if CHECKED && norm == 0.0 {
                    d
                } else {
                    let norm_s = norm + (*softening * *softening);
                    d * self.m / (norm_s * norm_s.sqrt())
                }
            }
        }

        crate::impl_acceleration_paired_cpu_scalar!($vector, $scalar);
    };
}

#[cfg(any(
    target_arch = "aarch64",
    target_feature = "sse2",
    target_feature = "simd128"
))]
mod simd_vectors {
    use super::*;
    impl_acceleration_cpu_scalar!(Vec3A, f32);
    impl_acceleration_cpu_scalar!(Vec4, f32);
}

crate::impl_acceleration_cpu_simd!(Vec2x4, f32x4, f32x4::recip_sqrt);
crate::impl_acceleration_cpu_simd!(Vec2x8, f32x8, f32x8::recip_sqrt);
crate::impl_acceleration_cpu_simd!(Vec3x4, f32x4, f32x4::recip_sqrt);
crate::impl_acceleration_cpu_simd!(Vec3x8, f32x8, f32x8::recip_sqrt);
crate::impl_acceleration_cpu_simd!(Vec3Ax4, f32x4, f32x4::recip_sqrt);
crate::impl_acceleration_cpu_simd!(Vec3Ax8, f32x8, f32x8::recip_sqrt);
crate::impl_acceleration_cpu_simd!(Vec4x4, f32x4, f32x4::recip_sqrt);
crate::impl_acceleration_cpu_simd!(Vec4x8, f32x8, f32x8::recip_sqrt);
crate::impl_acceleration_cpu_simd!(DVec2x2, f64x2, |f: f64x2| 1.0 / f.sqrt());
crate::impl_acceleration_cpu_simd!(DVec2x4, f64x4, |f: f64x4| 1.0 / f.sqrt());
crate::impl_acceleration_cpu_simd!(DVec3x2, f64x2, |f: f64x2| 1.0 / f.sqrt());
crate::impl_acceleration_cpu_simd!(DVec3x4, f64x4, |f: f64x4| 1.0 / f.sqrt());
crate::impl_acceleration_cpu_simd!(DVec4x2, f64x2, |f: f64x2| 1.0 / f.sqrt());
crate::impl_acceleration_cpu_simd!(DVec4x4, f64x4, |f: f64x4| 1.0 / f.sqrt());

#[cfg(feature = "gpu")]
crate::impl_acceleration_gpu_2d!(Vec2, f32);
#[cfg(feature = "gpu")]
crate::impl_acceleration_gpu_3d!(Vec3, f32);

#[cfg(feature = "gpu")]
impl<const CHECKED: bool> crate::gravity::newtonian::AccelerationGPU<CHECKED>
    for crate::gravity::GravitationalField<glam::Vec3A, f32>
{
    type Vector = glam::Vec3A;

    type GPUVector = crate::gravity::padded::Vector<4, glam::Vec3>;

    type GPUAffecting = crate::gravity::padded::GravitationalField<0, glam::Vec3, f32>;

    type GPUOutput = crate::gravity::padded::Vector<4, glam::Vec3>;

    type Output = glam::Vec3A;

    const SOURCE: &'static str = crate::gravity::impls::AccelerationShader::<3, CHECKED>::SOURCE;

    const SOURCE_SOFTENED: &'static str =
        crate::gravity::impls::AccelerationShader::<3, CHECKED>::SOURCE_SOFTENED;

    #[inline]
    fn to_gpu_position(position: &Self::Vector) -> Self::GPUVector {
        glam::Vec3::from(*position).into()
    }

    #[inline]
    fn to_gpu_particle(&self) -> Self::GPUAffecting {
        crate::gravity::GravitationalField::new(self.position.into(), self.m).into()
    }

    #[inline]
    fn to_cpu_output(output: &Self::GPUOutput) -> Self::Output {
        output.vector.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Div;

    crate::tests_algorithms!(Vec2, f32, splat, Div::div, [4, 8], vec2 on cpu and gpu);
    crate::tests_algorithms!(Vec3, f32, splat, Div::div, [4, 8], vec3 on cpu and gpu);
    crate::tests_algorithms!(Vec3A, f32, splat, Div::div, [4, 8], vec3a on cpu and gpu);
    crate::tests_algorithms!(Vec4, f32, splat, Div::div, [4, 8], vec4 on cpu);
    crate::tests_algorithms!(DVec2, f64, splat, Div::div, [2, 4], dvec2 on cpu);
    crate::tests_algorithms!(DVec3, f64, splat, Div::div, [2, 4], dvec3 on cpu);
    crate::tests_algorithms!(DVec4, f64, splat, Div::div, [2, 4], dvec4 on cpu);
}
