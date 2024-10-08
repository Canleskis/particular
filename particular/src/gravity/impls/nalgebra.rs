use nalgebra::{SVector, SimdPartialOrd, SimdValue};
use simba::simd::{WideF32x4, WideF32x8, WideF64x4};

impl<const D: usize, S: nalgebra::Scalar> crate::gravity::IntoArray for SVector<S, D> {
    type Array = [S; D];
}

impl<const D: usize, S> crate::gravity::Norm for SVector<S, D>
where
    S: nalgebra::ComplexField,
{
    type Output = S::RealField;

    #[inline]
    fn norm_squared(self) -> S::RealField {
        nalgebra::Normed::norm_squared(&self)
    }
}

impl<const D: usize, S> crate::gravity::Reduce for SVector<S, D>
where
    S: nalgebra::SimdComplexField,
    S::Element: nalgebra::Scalar,
{
    type Output = SVector<S::Element, D>;

    #[inline]
    fn reduce_sum(self) -> Self::Output {
        self.map(|v| v.simd_horizontal_sum())
    }
}

macro_rules! impl_to_simd {
    ([$scalar: ty; $lane: literal] => $simd_scalar: ty) => {
        impl<const D: usize> crate::gravity::newtonian::ToSimd<$lane>
            for crate::gravity::GravitationalField<SVector<$scalar, D>, $scalar>
        where
            SVector<$scalar, D>: Default,
        {
            type Simd = crate::gravity::GravitationalField<SVector<$simd_scalar, D>, $simd_scalar>;

            #[inline]
            fn lanes_splat(&self) -> Self::Simd {
                crate::gravity::GravitationalField::new(
                    SVector::splat(self.position),
                    <$simd_scalar>::splat(self.m),
                )
            }

            #[inline]
            fn lanes_array(array: [Self; $lane]) -> Self::Simd {
                crate::gravity::GravitationalField::new(
                    SVector::from(array.map(|p| p.position)),
                    <$simd_scalar>::from(array.map(|p| p.m)),
                )
            }

            #[inline]
            fn lanes_slice(slice: &[Self]) -> Self::Simd {
                crate::gravity::newtonian::ToSimd::<$lane>::lanes_array(std::array::from_fn(|i| {
                    slice.get(i).copied().unwrap_or_default()
                }))
            }
        }
    };
}

impl_to_simd!([f32; 4] => WideF32x4);
impl_to_simd!([f32; 8] => WideF32x8);
impl_to_simd!([f64; 4] => WideF64x4);

macro_rules! impl_acceleration_cpu_scalar {
    ($scalar: ty) => {
        impl<const D: usize> crate::gravity::newtonian::AccelerationAt
            for crate::gravity::GravitationalField<SVector<$scalar, D>, $scalar>
        {
            type Vector = SVector<$scalar, D>;

            type Softening = $scalar;

            type Output = SVector<$scalar, D>;

            #[inline]
            fn acceleration_at<const CHECKED: bool>(
                &self,
                at: &Self::Vector,
                softening: &Self::Softening,
            ) -> Self::Output {
                let d = self.position - *at;
                let norm = d.norm_squared();

                // Branch removed by the compiler when `CHECKED` is false.
                if CHECKED && norm == 0.0 {
                    d
                } else {
                    let norm_s = norm + (softening * softening);
                    d * (self.m / (norm_s * norm_s.sqrt()))
                }
            }
        }

        impl<const D: usize> $crate::gravity::newtonian::AccelerationPaired
            for $crate::gravity::GravitationalField<SVector<$scalar, D>, $scalar>
        {
            type Softening = $scalar;

            type Output = SVector<$scalar, D>;

            #[inline]
            fn acceleration_paired(
                &self,
                other: &Self,
                softening: &$scalar,
            ) -> (Self::Output, Self::Output) {
                // The caller is responsible for ensuring that the two gravitational fields are
                // different, so we can skip checking it here.
                let dir = $crate::gravity::newtonian::AccelerationAt::acceleration_at::<false>(
                    // a = G * m / r^2
                    // The returned value is: fd1 = 1.0 / r^2 = -fd2
                    &$crate::gravity::GravitationalField::new(other.position, 1.0),
                    &self.position,
                    softening,
                );

                // Fg = fd * G * m1 * m2,
                // So considering that a = F / m
                // We have: a1 = fd1 * G * m1 * m2 / m1 = fd1 * G * m2
                // Thus: a1 = fd1 * mu2, and similarly: a2 = fd2 * mu1 = -fd1 * mu1
                (dir * other.m, -dir * self.m)
            }
        }
    };
}

impl_acceleration_cpu_scalar!(f32);
impl_acceleration_cpu_scalar!(f64);

macro_rules! impl_acceleration_cpu_simd {
    ($scalar: ident, $rsqrt_fn: expr) => {
        impl<const D: usize> crate::gravity::newtonian::AccelerationAt
            for crate::gravity::GravitationalField<SVector<$scalar, D>, $scalar>
        {
            type Vector = SVector<$scalar, D>;

            type Softening = $scalar;

            type Output = SVector<$scalar, D>;

            #[inline]
            fn acceleration_at<const CHECKED: bool>(
                &self,
                at: &Self::Vector,
                softening: &Self::Softening,
            ) -> Self::Output {
                let d = self.position - *at;
                let norm = d.norm_squared();
                let norm_s = norm + (*softening * *softening);
                let mag = self.m * $scalar($rsqrt_fn(norm_s.0 * norm_s.0 * norm_s.0));

                // Branch removed by the compiler when `CHECKED` is false.
                if CHECKED {
                    d * (mag.select(norm.simd_ne(<$scalar>::ZERO), <$scalar>::ZERO))
                } else {
                    d * mag
                }
            }
        }
    };
}

impl_acceleration_cpu_simd!(WideF32x4, wide::f32x4::recip_sqrt);
impl_acceleration_cpu_simd!(WideF32x8, wide::f32x8::recip_sqrt);
impl_acceleration_cpu_simd!(WideF64x4, |f: wide::f64x4| 1.0 / f.sqrt());

#[cfg(feature = "gpu")]
crate::impl_acceleration_gpu_2d!(SVector<f32, 2>, f32);
#[cfg(feature = "gpu")]
crate::impl_acceleration_gpu_3d!(SVector<f32, 3>, f32);

#[cfg(test)]
mod tests {
    // use super::*;

    // #[inline]
    // fn div<const D: usize, S>(v1: SVector<S, D>, v2: SVector<S, D>) -> SVector<S, D>
    // where
    //     S: nalgebra::ComplexField,
    // {
    //     v1.component_div(&v2)
    // }

    // crate::tests_algorithms!(SVector::<f32, 2>, f32, repeat, div, [4, 8], vec2 on cpu and gpu);
    // crate::tests_algorithms!(SVector::<f32, 3>, f32, repeat, div, [4, 8], vec3 on cpu and gpu);
    // crate::tests_algorithms!(SVector::<f32, 4>, f32, repeat, div, [4, 8],vec4 on cpu);
    // crate::tests_algorithms!(SVector::<f32, 5>, f32, repeat, div, [4, 8], vec5 on cpu);
    // crate::tests_algorithms!(SVector::<f32, 6>, f32, repeat, div, [4, 8], vec6 on cpu);
    // crate::tests_algorithms!(SVector::<f64, 2>, f64, repeat, div, [4], dvec2 on cpu);
    // crate::tests_algorithms!(SVector::<f64, 3>, f64, repeat,div, [4], dvec3 on cpu);
    // crate::tests_algorithms!(SVector::<f64, 4>,f64, repeat, div, [4], dvec4 on cpu);
    // crate::tests_algorithms!(SVector::<f64, 5>, f64, repeat, div, [4], dvec5 on cpu);
    // crate::tests_algorithms!(SVector::<f64, 6>, f64, repeat, div, [4], dvec6 on cpu);
}
