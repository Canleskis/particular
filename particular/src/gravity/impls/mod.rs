#[cfg(feature = "glam")]
pub mod glam;
#[cfg(feature = "nalgebra")]
pub mod nalgebra;
#[cfg(feature = "ultraviolet")]
pub mod ultraviolet;

#[doc(hidden)]
#[macro_export]
macro_rules! impl_into_array {
    ([$scalar: ty; $dim: literal], $vector: ty) => {
        impl $crate::gravity::IntoArray for $vector {
            type Array = [$scalar; $dim];
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_norm {
    ($vector: ty, $scalar: ty, $norm_sq_fn: expr) => {
        impl $crate::gravity::Norm for $vector {
            type Output = $scalar;

            #[inline]
            fn norm_squared(self) -> $scalar {
                $norm_sq_fn(self)
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_reduce {
    ([<$vector: ty, $scalar:ty>; $lane: literal], $simd: ty => $reduce_fn: ident($($f: ident),+)) => {
        impl $crate::gravity::Reduce for $simd {
            type Output = $vector;

            #[inline]
            fn reduce_sum(self) -> Self::Output {
                Self::Output::new($(self.$f.$reduce_fn()),+)
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_to_simd {
    ([<$vector: ty, $scalar: ty>; $lane: literal] => <$simd_vector: ty, $simd_scalar: ty>) => {
        impl $crate::gravity::newtonian::ToSimd<$lane>
            for $crate::gravity::GravitationalField<$vector, $scalar>
        {
            type Simd = $crate::gravity::GravitationalField<$simd_vector, $simd_scalar>;

            #[inline]
            fn lanes_splat(&self) -> Self::Simd {
                $crate::gravity::GravitationalField::new(
                    <$simd_vector>::splat(self.position),
                    <$simd_scalar>::splat(self.m),
                )
            }

            #[inline]
            fn lanes_array(array: [Self; $lane]) -> Self::Simd {
                $crate::gravity::GravitationalField::new(
                    <$simd_vector>::from(array.map(|p| p.position)),
                    <$simd_scalar>::from(array.map(|p| p.m)),
                )
            }

            #[inline]
            fn lanes_slice(slice: &[Self]) -> Self::Simd {
                $crate::gravity::newtonian::ToSimd::<$lane>::lanes_array(std::array::from_fn(|i| {
                    slice.get(i).copied().unwrap_or_default()
                }))
            }
        }
    };
}

/// Trait for casting from one primitive to another.
pub trait FromPrimitive<U> {
    /// Casts to this primitive from the input primitive.
    fn from_primitive(p: U) -> Self;
}

macro_rules! impl_from_primitive {
    ($p1: ty => $p2: ty) => {
        impl FromPrimitive<$p1> for $p2 {
            #[inline]
            fn from_primitive(p: $p1) -> Self {
                p as Self
            }
        }
    };
}

impl_from_primitive!(usize => f32);
impl_from_primitive!(usize => f64);

impl<V, S> crate::gravity::newtonian::TreeData for crate::gravity::GravitationalField<V, S>
where
    V: std::ops::Add<V, Output = V>
        + std::ops::Mul<S, Output = V>
        + std::ops::Div<S, Output = V>
        + Default
        + Copy,
    S: FromPrimitive<usize>
        + std::ops::Div<Output = S>
        + PartialEq
        + Default
        + std::iter::Sum
        + Copy,
{
    type Data = Self;

    #[inline]
    fn centre_of_mass<I: ExactSizeIterator<Item = Self> + Clone>(particles: I) -> Self::Data {
        #[inline]
        fn sum<V: std::ops::Add<Output = V> + Default>(p: impl Iterator<Item = V>) -> V {
            p.fold(V::default(), V::add)
        }
        
        let tot = particles.clone().map(|p| p.m).sum();
        let com = if tot == S::default() {
            sum(particles.clone().map(|p| p.position)) / S::from_primitive(particles.len())
        } else {
            sum(particles.map(|p| p.position * (p.m / tot)))
        };

        Self::new(com, tot)
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_acceleration_cpu_scalar {
    ($vector: ty, $scalar: ty) => {
        impl $crate::gravity::newtonian::AccelerationAt
            for $crate::gravity::GravitationalField<$vector, $scalar>
        {
            type Vector = $vector;

            type Softening = $scalar;

            type Output = $vector;

            #[inline]
            fn acceleration_at<const CHECKED: bool>(
                &self,
                at: &$vector,
                softening: &$scalar,
            ) -> Self::Output {
                let d = self.position - *at;
                let norm = $crate::gravity::Norm::norm_squared(d);

                // Branch removed by the compiler when `CHECKED` is false.
                if CHECKED && norm == 0.0 {
                    d
                } else {
                    let norm_s = norm + (softening * softening);
                    d * (self.m / (norm_s * norm_s.sqrt()))
                }
            }
        }

        $crate::impl_acceleration_paired_cpu_scalar!($vector, $scalar);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_acceleration_paired_cpu_scalar {
    ($vector: ty, $scalar: ty) => {
        impl $crate::gravity::newtonian::AccelerationPaired
            for $crate::gravity::GravitationalField<$vector, $scalar>
        {
            type Softening = $scalar;

            type Output = $vector;

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

                // Fg = fd * G * m1 * m2
                // So considering that a = F / m
                // We have: a1 = fd1 * G * m1 * m2 / m1 = fd1 * G * m2
                // Thus: a1 = fd1 * mu2, and similarly: a2 = fd2 * mu1 = -fd1 * mu1
                (dir * other.m, -dir * self.m)
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_acceleration_cpu_simd {
    ($vector: ty, $scalar: ty, $rsqrt_fn: expr) => {
        impl $crate::gravity::newtonian::AccelerationAt
            for $crate::gravity::GravitationalField<$vector, $scalar>
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
                let norm = $crate::gravity::Norm::norm_squared(d);
                let norm_s = norm + (*softening * *softening);
                let mag = self.m * $rsqrt_fn(norm_s * norm_s * norm_s);

                // Branch removed by the compiler when `CHECKED` is false.
                if CHECKED {
                    d * wide::CmpNe::cmp_ne(norm, <$scalar>::ZERO).blend(mag, <$scalar>::ZERO)
                } else {
                    d * mag
                }
            }
        }
    };
}

// These shaders are used to compose the actual acceleration shader.
#[cfg(feature = "gpu")]
#[macro_use]
mod shaders {
    pub mod two_dimensions {
        pub mod types {
            wgsl_inline::wgsl! {
                struct Affected { position: vec2f }
                // vec4f has the same alignment as vec3f and gives better performance.
                struct Affecting { gravitational_field: vec4f }
                struct Interaction { acceleration: vec2f }
                fn position_affected(p: Affected) -> vec2f { return p.position; }
                fn position_affecting(p: Affecting) -> vec2f { return p.gravitational_field.xy; }
                fn mu(p: Affecting) -> f32 { return p.gravitational_field.z; }
            }
        }
        pub mod norm {
            wgsl_inline::wgsl! {
                fn norm(v: vec2f) -> f32 { return fma(v.x, v.x, v.y * v.y); }
            }
        }
        pub mod norm_softened {
            wgsl_inline::wgsl! {
                var<push_constant> softening: f32;
                fn norm(v: vec2f) -> f32 { return fma(v.x, v.x, fma(v.y, v.y, softening * softening)); }
            }
        }
    }

    pub mod three_dimensions {
        pub mod types {
            wgsl_inline::wgsl! {
                struct Affected { position: vec3f }
                struct Affecting { gravitational_field: vec4f }
                struct Interaction { acceleration: vec3f }
                fn position_affected(p: Affected) -> vec3f { return p.position; }
                fn position_affecting(p: Affecting) -> vec3f { return p.gravitational_field.xyz; }
                fn mu(p: Affecting) -> f32 { return p.gravitational_field.w; }
            }
        }
        pub mod norm {
            wgsl_inline::wgsl! {
                fn norm(v: vec3f) -> f32 { return fma(v.x, v.x, fma(v.y, v.y, v.z * v.z)); }
            }
        }
        pub mod norm_softened {
            wgsl_inline::wgsl! {
                var<push_constant> softening: f32;
                fn norm(v: vec3f) -> f32 { return fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, softening * softening))); }
            }
        }
    }

    // A little hacky, but wgsl_inline can't be configured to still compile when the shader is not
    // valid (the types and functions are not defined).
    pub mod acceleration {
        pub mod checked {
            pub const SOURCE: &str = "
fn compute(p1: Affected, p2: Affecting, out: ptr<function, Interaction>) {
    let dir = position_affecting(p2) - position_affected(p1);
    let norm = norm(dir);
    let acceleration = dir * (mu(p2) * inverseSqrt(norm * norm * norm));

    if norm != 0.0 {
        (*out).acceleration += acceleration;
    }
}";
        }
        pub mod unchecked {
            pub const SOURCE: &str = "
fn compute(p1: Particle, p2: Particle, out: ptr<function, Interaction>) {
    let dir = position_affecting(p2) - position_affected(p1);
    let norm = norm(dir);
    let acceleration = dir * (mu(p2) * inverseSqrt(norm * norm * norm));

    (*out).acceleration += acceleration;
}";
        }
    }
}

#[cfg(feature = "gpu")]
struct AccelerationShader<const D: usize, const CHECKED: bool>;

#[cfg(feature = "gpu")]
macro_rules! shader {
    ($dim: ident, $checked: ident) => {
        constcat::concat!(
            shaders::$dim::types::SOURCE,
            shaders::$dim::norm::SOURCE,
            shaders::acceleration::$checked::SOURCE,
        )
    };
}

#[cfg(feature = "gpu")]
macro_rules! shader_softened {
    ($dim: ident, $checked: ident) => {
        constcat::concat!(
            shaders::$dim::types::SOURCE,
            shaders::$dim::norm_softened::SOURCE,
            shaders::acceleration::$checked::SOURCE,
        )
    };
}

#[cfg(feature = "gpu")]
impl<const D: usize, const CHECKED: bool> AccelerationShader<D, CHECKED> {
    pub const SOURCE: &'static str = match D {
        2 => match CHECKED {
            true => shader!(two_dimensions, checked),
            false => shader!(two_dimensions, unchecked),
        },
        3 => match CHECKED {
            true => shader!(three_dimensions, checked),
            false => shader!(three_dimensions, unchecked),
        },
        _ => unimplemented!(),
    };

    pub const SOURCE_SOFTENED: &'static str = match D {
        2 => match CHECKED {
            true => shader_softened!(two_dimensions, checked),
            false => shader_softened!(two_dimensions, unchecked),
        },
        3 => match CHECKED {
            true => shader_softened!(three_dimensions, checked),
            false => shader_softened!(three_dimensions, unchecked),
        },
        _ => unimplemented!(),
    };
}

#[doc(hidden)]
#[macro_export]
#[cfg(feature = "gpu")]
macro_rules! impl_acceleration_gpu_2d {
    ($vector: ty, $scalar: ty) => {
        impl<const CHECKED: bool> $crate::gravity::newtonian::AccelerationGPU<CHECKED>
            for $crate::gravity::GravitationalField<$vector, $scalar>
        {
            type Vector = $vector;

            type GPUVector = $vector;

            type GPUAffecting = $crate::gravity::padded::GravitationalField<4, $vector, $scalar>;

            type GPUOutput = $vector;

            type Output = $vector;

            const SOURCE: &'static str =
                $crate::gravity::impls::AccelerationShader::<2, CHECKED>::SOURCE;

            const SOURCE_SOFTENED: &'static str =
                $crate::gravity::impls::AccelerationShader::<2, CHECKED>::SOURCE_SOFTENED;

            #[inline]
            fn to_gpu_position(position: &Self::Vector) -> Self::GPUVector {
                *position
            }

            #[inline]
            fn to_gpu_particle(&self) -> Self::GPUAffecting {
                From::from(*self)
            }

            #[inline]
            fn to_cpu_output(output: &Self::GPUOutput) -> Self::Output {
                *output
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
#[cfg(feature = "gpu")]
macro_rules! impl_acceleration_gpu_3d {
    ($vector: ty, $scalar: ty) => {
        impl<const CHECKED: bool> $crate::gravity::newtonian::AccelerationGPU<CHECKED>
            for $crate::gravity::GravitationalField<$vector, $scalar>
        {
            type Vector = $vector;

            type GPUVector = $crate::gravity::padded::Vector<4, $vector>;

            type GPUAffecting = $crate::gravity::padded::GravitationalField<0, $vector, $scalar>;

            type GPUOutput = $crate::gravity::padded::Vector<4, $vector>;

            type Output = $vector;

            const SOURCE: &'static str =
                $crate::gravity::impls::AccelerationShader::<3, CHECKED>::SOURCE;

            const SOURCE_SOFTENED: &'static str =
                $crate::gravity::impls::AccelerationShader::<3, CHECKED>::SOURCE_SOFTENED;

            #[inline]
            fn to_gpu_position(position: &Self::Vector) -> Self::GPUVector {
                From::from(*position)
            }

            #[inline]
            fn to_gpu_particle(&self) -> Self::GPUAffecting {
                From::from(*self)
            }

            #[inline]
            fn to_cpu_output(output: &Self::GPUOutput) -> Self::Output {
                output.vector
            }
        }
    };
}
