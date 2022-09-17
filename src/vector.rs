use std::ops::{Add, Div, Mul, Sub};

/// Representation of a vector as a type that is thread-safe, copyable, capable of standard arithmetic operations and with a "zero" value.
/// 
/// ---
/// **Favor using the [`vector_normed`](crate::vector_normed) macro instead of a manual implementation.
/// You shouldn't need to manually implement [`Vector`] but if you do, ensure [`zero_value`](Vector::zero_value) is inlined.**
pub trait Vector
where
    Self: Send
        + Sync
        + Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<f32, Output = Self>
        + Div<f32, Output = Self>,
{
    fn zero_value() -> Self;
}

/// Trait to describe a vector's norm/length.
///
/// This allows [`Particle`](crate::Particle) to be generic over the type describing its position.
/// 
/// ---
/// **Favor using the [`vector_normed`](crate::vector_normed) macro instead of a manual implementation.**
/// **You shouldn't need to manually implement [`NormedVector`] but if you do, ensure [`norm_sq`](NormedVector::norm_sq) is inlined.**
pub trait NormedVector: Vector {
    /// Squared length of the vector. Used by [`ParticleSet`](crate::ParticleSet) to compute the acceleration.
    fn norm_sq(self) -> f32;
}

/// Convenience macro to implement the [`NormedVector`] trait.
#[macro_export]
macro_rules! vector_normed {
    ($zero: ident, $norm: ident, $($t: ty),*) => { $(
        impl $crate::Vector for $t {
            #[inline]
            fn zero_value() -> Self {
                Self::$zero()
            }
        }

        impl $crate::NormedVector for $t {
            #[inline]
            fn norm_sq(self) -> f32 {
                self.$norm()
            }
        }
    )* };

    ($norm:ident, $($t:ty),*) => { $(
        $crate::vector_normed!(default, $norm, $t);
    )* };
}

#[cfg(feature = "glam")]
mod glam {
    crate::vector_normed!(
        length_squared,
        glam::Vec2,
        glam::Vec3,
        glam::Vec3A,
        glam::Vec4
    );
}

#[cfg(feature = "nalgebra")]
mod nalgebra {
    crate::vector_normed!(
        norm_squared,
        nalgebra::Vector1<f32>,
        nalgebra::Vector2<f32>,
        nalgebra::Vector3<f32>,
        nalgebra::Vector4<f32>,
        nalgebra::Vector5<f32>,
        nalgebra::Vector6<f32>
    );
}

#[cfg(feature = "cgmath")]
mod cgmath {
    use cgmath::{InnerSpace, Zero};
    crate::vector_normed!(
        zero,
        magnitude2,
        cgmath::Vector1<f32>,
        cgmath::Vector2<f32>,
        cgmath::Vector3<f32>,
        cgmath::Vector4<f32>
    );
}
