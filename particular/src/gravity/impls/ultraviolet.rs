use ultraviolet::{
    f32x4, f32x8, f64x2, f64x4, DVec2, DVec2x2, DVec2x4, DVec3, DVec3x2, DVec3x4, DVec4, DVec4x2,
    DVec4x4, Vec2, Vec2x4, Vec2x8, Vec3, Vec3x4, Vec3x8, Vec4, Vec4x4, Vec4x8,
};

crate::impl_into_array!([f32; 2], Vec2);
crate::impl_into_array!([f32; 3], Vec3);
crate::impl_into_array!([f32; 4], Vec4);
crate::impl_into_array!([f64; 2], DVec2);
crate::impl_into_array!([f64; 3], DVec3);
crate::impl_into_array!([f64; 4], DVec4);
crate::impl_into_array!([f32x4; 2], Vec2x4);
crate::impl_into_array!([f32x8; 2], Vec2x8);
crate::impl_into_array!([f32x4; 3], Vec3x4);
crate::impl_into_array!([f32x8; 3], Vec3x8);
crate::impl_into_array!([f32x4; 4], Vec4x4);
crate::impl_into_array!([f32x8; 4], Vec4x8);
crate::impl_into_array!([f64x2; 2], DVec2x2);
crate::impl_into_array!([f64x4; 2], DVec2x4);
crate::impl_into_array!([f64x2; 3], DVec3x2);
crate::impl_into_array!([f64x4; 3], DVec3x4);
crate::impl_into_array!([f64x2; 4], DVec4x2);
crate::impl_into_array!([f64x4; 4], DVec4x4);

crate::impl_norm!(Vec2, f32, |v: Vec2| v.mag_sq());
crate::impl_norm!(Vec3, f32, |v: Vec3| v.mag_sq());
crate::impl_norm!(Vec4, f32, |v: Vec4| v.mag_sq());
crate::impl_norm!(DVec2, f64, |v: DVec2| v.mag_sq());
crate::impl_norm!(DVec3, f64, |v: DVec3| v.mag_sq());
crate::impl_norm!(DVec4, f64, |v: DVec4| v.mag_sq());
crate::impl_norm!(Vec2x4, f32x4, |v: Vec2x4| v.mag_sq());
crate::impl_norm!(Vec2x8, f32x8, |v: Vec2x8| v.mag_sq());
crate::impl_norm!(Vec3x4, f32x4, |v: Vec3x4| v.mag_sq());
crate::impl_norm!(Vec3x8, f32x8, |v: Vec3x8| v.mag_sq());
crate::impl_norm!(Vec4x4, f32x4, |v: Vec4x4| v.mag_sq());
crate::impl_norm!(Vec4x8, f32x8, |v: Vec4x8| v.mag_sq());
crate::impl_norm!(DVec2x2, f64x2, |v: DVec2x2| v.mag_sq());
crate::impl_norm!(DVec2x4, f64x4, |v: DVec2x4| v.mag_sq());
crate::impl_norm!(DVec3x2, f64x2, |v: DVec3x2| v.mag_sq());
crate::impl_norm!(DVec3x4, f64x4, |v: DVec3x4| v.mag_sq());
crate::impl_norm!(DVec4x2, f64x2, |v: DVec4x2| v.mag_sq());
crate::impl_norm!(DVec4x4, f64x4, |v: DVec4x4| v.mag_sq());

crate::impl_reduce!([<Vec2, f32>; 4], Vec2x4 => reduce_add(x, y));
crate::impl_reduce!([<Vec2, f32>; 8], Vec2x8 => reduce_add(x, y));
crate::impl_reduce!([<Vec3, f32>; 4], Vec3x4 => reduce_add(x, y, z));
crate::impl_reduce!([<Vec3, f32>; 8], Vec3x8 => reduce_add(x, y, z));
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
crate::impl_acceleration_cpu_scalar!(Vec4, f32);
crate::impl_acceleration_cpu_scalar!(DVec2, f64);
crate::impl_acceleration_cpu_scalar!(DVec3, f64);
crate::impl_acceleration_cpu_scalar!(DVec4, f64);

crate::impl_acceleration_cpu_simd!(Vec2x4, f32x4, f32x4::recip_sqrt);
crate::impl_acceleration_cpu_simd!(Vec2x8, f32x8, f32x8::recip_sqrt);
crate::impl_acceleration_cpu_simd!(Vec3x4, f32x4, f32x4::recip_sqrt);
crate::impl_acceleration_cpu_simd!(Vec3x8, f32x8, f32x8::recip_sqrt);
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

#[cfg(test)]
mod tests {
    // use super::*;
    // use std::ops::Div;

    // crate::tests_algorithms!(Vec2, f32, broadcast, Div::div, [4, 8], vec2 on cpu and gpu);
    // crate::tests_algorithms!(Vec3, f32, broadcast, Div::div, [4, 8], vec3 on cpu and gpu);
    // crate::tests_algorithms!(Vec4, f32, broadcast, Div::div, [4, 8], vec4 on cpu);
    // crate::tests_algorithms!(DVec2, f64, broadcast, Div::div, [2, 4], dvec2 on cpu);
    // crate::tests_algorithms!(DVec3, f64, broadcast, Div::div, [2, 4], dvec3 on cpu);
    // crate::tests_algorithms!(DVec4, f64, broadcast, Div::div, [2, 4], dvec4 on cpu);
}
