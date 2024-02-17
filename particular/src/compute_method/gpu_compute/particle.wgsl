alias Vector = vec3f;
alias PointMass = vec4f;

fn position(point_mass: PointMass) -> Vector {
    return point_mass.xyz;
}

fn mass(point_mass: PointMass) -> f32 {
    return point_mass.w;
}

fn particle_acceleration(p1: PointMass, p2: PointMass, softening_squared: f32, acceleration: ptr<function, Vector>) {
    let dir = position(p2) - position(p1);
    let norm = fma(dir.x, dir.x, fma(dir.y, dir.y, fma(dir.z, dir.z, softening_squared)));
    let a = dir * (mass(p2) * inverseSqrt(norm * norm * norm));

    if norm != 0.0 {
        *acceleration += a;
    }
}
