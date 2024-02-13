@group(0) @binding(0) var<storage, read> particles : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> massive_particles : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> accelerations : array<vec3<f32>>;

@compute
@workgroup_size(R_WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let i = global_invocation_id.x;

    let p1 = particles[i];
    var acceleration = vec3<f32>(0.0);

    for (var j = 0u; j < arrayLength(&massive_particles); j++) {
        let p2 = massive_particles[j];

        let dir = p2.xyz - p1.xyz;
        let norm = dot(dir, dir);
        let inv = p2.w * inverseSqrt(norm * norm * norm);
        let a = dir * inv;

        if norm != 0.0 {
            acceleration += a;
        }
    }

    accelerations[i] = acceleration;
}