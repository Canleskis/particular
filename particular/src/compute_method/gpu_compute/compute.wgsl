@group(0) @binding(0) var<storage, read> particles: array<PointMass>;
@group(0) @binding(1) var<storage, read> massive_particles: array<PointMass>;
@group(0) @binding(2) var<storage, read_write> accelerations: array<Vector>;

var<push_constant> softening_squared: f32;

@compute @workgroup_size(#WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>, @builtin(local_invocation_id) local_invocation_id: vec3<u32>) {
    let massive_len = arrayLength(&massive_particles);
    let global_id = global_invocation_id.x;

    let p1 = particles[global_id];
    var acceleration = Vector(0.0);

    for (var j = 0u; j < massive_len; j++) {
        let p2 = massive_particles[j];

        particle_acceleration(p1, p2, softening_squared, &acceleration);
    }

    accelerations[global_id] = acceleration;
}
