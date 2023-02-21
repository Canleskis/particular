struct Particle {
  pos: vec3<f32>,
  mu: f32,
};

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<storage, read> massive_particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write> accelerations : array<vec3<f32>>;

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let i = global_invocation_id.x;

    let p1 = particles[i];
    var acceleration = vec3<f32>(0.0);

    let total = arrayLength(&massive_particles);

    for (var j = 0u; j < total; j++) {
        let p2 = massive_particles[j];

        let dir = p2.pos - p1.pos;
        let norm = dot(dir, dir);
        let a = dir * p2.mu / (norm * sqrt(norm));

        if i != j {
            acceleration += a;
        }
    }

    accelerations[i] = acceleration;
}