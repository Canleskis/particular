struct Particle {
  pos : vec3<f32>,
  mu: f32,
};

@group(0) @binding(1) var<storage, read_write> particles : array<Particle>;

@compute
@workgroup_size(128)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let index = global_invocation_id.x;

    let total = arrayLength(&particles);

    var vPos = particles[index].pos;
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    
    for (var i = 0u; i < total; i++) {
        if (i == index) { continue; }

        let p1 = particles[index];
        let p2 = particles[i];

        let dir = p2.pos - p1.pos;

        let norm = (dir.x * dir.x) + (dir.y * dir.y) + (dir.z * dir.z);

        acc += dir * particles[i].mu / (norm * sqrt(norm));
    }

  particles[index] = Particle(acc, particles[index].mu);
}
