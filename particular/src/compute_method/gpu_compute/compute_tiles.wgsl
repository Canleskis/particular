@group(0) @binding(0) var<storage, read> particles: array<PointMass>;
@group(0) @binding(1) var<storage, read> massive_particles: array<PointMass>;
@group(0) @binding(2) var<storage, read_write> accelerations: array<Vector>;

var<push_constant> softening_squared: f32;
var<workgroup> shared_particles: array<PointMass, #WORKGROUP_SIZE>;

@compute @workgroup_size(#WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>, @builtin(local_invocation_id) local_invocation_id: vec3<u32>) {
    let massive_len = arrayLength(&massive_particles);
    let global_id = global_invocation_id.x;
    let local_id = local_invocation_id.x;

    let p1 = particles[global_id];
    var acceleration = Vector(0.0);

    for (var i = 0u; i < massive_len; i += #WORKGROUP_SIZE) {
        shared_particles[local_id] = massive_particles[i + local_id];

        workgroupBarrier();

        for (var j = 0u; j < #WORKGROUP_SIZE; j++) {
            let p2 = shared_particles[j];

            particle_acceleration(p1, p2, softening_squared, &acceleration);
        }
        
        workgroupBarrier();
    }

    accelerations[global_id] = acceleration;
}
