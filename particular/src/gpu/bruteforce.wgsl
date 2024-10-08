@group(0) @binding(0) var<storage, read> affected: array<Affected>;
@group(0) @binding(1) var<storage, read> affecting: array<Affecting>;
@group(0) @binding(2) var<storage, read_write> interactions: array<Interaction>;

@compute @workgroup_size(#WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>, @builtin(local_invocation_id) local_invocation_id: vec3<u32>) {
    let affecting_len = arrayLength(&affecting);
    let global_id = global_invocation_id.x;

    let affected = affected[global_id];
    var out = Interaction();
    
    for (var j = 0u; j < affecting_len; j++) {
        let affecting = affecting[j];

        compute(affected, affecting, &out);
    }

    interactions[global_id] = out;
}
