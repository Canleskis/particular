# RUSTFLAGS='-C target-cpu=native' can result in significantly better performance for both scalar and simd compute methods.
baseline=$1
query=$2

cargo bench --all-features --bench benchmark -- --save-baseline $baseline $query
cargo bench --all-features --bench benchmark -- --export=$baseline > $baseline.json