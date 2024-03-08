# RUSTFLAGS='-C target-feature=+simd128' for supported browsers speeds up `BruteForceSIMD` significantly.
cargo build --bench=benchmark --release --target wasm32-wasi
cp `ls -t target/wasm32-wasi/release/deps/*.wasm | head -n 1` bench.wasm

# On https://webassembly.sh/, the following commands are then used to run and export the benchmark:
# bench --bench --save-baseline=wasm
# bench --bench --export=wasm | download