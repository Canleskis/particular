path=$1
out_dir=$2
target_dir=${3:-"target"}

metadata=$(grep -A3 "\[package.metadata.particular.rs\]" $path/Cargo.toml | grep -vE "^(#|\[)")

if [ -n "$metadata" ]; then
    name=$(basename $path)

    cargo build -p $name --release --target wasm32-unknown-unknown
    wasm-bindgen --no-typescript --out-name example --out-dir $out_dir/$name --target web $target_dir/wasm32-unknown-unknown/release/$name.wasm
    cp $path/preview.png $out_dir/$name/preview.png 2>/dev/null
    cp -r $path/assets $out_dir/$name/assets 2>/dev/null
    echo '+++
'"$metadata"'
template = "demo.html"
+++' >$out_dir/$name/index.md
fi
