path=$1
out_dir=$2
target_dir=${3:-"target"}

if [[ $path = "" ]]; then
    exit
fi

get_metadata() {
    awk -F '= ' '/'$1'/ { print }' $path/Cargo.toml
}

demo=$(get_metadata demo | awk '{ print $3 }')

if [[ $demo == true ]]; then
    title=$(get_metadata title)
    mobile=$(get_metadata mobile)
    name=$(get_metadata name | awk '{ gsub(/"/, ""); print $3 }')

    cargo build -p $name --release --target wasm32-unknown-unknown

    wasm-bindgen --no-typescript --out-name example --out-dir $out_dir/$name --target web $target_dir/wasm32-unknown-unknown/release/$name.wasm

    cp $path/preview.png $out_dir/$name/preview.png 2>/dev/null
    cp -r $path/assets $out_dir/$name/assets 2>/dev/null
    echo '+++
'$title'
'$mobile'
template = "demo.html"
+++' >$out_dir/$name/index.md
fi
