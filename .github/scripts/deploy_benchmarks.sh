# Add all the benchmarks to a single json file in a list.

out_dir=$1
name=${2:-"benchmarks"}

buffer=""

for file in particular/benches/results/*-*-*.json; do
    buffer+=$(cat "$file")
    buffer+=","
done

echo "[${buffer%?}]" > $out_dir/$name.json
