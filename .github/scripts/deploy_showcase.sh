for subfolder in examples/*; do
    if [ -d $subfolder ]; then
        ./scripts/deploy_example.sh $subfolder website/content/demos
    fi
done