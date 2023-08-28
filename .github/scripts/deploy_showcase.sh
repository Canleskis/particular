for subfolder in examples/*; do
    if [ -d $subfolder ]; then
        ./.github/scripts/deploy_example.sh $subfolder website/content/demos
    fi
done