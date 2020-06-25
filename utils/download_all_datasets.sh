#!/bin/bash

declare -a datasets=(
    "ae_photos"
    "apple2orange"
    "summer2winter_yosemite"
    "horse2zebra"
    "monet2photo"
    "cezanne2photo"
    "ukiyoe2photo"
    "vangogh2photo"
    "maps"
    "cityscapes"
    "facades"
    "iphone2dslr_flower"
)

cd ../PA2_Skeleton
for dataset in "${datasets[@]}"
do
    bash datasets/download_cyclegan_dataset.sh ${dataset}
done
cd -