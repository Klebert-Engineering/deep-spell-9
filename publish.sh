#!/bin/bash

function zipdir {
    rm $1.zip
    zip -r $1.zip $1 -x "*.DS_Store"
}

exportdir="$(head -n1 README.md)-demo"
exportpath="../$exportdir"

rm -rf $exportpath
find . -name __pycache__ | xargs -n1 rm -rf

mkdir $exportpath
mkdir $exportpath/modules
mkdir $exportpath/models

cp -r modules/deepspell $exportpath/modules
cp -r modules/deepspell_service $exportpath/modules
cp models/deepsp_extra-v2_na_lr003_dec50_bat3192_128-128-128.* $exportpath/models
cp models/deepsp_discr-v3_na-lower_lr003_dec50_bat3072_fw128-128_bw128.* $exportpath/models
cp service.* $exportpath
cp README.md $exportpath

cd ..
zipdir $exportdir