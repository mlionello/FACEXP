#!/usr/bin/bash

for tr_ch in 0 1; do
for tr_rep in 0 1; do
for tst_rep in 0 1; do
for tst_intensity in 0 1; do
for tr_intensity in 0 1; do

outputid=trch_"$tr_ch"_trrep_"$tr_rep"_trintensity_"$tr_intensity"_tstrep_"$tst_rep"_tstintensity_"$tst_intensity"

python RAVNESS_analyser --input /home/matteo.lionello/RAVNESS/pdist_ownref/ \
        --output $outputid \
        --tr_ch $tr_ch \
        --tr_rep $tr_rep \
        --tst_rep $tst_rep \
        --tst_intensity $tst_intensity \
        --tr_intensity $tr_intensity

done
done
done
done
done