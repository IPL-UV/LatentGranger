#!/usr/bin/env bash


TMS=$1
PTH="viz/$TMS"

cd "$PTH"
mkdir tmp

convert -size 10x100  gradient:blue-red "gradient.png"

convert nig0/*.tiff -normalize tmp/nig0.png

for i in {0..229}; do
convert tmp/nig0-${i}.png gradient.png -clut -gravity South -pointsize 10 -annotate +0+0 $i tmp/col-$(printf "%03d\n" $i).png
done



convert tmp/col-*.png nig0.gif

convert -delay 20 -loop 0 nig0.gif -auto-level nig0.gif
