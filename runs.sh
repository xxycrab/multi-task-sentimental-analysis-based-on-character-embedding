#!/bin/sh

for l2 in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
	python train.py --l2_reg_lambda=$l2
done
