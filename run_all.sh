#!/bin/bash
# Reproduces every simulation result in the paper (5 test seeds).
cd "$(dirname "$0")"
SEEDS="11 23 37 51 73"
( for sc in combined covariate gradual concept; do python3 -W ignore exp_sim.py $sc topology $SEEDS; done ) > ../results/runA.log 2>&1 &
( for sc in none individual; do python3 -W ignore exp_sim.py $sc topology $SEEDS; done
  python3 -W ignore exp_sim.py combined similarity $SEEDS ) > ../results/runB.log 2>&1 &
wait
