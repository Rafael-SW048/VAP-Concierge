#!/bin/bash
fair_res=/home/cc/ce/figure/dat/mix12/fair/fair-360-mix2.dat
grad_res=/home/cc/ce/figure/dat/mix12/conserv/grad-a0.8-t1000-360-mix2.dat
paste -d ' ' $fair_res $grad_res \
	| awk '{if (NR == 0) {prev = 0;} ewma = prev * 0.99 + ($6 - $3) * 0.01; print $1, $2, $3, $5, $6, $6-$3, ewma; prev = ewma}'\
	> dat/mix12/aggregate-conserv-mix2.dat
gnuplot plot/agg-conserv-mix2.gpi

fair_res=/home/cc/ce/figure/dat/mix12/fair/fair-360-mix1.dat
grad_res=/home/cc/ce/figure/dat/mix12/conserv/grad-a0.8-t1000-360-mix1.dat
paste -d ' ' $fair_res $grad_res \
	| awk '{if (NR == 0) {prev = 0;} ewma = prev * 0.99 + ($6 - $3) * 0.01; print $1, $2, $3, $5, $6, $6-$3, ewma; prev = ewma}'\
	> dat/mix12/aggregate-conserv-mix1.dat
gnuplot plot/agg-conserv-mix1.gpi
