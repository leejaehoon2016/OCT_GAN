
#!/bin/sh

# 2020.10.05 modified
# shell parameters: g_lr

for seed in 777
do 
    for lr in 2e-4 2e-5
    do
        for batch in 500 1000
        do
            for num_split in 3 5
            do
                CUDA_VISIBLE_DEVICES=0,1 python real.py --seed $seed --D_lr $lr --G_lr $lr --batch $batch --num_split $num_split --data $1 > real_$1/${seed}_${lr}_${lr}_${batch}_${num_split}.log
            done
        done
    done
done