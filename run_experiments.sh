#!/bin/bash
# run_experiments.sh — All experiments for the AoD paper
#
# Produces results for:
#   - Table 1 / Figure 3: Main benchmark (5 models × 3 envs × 5 seeds)
#   - Figure 4: Lambda sweep (5 lambdas × 5 seeds)
#   - Figure 6: Corruption (2 models × 3 corruption levels × 5 seeds)
#
# Usage:
#./run_experiments.sh          # Run everything
#./run_experiments.sh --dry    # Print commands without running

set -e

DRY_RUN=false
if [[ "$1" == "--dry" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — printing commands only ==="
fi

COMMON="--total_steps 2000000 --n_envs 8 --n_steps 128 --n_epochs 4 \
        --batch_size 256 --lr 3e-4 --hidden_dim 128 --n_heads 4 \
        --seq_len 16 --log_interval 10 --save_interval 100"

run() {
    if $DRY_RUN; then
        echo "  python train.py $@"
    else
        echo ">>> python train.py $@"
        python train.py "$@"
    fi
}

# ─────────────────────────────────────────────
# 1. Main benchmark (Table 1, Figure 3)
# ─────────────────────────────────────────────
echo ""
echo "========== MAIN BENCHMARK =========="

MODELS="mlp gru transformer fixed_mixture aod"
ENVS="MemoryS7 DoorKey DynamicObstacles"
SEEDS="0 1 2 3 4"

for model in $MODELS; do
    for env in $ENVS; do
        for seed in $SEEDS; do
            run --model $model --env $env --seed $seed \
                --exp_name ${model}_${env}_seed${seed} \
                $COMMON
        done
    done
done

# ─────────────────────────────────────────────
# 2. Lambda sweep (Figure 4 — Pareto frontier)
#    Note: lambda=0.01 is the default, already run above
# ─────────────────────────────────────────────
echo ""
echo "========== LAMBDA SWEEP =========="

LAMBDAS="0.001 0.005 0.05 0.1"

for lam in $LAMBDAS; do
    for seed in $SEEDS; do
        # Format lambda for filename: 0.001 -> "0.001", 0.1 -> "0.1"
        lam_str=$(python -c "lam=$lam; print(f'{lam:g}')")
        run --model aod --env MemoryS7 --seed $seed \
            --lambda_compute $lam \
            --exp_name aod_lambda${lam_str}_MemoryS7_seed${seed} \
            $COMMON
    done
done

# ─────────────────────────────────────────────
# 3. Corruption experiments (Figure 6)
#    Note: p=0.0 is the default, already run above
# ─────────────────────────────────────────────
echo ""
echo "========== CORRUPTION EXPERIMENTS =========="

CORRUPTIONS="0.1 0.3 0.5"

for p in $CORRUPTIONS; do
    for seed in $SEEDS; do
        p_str=$(python -c "p=$p; print(f'{p:.1f}')")
        run --model aod --env MemoryS7 --seed $seed \
            --corruption_p $p \
            --exp_name aod_MemoryS7_corrupt${p_str}_seed${seed} \
            $COMMON

        run --model transformer --env MemoryS7 --seed $seed \
            --corruption_p $p \
            --exp_name transformer_MemoryS7_corrupt${p_str}_seed${seed} \
            $COMMON
    done
done

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
echo ""
echo "========== COMPLETE =========="
echo "Main benchmark:  75 runs (5 models × 3 envs × 5 seeds)"
echo "Lambda sweep:    20 runs (4 non-default lambdas × 5 seeds)"
echo "Corruption:      30 runs (2 models × 3 levels × 5 seeds)"
echo "Total:          125 runs"
echo ""
echo "Next steps:"
echo "  1. python save_trajectory.py --checkpoint results/aod_MemoryS7_seed0/model_final.pt"
echo "  2. python analyze.py"