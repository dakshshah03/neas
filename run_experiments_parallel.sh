#!/bin/bash

# Parse CLI args: -g/--gpus "0,1" (leave empty to auto-detect all GPUs)
GPU_LIST=""

usage() {
    echo "Usage: $0 [-g gpu_list]"
    echo ""
    echo "  -g, --gpus   Comma-separated list of GPU indices to use (e.g. \"0,1,3\")."
    echo "               Omit to auto-detect and use all available GPUs."
    echo "  -h, --help   Show this help message."
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        -g|--gpus)
            if [ -z "$2" ]; then
                echo "Error: Missing argument for $1"
                usage
            fi
            GPU_LIST="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# distributed across specified gpus
if [ -z "$GPU_LIST" ]; then
    # Use all available GPUs if GPU_LIST is empty
    NUM_PHYS_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if [ "$NUM_PHYS_GPUS" -eq 0 ]; then
        echo "No CUDA devices found!"
        exit 1
    fi
    AVAILABLE_GPUS=($(seq 0 $((NUM_PHYS_GPUS-1))))
else
    # Use specified GPUs (validate indices)
    IFS=',' read -ra AVAILABLE_GPUS <<< "$GPU_LIST"
    NUM_PHYS_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if [ "$NUM_PHYS_GPUS" -eq 0 ]; then
        echo "No CUDA devices found!"
        exit 1
    fi
    for i in "${!AVAILABLE_GPUS[@]}"; do
        # trim whitespace
        g="$(echo "${AVAILABLE_GPUS[$i]}" | xargs)"
        if ! [[ "$g" =~ ^[0-9]+$ ]]; then
            echo "ERROR: Invalid GPU index: '$g' (must be numeric)"
            exit 1
        fi
        if [ "$g" -lt 0 ] || [ "$g" -ge "$NUM_PHYS_GPUS" ]; then
            echo "ERROR: GPU index out of range: $g (found $NUM_PHYS_GPUS GPUs)"
            exit 1
        fi
        AVAILABLE_GPUS[$i]="$g"
    done
fi

NUM_GPUS=${#AVAILABLE_GPUS[@]}
echo "Using $NUM_GPUS CUDA devices: ${AVAILABLE_GPUS[*]}"

export TORCH_CUDA_ARCH_LIST="8.6"

configs=(
    "./config/foot_configs/foot_50_1m_freq.yaml"
    "./config/foot_configs/foot_50_2m_freq.yaml"
    "./config/foot_configs/foot_50_1m_hash.yaml"
    "./config/foot_configs/foot_50_2m_hash.yaml"
)

# Log file with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="experiments/${timestamp}"
mkdir -p "$EXP_DIR"
log_file="$EXP_DIR/experiment_batch_${timestamp}.log"

echo "Starting distributed experiments across $NUM_GPUS GPUs" | tee -a $log_file
echo "Logging to: $log_file" | tee -a $log_file
echo "Configs to run: ${#configs[@]}" | tee -a $log_file
echo "=================================" | tee -a $log_file

# Function to run a single config on a specific GPU
run_config() {
    local gpu_id=$1
    local config=$2
    local config_index=$3
    local total_configs=$4
    local gpu_log_file="$EXP_DIR/experiment_gpu${gpu_id}_${timestamp}.log"

    export CUDA_VISIBLE_DEVICES=$gpu_id

    echo "GPU $gpu_id: [$((config_index+1))/$total_configs] Starting: $config" | tee -a $log_file
    echo "GPU $gpu_id: Time: $(date)" | tee -a $log_file
    echo "GPU $gpu_id: Log: $gpu_log_file" | tee -a $log_file
    echo "---------------------------------" | tee -a $log_file

    if [ ! -f "$config" ]; then
        echo "GPU $gpu_id: ERROR: Config file not found: $config" | tee -a $log_file
        return 1
    fi

    start_time=$(date +%s)
    uv run python train.py --config "$config" > "$gpu_log_file" 2>&1
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        echo "GPU $gpu_id: ✓ COMPLETED: $config (${duration}s)" | tee -a $log_file
    else
        echo "GPU $gpu_id: ✗ FAILED: $config (exit code: $exit_code, ${duration}s)" | tee -a $log_file
        echo "GPU $gpu_id: Check $gpu_log_file for details" | tee -a $log_file
    fi

    return $exit_code
}

# Track running jobs and their PIDs
declare -A gpu_jobs
declare -A gpu_configs
declare -A gpu_indices

config_index=0
completed_count=0
failed_count=0

# Initial assignment: start one job per GPU
for gpu_id in "${AVAILABLE_GPUS[@]}"; do
    if [ $config_index -lt ${#configs[@]} ]; then
        config="${configs[$config_index]}"
        run_config $gpu_id "$config" $config_index ${#configs[@]} &
        gpu_jobs[$gpu_id]=$!
        gpu_configs[$gpu_id]="$config"
        gpu_indices[$gpu_id]=$config_index
        echo "Launched config $((config_index+1))/${#configs[@]} on GPU $gpu_id (PID: ${gpu_jobs[$gpu_id]})"
        ((config_index++))
    fi
done

# Wait for jobs to complete and assign new ones
while [ $completed_count -lt ${#configs[@]} ]; do
    for gpu_id in "${!gpu_jobs[@]}"; do
        pid=${gpu_jobs[$gpu_id]}

        # Check if job is still running
        if ! kill -0 $pid 2>/dev/null; then
            # Job finished, get exit code
            wait $pid
            exit_code=$?

            if [ $exit_code -eq 0 ]; then
                ((completed_count++))
            else
                ((failed_count++))
                ((completed_count++))
            fi

            # Remove from tracking
            unset gpu_jobs[$gpu_id]
            unset gpu_configs[$gpu_id]
            unset gpu_indices[$gpu_id]

            # Assign next config if available
            if [ $config_index -lt ${#configs[@]} ]; then
                config="${configs[$config_index]}"
                run_config $gpu_id "$config" $config_index ${#configs[@]} &
                gpu_jobs[$gpu_id]=$!
                gpu_configs[$gpu_id]="$config"
                gpu_indices[$gpu_id]=$config_index
                echo "Launched config $((config_index+1))/${#configs[@]} on GPU $gpu_id (PID: ${gpu_jobs[$gpu_id]})"
                ((config_index++))
            fi
        fi
    done

    # Brief sleep to avoid busy waiting
    sleep 10

    # Show progress
    if [ ${#gpu_jobs[@]} -gt 0 ]; then
        echo "Progress: $completed_count/${#configs[@]} completed, ${#gpu_jobs[@]} running..."
    fi
done

# Wait for any remaining jobs
for gpu_id in "${!gpu_jobs[@]}"; do
    wait ${gpu_jobs[$gpu_id]}
done

# Wrapping up
echo "" | tee -a $log_file
echo "=================================" | tee -a $log_file
echo "Distributed execution completed at $(date)" | tee -a $log_file
echo "Total configs: ${#configs[@]}" | tee -a $log_file
echo "Completed successfully: $((${#configs[@]} - failed_count))" | tee -a $log_file
echo "Failed: $failed_count" | tee -a $log_file
echo "Main log: $log_file" | tee -a $log_file
echo "Individual GPU logs: $EXP_DIR/experiment_gpu*_${timestamp}.log" | tee -a $log_file