#!/bin/bash

# 定义所有要测试的参数
SCHEDULER_TYPES=("round_robin" "lor" "lor3" "lor4" )
# 定义 A100 和 H100 的配对 (格式: "A100数量 H100数量")
GPU_PAIRS=(
    "15 3"
    "9 9"
    "12 6"
    "6 12"
    "18 2"
)
TENSOR_PARALLEL_SIZES=(2 4)
PIPELINE_SIZES=(2 4)
BATCH_SIZES=(64 128 256)  # 新增批处理大小参数


# 创建日志目录
LOG_DIR="experiment_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 记录实验配置
echo "Starting experiments at $(date)" | tee "$LOG_DIR/experiment_summary.log"
echo "Schedulers: ${SCHEDULER_TYPES[*]}" | tee -a "$LOG_DIR/experiment_summary.log"
echo "GPU pairs (A100 H100): ${GPU_PAIRS[*]}" | tee -a "$LOG_DIR/experiment_summary.log"
echo "Tensor parallel sizes: ${TENSOR_PARALLEL_SIZES[*]}" | tee -a "$LOG_DIR/experiment_summary.log"
echo "Pipeline sizes: ${PIPELINE_SIZES[*]}" | tee -a "$LOG_DIR/experiment_summary.log"
echo "Batch sizes: ${BATCH_SIZES[*]}" | tee -a "$LOG_DIR/experiment_summary.log"
echo "Schedulers: ${SCHEDULER_TYPES[*]}" | tee -a "$LOG_DIR/experiment_summary.log"

# 主实验循环

for gpu_pair in "${GPU_PAIRS[@]}"
do
        # 从配对中提取 A100 和 H100 的数量
    a100_count=$(echo $gpu_pair | cut -d' ' -f1)
    h100_count=$(echo $gpu_pair | cut -d' ' -f2)
        
    for tp_size in "${TENSOR_PARALLEL_SIZES[@]}"
    do
        for pipeline_size in "${PIPELINE_SIZES[@]}"
        do
            for batch_size in "${BATCH_SIZES[@]}"
            do
                for scheduler in "${SCHEDULER_TYPES[@]}"
                do
                    echo "Running experiment:"
                    echo "  A100 count: $a100_count"
                    echo "  H100 count: $h100_count"
                    echo "  Tensor parallel size: $tp_size"
                    echo "  Pipeline size: $pipeline_size"
                    echo "  Batch size: $batch_size"
                    echo "  Scheduler: $scheduler"

                    # 构建实验名称
                    exp_name="a${a100_count}_h${h100_count}_tp${tp_size}_pl${pipeline_size}_b${batch_size}_${scheduler}"
                    log_file="$LOG_DIR/${exp_name}.log"

                    # 运行实验
                    python -m vidur.main \
                        --cluster_config_a100_count="$a100_count" \
                        --cluster_config_h100_count="$h100_count" \
                        --cluster_config_tensor_parallel_size="$tp_size" \
                        --cluster_config_pipeline_size="$pipeline_size" \
                        --global_scheduler_config_type="$scheduler" \
                        --vllm_scheduler_config_batch_size_cap="$batch_size" \
                        --metrics_config_output_dir="$LOG_DIR/${exp_name}" \
                        2>&1 | tee "$log_file"

                    # 检查是否成功
                    if [ $? -eq 0 ]; then
                        echo "Experiment $exp_name completed successfully" | tee -a "$LOG_DIR/experiment_summary.log"
                    else
                        echo "Experiment $exp_name failed" | tee -a "$LOG_DIR/experiment_summary.log"
                    fi
                    echo "----------------------------------------"
                    # 可选：在实验之间添加短暂延迟
                    sleep 2
                done
            done
        done
    done
done


echo "All experiments completed at $(date)" | tee -a "$LOG_DIR/experiment_summary.log"
