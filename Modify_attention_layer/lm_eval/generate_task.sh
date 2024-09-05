shots=5
# Prepare dataset
echo "prepare dataset"
for task in copa openbookqa winogrande piqa rte; do
  python -u generate_task_data.py \
  --output-file "results/${task}-${shots}.jsonl" \
  --task-name ${task} \
  --num-fewshot ${shots} 
done