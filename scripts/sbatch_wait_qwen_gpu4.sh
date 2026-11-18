#!/bin/bash
#SBATCH -p debug
#SBATCH -t 0-02:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=GPU1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --qos=normal
#SBATCH --job-name=wait-qwen-gpu4
#SBATCH --output=/data-nfs/gpu2/u18871384022/CoRProRAG/logs/wait_qwen_gpu4_%j.out
#SBATCH --error=/data-nfs/gpu2/u18871384022/CoRProRAG/logs/wait_qwen_gpu4_%j.err
set -euo pipefail

endpoint="http://GPU4:18000/v1/models"
for _ in $(seq 1 480); do
    if curl --noproxy '*' --silent --fail --connect-timeout 5 --max-time 10 "$endpoint" |
        grep -q 'qwen3-vl-32b-instruct'; then
        echo "Qwen3-VL service ready at $endpoint"
        exit 0
    fi
    sleep 15
done
echo "Qwen3-VL service did not become ready within two hours: $endpoint" >&2
exit 1
