#!/bin/bash

# Set the path to the embeddings root directory
EMBEDDINGS_ROOT="/Users/riccardoconci/Local_documents/!!GeomML_2025/GeomMLProj/Embeddings"

# Get the current timestamp for unique file names
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting linear probe training (hidden_dim = None)..."
python contrastive_experiments/run_all_probes.py \
    --embeddings_root $EMBEDDINGS_ROOT \
    --output_csv "probe_results_linear_${TIMESTAMP}.csv" \
    --phecode_size 1788

echo
echo "Starting hidden layer probe training (hidden_dim = 128)..."
python contrastive_experiments/run_all_probes.py \
    --embeddings_root $EMBEDDINGS_ROOT \
    --output_csv "probe_results_hidden128_${TIMESTAMP}.csv" \
    --phecode_size 1788 \
    --hidden_dim 128

echo
echo "All probe training complete!"
echo "Results saved to:"
echo "  probe_results_linear_${TIMESTAMP}.csv"
echo "  probe_results_hidden128_${TIMESTAMP}.csv" 