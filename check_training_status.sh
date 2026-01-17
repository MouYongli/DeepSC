#!/bin/bash
# Check DeepSC Perturbation Prediction Training Status

echo "============================================="
echo "DeepSC Perturbation Training Status"
echo "============================================="
echo ""

# Check if process is running
if ps aux | grep -v grep | grep "run_perturbation_simple.py" > /dev/null; then
    echo "✓ Training is RUNNING"
    echo ""

    # Get PID
    PID=$(ps aux | grep -v grep | grep "run_perturbation_simple.py" | awk 'NR==1{print $2}')
    echo "Process ID: $PID"

    # GPU usage
    echo ""
    echo "GPU Usage:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep $PID || echo "  (Not using GPU or data not available)"

    # Latest output directory
    echo ""
    echo "Latest Output Directory:"
    LATEST_DIR=$(ls -td /DATA2/DeepSC/results/perturbation_prediction/*/* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo "  $LATEST_DIR"
        echo ""
        echo "Directory contents:"
        ls -lh "$LATEST_DIR"

        # Check log file
        if [ -f "$LATEST_DIR/run_perturbation_simple_0.log" ]; then
            echo ""
            echo "Last 20 lines of log:"
            tail -20 "$LATEST_DIR/run_perturbation_simple_0.log"
        fi
    fi
else
    echo "✗ Training is NOT running"
    echo ""
    echo "Latest results:"
    ls -lhtr /DATA2/DeepSC/results/perturbation_prediction/*/* 2>/dev/null | tail -5
fi

echo ""
echo "============================================="
