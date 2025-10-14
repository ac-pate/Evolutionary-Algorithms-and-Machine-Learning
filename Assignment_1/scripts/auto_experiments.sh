#!/bin/bash

# Auto-Sequential Experiment Runner for RNA Folding EA
# Runs multiple experiments automatically one after another

echo "========================================="
echo "Auto-Sequential Experiment Runner Started"
echo "========================================="

# Define experiments array
experiments=(

    "odin_pop_1000_gen_200_mut_02"
    "odin_pop_600_gen_200_mut_012"
    "odin_pop_400_gen_200_mut_012"
)

# Base command template
BASE_CMD="python3 src/ea_runner.py config/device_experiments.yml --device odin --run 1"

# Track overall start time
OVERALL_START=$(date +%s)
echo "Overall start time: $(date)"
echo ""

# Run each experiment sequentially
for i in "${!experiments[@]}"; do
    exp_name="${experiments[$i]}"
    exp_num=$((i + 1))
    total_experiments=${#experiments[@]}
    
    echo "========================================="
    echo "Experiment $exp_num/$total_experiments: $exp_name"
    echo "========================================="
    
    # Record experiment start time
    EXP_START=$(date +%s)
    echo "Starting experiment: $exp_name at $(date)"
    
    # Construct and run command
    CMD="$BASE_CMD --experiment $exp_name"
    echo "Command: $CMD"
    echo ""
    
    # Execute the experiment
    if $CMD; then
        # Calculate experiment duration
        EXP_END=$(date +%s)
        EXP_DURATION=$((EXP_END - EXP_START))
        EXP_MINUTES=$((EXP_DURATION / 60))
        EXP_SECONDS=$((EXP_DURATION % 60))
        
        echo ""
        echo "✅ Experiment $exp_name completed successfully!"
        echo "   Duration: ${EXP_MINUTES}m ${EXP_SECONDS}s"
        echo "   Finished at: $(date)"
        
        # Calculate overall progress
        OVERALL_CURRENT=$(date +%s)
        OVERALL_ELAPSED=$((OVERALL_CURRENT - OVERALL_START))
        OVERALL_MINUTES=$((OVERALL_ELAPSED / 60))
        OVERALL_SECONDS=$((OVERALL_ELAPSED % 60))
        
        echo "   Overall progress: $exp_num/$total_experiments experiments completed"
        echo "   Total elapsed time: ${OVERALL_MINUTES}m ${OVERALL_SECONDS}s"
        
        # Add delay between experiments for system cooldown
        if [ $exp_num -lt $total_experiments ]; then
            echo ""
            echo "⏸️  Cooling down for 30 seconds before next experiment..."
            sleep 30
        fi
        
    else
        echo ""
        echo "❌ Experiment $exp_name FAILED!"
        echo "   Error occurred at: $(date)"
        
        # Ask user what to do on failure
        echo ""
        echo "Options:"
        echo "1) Continue with next experiment"
        echo "2) Retry this experiment"  
        echo "3) Abort all experiments"
        read -p "Choose option (1/2/3): " choice
        
        case $choice in
            1)
                echo "Continuing with next experiment..."
                ;;
            2)
                echo "Retrying experiment: $exp_name"
                ((i--))  # Decrement counter to retry
                ;;
            3)
                echo "Aborting all experiments."
                exit 1
                ;;
            *)
                echo "Invalid choice. Continuing with next experiment..."
                ;;
        esac
    fi
    
    echo ""
done

# Calculate and display final summary
OVERALL_END=$(date +%s)
OVERALL_TOTAL=$((OVERALL_END - OVERALL_START))
OVERALL_HOURS=$((OVERALL_TOTAL / 3600))
OVERALL_MINUTES=$(((OVERALL_TOTAL % 3600) / 60))
OVERALL_SECONDS=$((OVERALL_TOTAL % 60))

echo "========================================="
echo "🎉 All Experiments Completed Successfully!"
echo "========================================="
echo "Total experiments run: ${#experiments[@]}"
echo "Total runtime: ${OVERALL_HOURS}h ${OVERALL_MINUTES}m ${OVERALL_SECONDS}s"
echo "Finished at: $(date)"
echo ""
echo "Experiments completed:"
for exp in "${experiments[@]}"; do
    echo "  ✅ $exp"
done
echo ""
echo "Results available in:"
echo "  - data/all_experiments_master.csv"
echo "  - data/master_output_solutions.csv"
echo "  - results/ directory"
echo "========================================="