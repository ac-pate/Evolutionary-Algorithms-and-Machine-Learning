# RNA Folding EA Assignment
**Achal Patel - 40227663**

## Quick Start

### 1. Environment Setup (One Time)
```bash
# Run comprehensive setup
./scripts/setup_machine.sh

# Test everything works
python3 scripts/test_setup.py
```

**Important:** If you get Docker permission errors, restart your terminal session after setup.

### 2. Run Assignment Experiments

#### Basic Usage
```bash
# Simple run - auto-generates output in correct format
python3 src/ea_runner.py --experiment my_test

# With device optimization (if you have similar hardware)
python3 src/ea_runner.py --experiment my_test --device odin
```

#### Advanced Usage with Input Flag
```bash
# Custom input file
python3 src/ea_runner.py --experiment custom_test \
  --input path/to/your/constraints.csv

# Specific problems only
python3 src/ea_runner.py --experiment quick_test \
  --input src/EA_Assignment_1_constraints.csv \
  --problems "1.1,2.2,3.2"

# Device-optimized with custom input
python3 src/ea_runner.py --experiment production_run \
  --input src/EA_Assignment_1_constraints.csv \
  --device nyquist

# Enable experiment tracking with Weights & Biases
python3 src/ea_runner.py --experiment tracked_run \
  --wandb-enable

# Disable tracking (default behavior)
python3 src/ea_runner.py --experiment standard_run \
  --wandb-enable=false
```

### 3. Interactive Testing
```bash
# Menu-driven testing interface
python3 test_runner.py
```

## Experiment Tracking

### Weights & Biases Integration
- **Default**: Tracking disabled for fast assignment completion
- **Enable**: Use `--wandb-enable` flag for detailed experiment logging
- **Features**: Real-time fitness plots, parameter tracking, experiment comparison

```bash
# Run with experiment tracking
python3 src/ea_runner.py --experiment my_experiment --wandb-enable

# Standard run without tracking (default)
python3 src/ea_runner.py --experiment my_experiment
```

**Note**: Install wandb (`pip install wandb`) and configure (`wandb login`) for tracking features.

## Core Source Files

### `src/ea_runner.py` - Main Assignment Runner
- **Purpose**: Entry point for all experiments
- **Input handling**: If `--input` not provided, uses `src/EA_Assignment_1_constraints.csv`
- **Output naming**: Auto-generates `output/EA_Assignment_1_output-sheet1_{experiment}_{timestamp}.csv`
- **Device support**: Applies hardware-optimized configurations
- **EA integration**: Calls `rna_folding_ea.py` for each problem
- **Monitoring**: Uses `progress_monitor.py` for real-time updates

### `src/rna_folding_ea.py` - Core Evolutionary Algorithm
- **Features**: Diversity-aware termination, fitness caching, mutation strategies
- **Called by**: `ea_runner.py` for each RNA problem
- **Monitoring**: Integrates with `progress_monitor.py`
- **Output**: Returns top 5 diverse sequences per problem

### `src/csv_processor.py` - Assignment Format Handler
- **Input parsing**: Reads assignment CSV with bracket notation `[[` `]]`
- **Output formatting**: Generates correct `id,result_1,result_2,result_3,result_4,result_5` format
- **Problem support**: Handles all 6 assignment problems (1.1, 1.2, 2.1, 2.2, 3.1, 3.2)

### `src/progress_monitor.py` - Real-time Tracking
- **Features**: Generation progress, fitness tracking, ETA calculation
- **Safe calculations**: Robust time handling prevents display issues
- **Integration**: Called by EA during evolution

## Device Configurations

If you have similar hardware specs, use these optimized settings:

### ODIN Configuration (32-thread Powerhouse)
```bash
python3 src/ea_runner.py --experiment production --device odin
# Population: 500, Generations: 200, Workers: 28
```

### NYQUIST Configuration (12-thread Balanced)
```bash
python3 src/ea_runner.py --experiment balanced --device nyquist  
# Population: 250, Generations: 150, Workers: 10
```

### LAPTOP Configuration (12-thread Efficient)
```bash
python3 src/ea_runner.py --experiment efficient --device laptop
# Population: 150, Generations: 120, Workers: 8
```

### MINIPC Configuration (8-thread Conservative)
```bash
python3 src/ea_runner.py --experiment conservative --device minipc
# Population: 100, Generations: 130, Workers: 6
```

## Project Structure

```
Assignment_1/
├── README.md                          # This documentation
├── config.py                          # Central configuration (assignment format)
├── requirements.txt                   # Python dependencies for venv
├── test_runner.py                     # Interactive experiment testing
│
├── scripts/                           # Setup and utility scripts
│   ├── setup_machine.sh              # Complete environment setup
│   ├── test_setup.py                 # Validate setup after installation
│   ├── auto_experiments.sh           # Automated batch experiments
│   ├── ea_visualization_and_plots.py # Results analysis and plots
│   └── generate_output.py            # Output file generation
│
├── src/                              # Core source code
│   ├── EA_Assignment_1_constraints.csv # Assignment input (6 problems)
│   ├── ea_runner.py                  # Main assignment runner
│   ├── csv_processor.py              # CSV input/output handler
│   ├── rna_folding_ea.py            # Core evolutionary algorithm
│   └── progress_monitor.py          # Real-time progress tracking
│
├── output/                           # Auto-generated results
│   └── EA_Assignment_1_output-sheet1_*.csv
│
├── results/                          # Detailed experiment logs
├── data/                            # Master experiment data
├── ipknot/                          # IPknot RNA folding tool
└── venv/                            # Python virtual environment (optional)
```

## Scripts Folder

### `scripts/setup_machine.sh`
- **Purpose**: Complete environment setup
- **Features**: Docker installation, user group setup, IPknot container, Python dependencies
- **Docker fix**: Adds user to docker group for sudo-free access
- **Validation**: Tests all components after installation

### `scripts/test_setup.py`
- **Purpose**: Validate setup after installation
- **Checks**: Python dependencies, Docker access, IPknot container
- **Usage**: Run after `setup_machine.sh` to ensure everything works

### `scripts/auto_experiments.sh`
- **Purpose**: Automated batch experiments
- **Features**: Runs multiple configurations sequentially
- **Usage**: For comprehensive testing across different parameters

### `scripts/ea_visualization_and_plots.py`
- **Purpose**: Results analysis and visualization
- **Features**: Fitness plots, diversity charts, convergence analysis
- **Usage**: Generate publication-quality plots from results

### `scripts/generate_output.py`
- **Purpose**: Output file utilities
- **Features**: Format conversion, result aggregation
- **Usage**: Post-processing experiment results

## Virtual Environment

### Using Virtual Environment (Recommended)
```bash
# Virtual environment is created by setup_machine.sh
source venv/bin/activate
python3 src/ea_runner.py --experiment test
deactivate
```

### System Installation (Alternative)
```bash
# If you prefer system-wide installation
sudo apt install python3-numpy python3-matplotlib python3-pandas
python3 src/ea_runner.py --experiment test
```

**Requirements.txt** contains: `numpy`, `matplotlib`, `seaborn`, `pandas`, `pyyaml`, `wandb`

Virtual environment is optional but recommended to avoid system package conflicts.

## Output Files

### Auto-Generated Assignment Output
```
output/EA_Assignment_1_output-sheet1_my_test_20251014_143052.csv
```

**Format:**
```csv
id,result_1,result_2,result_3,result_4,result_5
1.1,CGAAUCGAUC...,UGCAAGCUA...,ACGUCGACU...,((...)),[[...]]
1.2,AUCGAUCGAU...,GCAAGCUAU...,CGUCGACUG...,((...)),((...))
...
```

### Results Folder Contents
- **Timestamped folders**: `2025-10-14_15-23-45_odin_experiment_name/`
- **Fitness logs**: `fitness_history.json`
- **Solution details**: `best_solutions.csv`
- **Experiment config**: `experiment_config.yaml`
- **Progress logs**: `evolution_log.txt`

### Data Folder
- **`master_output_fitness_data.json`**: Aggregated fitness data across all experiments
- **`master_output_solutions.csv`**: Best solutions from all runs
- **`all_experiments_master.csv`**: Experiment metadata and parameters

## Example Workflows

### Quick Assignment Run
```bash
./scripts/setup_machine.sh
python3 scripts/test_setup.py
python3 src/ea_runner.py --experiment assignment_submission
```

### Custom Problem Set
```bash
python3 src/ea_runner.py --experiment custom \
  --input my_problems.csv \
  --problems "1.1,3.2" \
  --device odin
```

### Batch Experiments
```bash
./scripts/auto_experiments.sh
python3 scripts/ea_visualization_and_plots.py
```

## Troubleshooting

### Docker Issues
```bash
# If Docker requires sudo after setup
newgrp docker
# Or restart terminal session
```

### IPknot Container Problems
```bash
# Restart container
docker rm -f ipknot_runner
./scripts/setup_machine.sh  # Will recreate container
```

### Python Import Errors
```bash
# Use virtual environment
source venv/bin/activate
# Or set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### Missing Output Files
- Check `output/` folder for auto-generated files
- Verify experiment completed without errors
- Use `python3 scripts/test_setup.py` to validate environment

## Assignment Compliance

- **Input**: `src/EA_Assignment_1_constraints.csv` (6 problems)
- **Output**: Auto-generated in `output/` with correct naming
- **Format**: Assignment-required CSV structure
- **Configuration**: Uses `config.py` as specified
- **Diversity**: Maintains 5 substantially different sequences per problem

