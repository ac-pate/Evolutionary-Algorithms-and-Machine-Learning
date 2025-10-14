# RNA Folding Evolutionary Algorithm - README

## Quick Start Guide

### Prerequisites
- Ubuntu/Debian Linux system
- Git installed
- Internet connection for Docker setup

### Setup Instructions

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd Evolutionary-Algorithms-and-Machine-Learning/Assignment_1
```

2. **Run the setup script:**
```bash
./setup_machine.sh
```
This will install Docker, set up IPknot, and prepare the environment.

3. **Run a single experiment:**
```bash
python3 src/rna_folding_ea.py
```

4. **Run device-optimized experiments:**
```bash
python3 src/ea_runner.py config/device_experiments.yml --device odin
```

5. **Generate visualization plots:**
```bash
./generate_plots.sh
```

6. **Check results:**
```bash
ls plots/  # See all generated visualization plots
cat assignment1_results.txt  # See final sequences
```

### For Multiple Devices

## Hardware-Optimized Experiment Configuration

This project supports device-specific configurations optimized for different hardware setups:

#### Device Specifications & Optimization Strategy

**ODIN (AMD Ryzen 9 9950X - 32 threads)**
- **Strategy**: Maximum population sizes and computational intensity
- **Population**: 400-500 individuals 
- **Generations**: 150-200
- **Rationale**: With 32 threads, this powerhouse can handle massive parallel fitness evaluations. Large populations ensure maximum diversity exploration while high generation counts allow thorough convergence.

**NYQUIST (AMD Ryzen 5 3600 - 12 threads)**
- **Strategy**: Balanced performance with good throughput
- **Population**: 200-250 individuals
- **Generations**: 120-150  
- **Rationale**: 12 threads provide solid parallel processing. Medium-high populations balance diversity with computational efficiency, while moderate generation counts ensure reasonable runtime.

**LAPTOP (HP Spectre Intel i7-1260P - 12 threads)**
- **Strategy**: Power-efficient with thermal management
- **Population**: 120-150 individuals
- **Generations**: 100-130
- **Rationale**: Laptop CPUs need thermal-aware configurations. Smaller populations reduce sustained CPU load while optimized generation counts prevent overheating during long runs.

**MINI-PC (Dell OptiPlex 7050 - ~8 threads estimated)**
- **Strategy**: Conservative but thorough exploration
- **Population**: 80-100 individuals  
- **Generations**: 120-160
- **Rationale**: Limited cores require smaller populations, but longer generation runs ensure thorough search space exploration despite computational constraints.

1. **Setup on each device:**
```bash
git clone <your-repo-url>
cd Evolutionary-Algorithms-and-Machine-Learning/Assignment_1
./setup_machine.sh
```

2. **Run device-optimized experiments:**
```bash
# On ODIN (powerhouse configuration)
python3 src/ea_runner.py config/device_experiments.yml --device odin

# On NYQUIST (balanced configuration)  
python3 src/ea_runner.py config/device_experiments.yml --device nyquist

# On LAPTOP (efficient configuration)
python3 src/ea_runner.py config/device_experiments.yml --device laptop

# On MINI-PC (conservative configuration)
python3 src/ea_runner.py config/device_experiments.yml --device minipc
```

3. **Collect results:**
All devices will generate `results_*.txt` and `stats_*.json` files. Combine the best results from all devices for comprehensive analysis.

### Files Structure
```
Assignment_1/
├── src/
│   ├── rna_folding_ea.py          # Main EA implementation
│   ├── ea_runner.py               # Multi-device experiment launcher
│   ├── ea_visualizer.py           # Post-processing visualization
│   └── progress_monitor.py        # Lightweight progress tracking
├── config/
│   ├── device_experiments.yml     # Device-optimized configurations
│   ├── experiments_simple.yml     # Simple test configurations
│   └── experiments_config.yml     # Full experiment configurations
├── setup_machine.sh              # Device setup script
├── convert_config.py             # JSON to YAML converter
├── test_setup.py                 # Setup validation script
├── README.md                     # This file
└── assignment1_results.txt       # Final output file
```

### Dependencies
- Python 3.x
- Docker
- numpy, matplotlib, seaborn, pandas (for visualization)

All dependencies are installed automatically by the setup script.

### Output
- `assignment1_results.txt` - Final RNA sequences for submission
- `results_*.txt` - Individual experiment results
- `stats_*.json` - Detailed statistics and fitness history
- `plots/` - Visualization plots for analysis and reports

### Troubleshooting

**Docker permission errors:**
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

**IPknot container not starting:**
```bash
docker rm -f ipknot_runner
docker run -dit --name ipknot_runner -v ${PWD}:/work -w /work --entrypoint bash ipknot -c "sleep infinity"
```

**Python import errors:**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```