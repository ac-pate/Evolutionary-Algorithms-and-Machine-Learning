# Achal Patel - 40227663
# Multi-Device Experiment Launcher for RNA Folding EA
# Optimized configurations for different hardware setups

import json
import sys
import os
import argparse
from rna_folding_ea import RNAFoldingEA

try:
    import yaml
except ImportError:
    print("Error: PyYAML is not installed. Please install it with:")
    print("pip3 install pyyaml")
    print("or run: ./setup_machine.sh")
    sys.exit(1)

def get_device_workers(device):
    """
    Get optimal worker count for specific device
    
    Args:
        device (str): Device name (odin, nyquist, laptop, minipc)
    
    Returns:
        int: Optimal number of workers for the device
    """
    device_configs = {
        'odin': 28,    # AMD Ryzen 9950X (32 threads) - 87.5% utilization
        'nyquist': 8,  # AMD Ryzen 5 3600 (12 threads) - 66% utilization  
        'laptop': 8,   # Intel i7-1260P (12 threads) - 66% thermal-aware
        'minipc': 6    # Dell OptiPlex (~8 threads) - 75% utilization
    }
    
    return device_configs.get(device, 8)  # default to 8 if device unknown

def load_experiment_config(config_file):
    """load experiment configuration from yaml or json file"""
    with open(config_file, 'r') as f:
        if config_file.endswith('.yml') or config_file.endswith('.yaml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)

def run_experiment(config, device=None):
    """run a single experiment with given configuration"""
    print(f"Starting experiment: {config['NAME']}")
    print(f"Parameters: Pop={config['POPULATION_SIZE']}, Gen={config['GENERATIONS']}")
    
    # determine optimal worker count
    if device:
        max_workers = get_device_workers(device)
        print(f"Device: {device.upper()} | Workers: {max_workers}")
    else:
        max_workers = 8  # default
        print(f"Using default {max_workers} workers")
    
    # create ea instance
    ea = RNAFoldingEA(
        population_size=config['POPULATION_SIZE'],
        generations=config['GENERATIONS'],
        sequence_constraint=config['SEQUENCE_CONSTRAINT'],
        structure_constraint=config['STRUCTURE_CONSTRAINT'],
        max_workers=max_workers
    )
    
    # override default parameters if specified
    if 'CROSSOVER_RATE' in config:
        ea.crossover_rate = config['CROSSOVER_RATE']
    if 'MUTATION_RATE' in config:
        ea.mutation_rate = config['MUTATION_RATE']
    if 'TOURNAMENT_SIZE' in config:
        ea.tournament_size = config['TOURNAMENT_SIZE']
    
    # Add lightweight progress monitoring (no performance impact)
    try:
        from progress_monitor import add_progress_monitoring
        ea = add_progress_monitoring(ea)
        print("Enhanced progress monitoring enabled")
    except ImportError:
        print("Progress monitoring not available, running standard EA...")
    
    # run evolution
    ea.run_evolution()
    
    # save results with experiment name
    output_file = f"results_{config['NAME']}.txt"
    ea.save_results(output_file)
    
    # save detailed statistics
    stats_file = f"stats_{config['NAME']}.json"
    stats = {
        'name': config['NAME'],
        'parameters': config,
        'fitness_history': ea.fitness_history,
        'final_population_size': len(ea.population),
        'best_individuals_count': len(ea.best_individuals)
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Experiment {config['NAME']} completed!")
    print(f"Results: {output_file}, Stats: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='Launch RNA folding experiments optimized for specific devices')
    parser.add_argument('config', help='YAML or JSON configuration file')
    parser.add_argument('--device', choices=['odin', 'nyquist', 'laptop', 'minipc'], 
                       help='Run experiments optimized for specific device')
    parser.add_argument('--experiment', help='Specific experiment name to run')
    
    args = parser.parse_args()
    
    # load configuration
    config_data = load_experiment_config(args.config)
    
    if args.device:
        # run device-specific experiments
        device_experiments = [exp for exp in config_data['EXPERIMENTS'] 
                            if exp['NAME'].startswith(args.device)]
        if not device_experiments:
            print(f"No experiments found for device '{args.device}'!")
            print("Available devices: odin, nyquist, laptop, minipc")
            sys.exit(1)
        
        print(f"Running {len(device_experiments)} experiments for device: {args.device}")
        for experiment in device_experiments:
            run_experiment(experiment, device=args.device)
            
    elif args.experiment:
        # run specific experiment
        experiment = next((exp for exp in config_data['EXPERIMENTS'] if exp['NAME'] == args.experiment), None)
        if experiment:
            # try to detect device from experiment name
            device = None
            for dev in ['odin', 'nyquist', 'laptop', 'minipc']:
                if experiment['NAME'].startswith(dev):
                    device = dev
                    break
            run_experiment(experiment, device=device)
        else:
            print(f"Experiment '{args.experiment}' not found!")
            sys.exit(1)
    else:
        # run all experiments
        print("Running ALL experiments. This may take a very long time!")
        print("Consider using --device flag for device-specific experiments.")
        for experiment in config_data['EXPERIMENTS']:
            # try to detect device from experiment name
            device = None
            for dev in ['odin', 'nyquist', 'laptop', 'minipc']:
                if experiment['NAME'].startswith(dev):
                    device = dev
                    break
            run_experiment(experiment, device=device)

if __name__ == "__main__":
    main()