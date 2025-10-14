# Achal Patel - 40227663
# Multi-Device Multi-Problem Experiment Launcher for RNA Folding EA
# Optimized configurations for different hardware setups
# Supports all 6 constraint problems with wandb tracking and local backups

import json
import sys
import os
import argparse
import datetime
import csv
from pathlib import Path
from rna_folding_ea import RNAFoldingEA

try:
    import yaml
except ImportError:
    print("Error: PyYAML is not installed. Please install it with:")
    print("pip3 install pyyaml")
    print("or run: ./setup_machine.sh")
    sys.exit(1)

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed. Please install it with:")
    print("pip3 install wandb")
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
        'nyquist': 10,  # AMD Ryzen 5 3600 (12 threads) - 66% utilization
        'laptop': 8,    # Intel i7-1260P (12 threads) - 66% thermal-aware
        'minipc': 6     # Dell OptiPlex (~8 threads) - 75% utilization
    }
    
    return device_configs.get(device, 8)  # default to 8 if device unknown

def load_experiment_config(config_file):
    """load experiment configuration from yaml or json file"""
    with open(config_file, 'r') as f:
        if config_file.endswith('.yml') or config_file.endswith('.yaml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)

def create_experiment_folders(experiment_name, device, problems_to_run, run_number=1):
    """Create folder structure for experiment results and logging"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"{timestamp}_{device}_{experiment_name}_run{run_number:03d}"
    
    # Create main experiment folder
    results_dir = Path("results") / base_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create problem subfolders  
    problem_dirs = {}
    for problem_id in problems_to_run:
        problem_dir = results_dir / f"problem_{problem_id}"
        problem_dir.mkdir(exist_ok=True)
        problem_dirs[problem_id] = problem_dir
    
    # Create data tracking folder
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    return results_dir, problem_dirs, data_dir

def save_experiment_summary(results_dir, experiment_name, device, config, all_results):
    """Save experiment summary CSV and output CSV matching required format"""
    
    # Summary CSV with all experiment details
    summary_file = results_dir / "experiment_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'problem_id', 'best_fitness', 'generation_found', 'avg_fitness_final', 
            'avg_diversity_final', 'total_time_seconds', 'population_size', 
            'generations', 'device', 'experiment_name'
        ])
        
        for problem_id, result in all_results.items():
            writer.writerow([
                problem_id, result['best_fitness'], result['generation_found'],
                result['avg_fitness_final'], result.get('avg_diversity_final', 0.0), 
                result['total_time'], config['POPULATION_SIZE'], config['GENERATIONS'],
                device, experiment_name
            ])
    
    # Output CSV in assignment format (top 5 sequences per problem)
    output_file = results_dir / "output_results.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'result_1', 'result_2', 'result_3', 'result_4', 'result_5'])
        
        for problem_id in sorted(all_results.keys()):
            result = all_results[problem_id]
            top_sequences = result.get('top_sequences', [''] * 5)
            # Ensure we have exactly 5 results
            while len(top_sequences) < 5:
                top_sequences.append('')
            writer.writerow([problem_id] + top_sequences[:5])
    
    return summary_file, output_file

def update_master_data_file(data_dir, experiment_name, device, config, all_results, total_time, run_number=1):
    """Update master data file with high-level experiment information and upload to wandb"""
    master_file = data_dir / "all_experiments_master.csv"
    
    # Check if file exists and create header if not
    file_exists = master_file.exists()
    
    with open(master_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'timestamp', 'experiment_name', 'device', 'run_number', 'population_size', 'generations',
                'crossover_rate', 'mutation_rate', 'tournament_size', 'total_runtime_minutes',
                'problems_solved', 'avg_best_fitness', 'best_overall_fitness', 'total_valid_solutions',
                'avg_generation_found', 'efficiency_score', 'problem_1.1_fitness', 'problem_1.2_fitness',
                'problem_2.1_fitness', 'problem_2.2_fitness', 'problem_3.1_fitness', 'problem_3.2_fitness'
            ])
        
        # Calculate aggregate statistics
        problems_solved = len(all_results)
        avg_best_fitness = sum(r['best_fitness'] for r in all_results.values()) / problems_solved if problems_solved > 0 else 0
        best_overall_fitness = max(r['best_fitness'] for r in all_results.values()) if problems_solved > 0 else 0
        total_valid_solutions = sum(len(r.get('top_sequences', [])) for r in all_results.values())
        avg_generation_found = sum(r['generation_found'] for r in all_results.values()) / problems_solved if problems_solved > 0 else 0
        efficiency_score = (avg_best_fitness * problems_solved) / (total_time / 60) if total_time > 0 else 0
        
        # Individual problem fitnesses for comparison
        problem_fitnesses = {}
        for pid in ['1.1', '1.2', '2.1', '2.2', '3.1', '3.2']:
            problem_fitnesses[f'problem_{pid}_fitness'] = all_results.get(pid, {}).get('best_fitness', 0)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            timestamp, experiment_name, device, run_number, config['POPULATION_SIZE'], config['GENERATIONS'],
            config.get('CROSSOVER_RATE', 0.8), config.get('MUTATION_RATE', 0.01), 
            config.get('TOURNAMENT_SIZE', 3), round(total_time / 60, 2),
            problems_solved, round(avg_best_fitness, 4), round(best_overall_fitness, 4),
            total_valid_solutions, round(avg_generation_found, 2), round(efficiency_score, 4),
            problem_fitnesses['problem_1.1_fitness'], problem_fitnesses['problem_1.2_fitness'],
            problem_fitnesses['problem_2.1_fitness'], problem_fitnesses['problem_2.2_fitness'],
            problem_fitnesses['problem_3.1_fitness'], problem_fitnesses['problem_3.2_fitness']
        ])
    
    return master_file

def run_single_problem(problem_id, problem_config, experiment_config, device, max_workers, problem_dir, wandb_run, generation_offset):
    """Run EA for a single problem and return results"""
    print(f"\n{'='*60}")
    print(f"Running Problem {problem_id}")
    print(f"Structure: {problem_config['STRUCTURE_CONSTRAINT'][:50]}...")
    print(f"Sequence:  {problem_config['SEQUENCE_CONSTRAINT'][:50]}...")
    print(f"{'='*60}")
    
    # Create EA instance with 1% elitism
    ea = RNAFoldingEA(
        population_size=experiment_config['POPULATION_SIZE'],
        generations=experiment_config['GENERATIONS'],
        sequence_constraint=problem_config['SEQUENCE_CONSTRAINT'],
        structure_constraint=problem_config['STRUCTURE_CONSTRAINT'],
        max_workers=max_workers,
        elite_percentage=0.01  # Force 1% elitism
    )
    
    # Override default parameters if specified (except elite_percentage)
    if 'CROSSOVER_RATE' in experiment_config:
        ea.crossover_rate = experiment_config['CROSSOVER_RATE']
    if 'MUTATION_RATE' in experiment_config:
        ea.mutation_rate = experiment_config['MUTATION_RATE']
    if 'TOURNAMENT_SIZE' in experiment_config:
        ea.tournament_size = experiment_config['TOURNAMENT_SIZE']
    if 'EARLY_TERMINATION_FITNESS' in experiment_config:
        ea.early_termination_fitness = experiment_config['EARLY_TERMINATION_FITNESS']
    if 'HIGH_FITNESS_STREAK_THRESHOLD' in experiment_config:
        ea.high_fitness_streak_threshold = experiment_config['HIGH_FITNESS_STREAK_THRESHOLD']
    
    # Add progress monitoring
    try:
        from progress_monitor import add_progress_monitoring
        ea = add_progress_monitoring(ea)
        print("Enhanced progress monitoring enabled")
    except ImportError:
        print("Progress monitoring not available, running standard EA...")
    
    # Log experiment configuration
    if wandb_run:
        wandb_run.config.update({
            f"problem_{problem_id}": {
                "population_size": experiment_config['POPULATION_SIZE'],
                "generations": experiment_config['GENERATIONS'],
                "crossover_rate": experiment_config.get('CROSSOVER_RATE', 0.8),
                "mutation_rate": experiment_config.get('MUTATION_RATE', 0.01),
                "tournament_size": experiment_config.get('TOURNAMENT_SIZE', 3),
                "elite_percentage": 0.01,  # Always 1%
                "sequence_length": len(problem_config['SEQUENCE_CONSTRAINT']),
                "sequence_constraint": problem_config['SEQUENCE_CONSTRAINT'],
                "structure_constraint": problem_config['STRUCTURE_CONSTRAINT']
            }
        })
    
    import time
    start_time = time.time()
    
    history = None
    metric_prefix = None

    if wandb_run:
        metric_prefix = f"problem_{problem_id}"
        history = {
            "global_step": [],
            "problem_generation": [],
            "fitness_max": [],
            "fitness_avg": [],
            "diversity": []
        }

        def wandb_callback(generation, best_fitness, avg_fitness, diversity):
            global_step = generation_offset + generation
            history["global_step"].append(global_step)
            history["problem_generation"].append(generation)
            history["fitness_max"].append(best_fitness)
            history["fitness_avg"].append(avg_fitness)
            history["diversity"].append(diversity)

            wandb_run.log({
                "global_step": global_step,
                "generation": generation,
                "problem_id": problem_id,
                "fitness_max": best_fitness,
                "fitness_avg": avg_fitness,
                "diversity": diversity,
                f"{metric_prefix}_fitness_max": best_fitness,
                f"{metric_prefix}_fitness_avg": avg_fitness,
                f"{metric_prefix}_diversity": diversity
            })
        
        # Add callback to EA
        ea.add_callback(wandb_callback)
        print(f"Wandb real-time callback registered for problem {problem_id}")
    else:
        print(f"No wandb run for problem {problem_id}")
    
    ea.run_evolution()

    if wandb_run and history and history["global_step"]:
        # Create a table and combined chart for the charts dashboard
        combined_data = list(zip(
            history["global_step"],
            history["problem_generation"],
            history["fitness_max"],
            history["fitness_avg"],
            history["diversity"]
        ))

        combined_table = wandb.Table(
            columns=["global_step", "problem_generation", "fitness_max", "fitness_avg", "diversity"],
            data=combined_data
        )

        wandb_run.log({
            f"{metric_prefix}_fitness_diversity_table": combined_table
        })

        combined_chart = wandb.plot.line_series(
            xs=history["global_step"],
            ys=[history["fitness_max"], history["fitness_avg"], history["diversity"]],
            keys=["fitness_max", "fitness_avg", "diversity"],
            title=f"Problem {problem_id} Fitness & Diversity",
            xname="global_step"
        )

        wandb_run.log({
            f"{metric_prefix}_fitness_diversity_chart": combined_chart
        })
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Get results
    best_individuals = ea.get_best_individuals()
    fitness_history = ea.fitness_history
    
    # Extract fitness values from fitness_history tuples (generation, max_fitness, avg_fitness)
    if fitness_history and isinstance(fitness_history[0], tuple):
        fitness_values = [entry[1] for entry in fitness_history]  # Extract max_fitness from each tuple
    else:
        fitness_values = fitness_history  # Fallback if it's just numbers
    
    # Find generation where best fitness was first achieved
    best_fitness = max(fitness_values) if fitness_values else 0
    generation_found = next((entry[0] for entry in fitness_history if entry[1] == best_fitness), 0) if fitness_history else 0
    
    # Get top 5 valid sequences above fitness threshold
    valid_sequences = [ind for ind in best_individuals if ind['fitness'] > 0.5]
    valid_sequences.sort(key=lambda x: x['fitness'], reverse=True)
    top_sequences = [seq['sequence'] for seq in valid_sequences[:5]]
    
    # Save problem-specific results with improved naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    problem_results_file = problem_dir / f"problem_{problem_id}_{timestamp}_sequences.txt"
    
    # Custom save for individual problem
    with open(problem_results_file, 'w') as f:
        f.write(f"# Problem {problem_id} Results\n")
        f.write(f"# Best Fitness: {best_fitness:.4f}\n")
        f.write(f"# Generation Found: {generation_found}\n")
        f.write(f"# Runtime: {runtime:.2f}s\n")
        f.write(f"# Valid Sequences: {len(valid_sequences)}\n\n")
        for i, seq_data in enumerate(valid_sequences[:10]):  # Top 10
            f.write(f"# Rank {i+1}: Fitness {seq_data['fitness']:.4f}\n")
            f.write(f"{seq_data['sequence']}\n\n")
    
    print(f"Problem {problem_id} sequences saved to: {problem_results_file.name}")
    
    # Save problem-specific stats
    if fitness_history:
        # Calculate final average fitness from last 10 generations
        final_avg_fitness = sum(entry[2] for entry in fitness_history[-10:]) / min(10, len(fitness_history))
    else:
        final_avg_fitness = 0
    
    # Calculate final diversity if available
    try:
        final_diversity = ea.calculate_diversity(ea.population) if hasattr(ea, 'calculate_diversity') and ea.population else 0.0
    except:
        final_diversity = 0.0
        
    problem_stats = {
        'problem_id': problem_id,
        'best_fitness': best_fitness,
        'generation_found': generation_found,
        'avg_fitness_final': final_avg_fitness,
        'avg_diversity_final': final_diversity,
        'total_time': runtime,
        'top_sequences': top_sequences,
        'fitness_history': fitness_history
    }
    
    problem_stats_file = problem_dir / f"problem_{problem_id}_stats.json"
    with open(problem_stats_file, 'w') as f:
        json.dump(problem_stats, f, indent=2)
    
    # Log final results to wandb summary (not charts)
    if wandb_run:
        # Use wandb.summary for final metrics (tables, not time-series charts)
        wandb_run.summary.update({
            f"final_best_fitness": best_fitness,
            f"final_generation_found": generation_found,
            f"final_runtime_seconds": runtime,
            f"final_valid_sequences": len(valid_sequences),
            f"final_problem": problem_id
        })
    
    print(f"Problem {problem_id} completed!")
    print(f"Best fitness: {best_fitness:.4f} (found at generation {generation_found})")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Valid sequences found: {len(valid_sequences)}")
    
    return problem_stats
def run_experiment(experiment_config, device=None, problems_to_run=None, run_number=1, enable_wandb=True):
    """Run experiment for all problems or specific problems with wandb tracking"""
    print(f"\n{'='*80}")
    print(f"Starting experiment: {experiment_config['NAME']} (Run #{run_number})")
    print(f"Parameters: Pop={experiment_config['POPULATION_SIZE']}, Gen={experiment_config['GENERATIONS']}")
    
    # Determine optimal worker count
    if device:
        max_workers = get_device_workers(device)
        print(f"Device: {device.upper()} | Workers: {max_workers}")
    else:
        max_workers = 8  # default
        print(f"Using default {max_workers} workers")
    
    # Determine which problems to run
    all_problems = ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2"]
    if problems_to_run is None:
        problems_to_run = all_problems
    
    print(f"Running problems: {', '.join(problems_to_run)}")
    print(f"{'='*80}")
    
    # Create folder structure with run number
    results_dir, problem_dirs, data_dir = create_experiment_folders(
        experiment_config['NAME'], device or 'unknown', problems_to_run, run_number
    )
    
    # Initialize wandb run if enabled
    wandb_run = None
    if enable_wandb:
        try:
            # Create meaningful experiment grouping
            experiment_group = f"{device}_{experiment_config['NAME'].split('_')[1]}"  # e.g., "odin_intensive", "nyquist_standard"
            
            wandb_run = wandb.init(
                project="RNA-Folding-EA",  # Correct project name
                group=experiment_group,  # Group by device and experiment type
                name=f"{device}_{experiment_config['NAME']}_run{run_number:03d}_{datetime.datetime.now().strftime('%m%d_%H%M')}",
                config={
                    "experiment_name": experiment_config['NAME'],
                    "run_number": run_number,
                    "device": device,
                    "device_workers": max_workers,
                    "population_size": experiment_config['POPULATION_SIZE'],
                    "generations": experiment_config['GENERATIONS'],
                    "crossover_rate": experiment_config.get('CROSSOVER_RATE', 0.8),
                    "mutation_rate": experiment_config.get('MUTATION_RATE', 0.01),
                    "tournament_size": experiment_config.get('TOURNAMENT_SIZE', 3),
                    "problems": problems_to_run,
                },
                tags=[
                    device, 
                    experiment_config['NAME'].split('_')[1],  
                    f"run-{run_number}",
                    f"pop-{experiment_config['POPULATION_SIZE']}",
                    f"gen-{experiment_config['GENERATIONS']}",
                ],
                notes=f"Multi-problem RNA folding experiment on {device.upper()} - {len(problems_to_run)} problems with {experiment_config['POPULATION_SIZE']} population over {experiment_config['GENERATIONS']} generations"
            )
            print(f"Wandb tracking initialized: {wandb_run.url}")
            print(f"Project: RNA-Folding-EA")
            print(f"Group: {experiment_group}")

            # Define global metrics for the run (charts use global_step as x-axis)
            wandb_run.define_metric("global_step", summary="last")
            wandb_run.define_metric("generation", summary="last")
            for metric in ["fitness_max", "fitness_avg", "diversity"]:
                wandb_run.define_metric(metric, step_metric="global_step")
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            print("Continuing with local logging only...")
            wandb_run = None
    else:
        print("Wandb tracking disabled")
    
    # Load problem configurations
    config_file = "config/device_experiments.yml"
    with open(config_file, 'r') as f:
        full_config = yaml.safe_load(f)
    problem_configs = full_config['PROBLEMS']
    
    # Track overall experiment time
    import time
    experiment_start_time = time.time()
    
    all_results = {}
    generation_offset = 0
    
    # Create experiment log file and ensure it exists
    log_file = results_dir / "experiment_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"RNA Folding EA Experiment Log\n")
        f.write(f"Experiment: {experiment_config['NAME']} (Run #{run_number})\n")
        f.write(f"Device: {device or 'unknown'}\n")
        f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters: Pop={experiment_config['POPULATION_SIZE']}, Gen={experiment_config['GENERATIONS']}\n")
        f.write(f"Problems: {', '.join(problems_to_run)}\n")
        f.write(f"{'='*80}\n\n")
    
    print(f"Experiment log: {log_file}")
    
    # Run each problem
    for problem_id in problems_to_run:
        if problem_id not in problem_configs:
            print(f"Warning: Problem {problem_id} not found in configuration!")
            continue
            
        problem_config = problem_configs[problem_id]
        problem_dir = problem_dirs[problem_id]
        
        try:
            # Log problem start to file
            with open(log_file, 'a') as f:
                f.write(f"Starting Problem {problem_id} at {datetime.datetime.now().strftime('%H:%M:%S')}\n")
                f.write(f"Structure: {problem_config['STRUCTURE_CONSTRAINT'][:50]}...\n")
                f.write(f"Sequence:  {problem_config['SEQUENCE_CONSTRAINT'][:50]}...\n")
            
            result = run_single_problem(
                problem_id, problem_config, experiment_config,
                device, max_workers, problem_dir, wandb_run, generation_offset
            )
            all_results[problem_id] = result

            # Advance offset so global step grows monotonically across problems
            generation_offset += experiment_config['GENERATIONS']
            
            # Log completion to file
            with open(log_file, 'a') as f:
                f.write(f"Completed Problem {problem_id} at {datetime.datetime.now().strftime('%H:%M:%S')}\n")
                f.write(f"Best fitness: {result['best_fitness']:.4f}, Runtime: {result['total_time']:.2f}s\n\n")
            
        except Exception as e:
            error_msg = f"Error running problem {problem_id}: {e}"
            print(error_msg)
            
            # Log error to file
            with open(log_file, 'a') as f:
                f.write(f"ERROR in Problem {problem_id}: {e}\n\n")
                
            if wandb_run:
                wandb_run.log({f"problem_{problem_id}/error": str(e)})
    
    # Calculate total experiment time
    total_experiment_time = time.time() - experiment_start_time
    
    # Save experiment summary 
    summary_file, output_file = save_experiment_summary(
        results_dir, experiment_config['NAME'], device, experiment_config, all_results
    )
    
    # Log simple summary to wandb summary (not charts)
    if wandb_run:
        # Use wandb.summary for experiment-level metrics (tables, not time-series charts)
        wandb_run.summary.update({
            "experiment_total_runtime_minutes": total_experiment_time / 60,
            "experiment_problems_completed": len(all_results),
            "experiment_avg_best_fitness": sum(r['best_fitness'] for r in all_results.values()) / len(all_results) if all_results else 0,
            "experiment_best_overall_fitness": max(r['best_fitness'] for r in all_results.values()) if all_results else 0
        })
        
        wandb_run.finish()
    
    print(f"\n{'='*80}")
    print(f"Experiment {experiment_config['NAME']} (Run #{run_number}) completed!")
    print(f"Total runtime: {total_experiment_time/60:.2f} minutes")
    print(f"Results saved to: {results_dir}")
    print(f"Summary: {summary_file}")
    print(f"Output: {output_file}")
    if wandb_run:
        print(f"Wandb: {wandb_run.url}")
    print(f"{'='*80}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Launch RNA folding experiments optimized for specific devices')
    parser.add_argument('config', help='YAML or JSON configuration file')
    parser.add_argument('--device', choices=['odin', 'nyquist', 'laptop', 'minipc'], 
                       help='Run experiments optimized for specific device')
    parser.add_argument('--experiment', help='Specific experiment name to run')
    parser.add_argument('--constraint', help='Single constraint to run (e.g., 1.1, 2.2)')
    parser.add_argument('--problems', nargs='+', help='Specific problems to run (e.g., 1.1 1.2 2.1)')
    parser.add_argument('--run', type=int, default=1, help='Run number for experiment tracking (default: 1)')
    parser.add_argument('--wandb', type=str, choices=['true', 'false'], default='true', 
                       help='Enable wandb tracking (default: true)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_data = load_experiment_config(args.config)
    
    # Determine which problems to run
    problems_to_run = None
    if args.constraint:
        problems_to_run = [args.constraint]
        print(f"Running single constraint: {args.constraint}")
    elif args.problems:
        problems_to_run = args.problems
        print(f"Running selected problems: {', '.join(problems_to_run)}")
    else:
        print("Running ALL 6 problems: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2")
    
    # Convert wandb string argument to boolean
    enable_wandb = args.wandb.lower() == 'true'
    
    if args.device and args.experiment:
        # Run specific experiment for specific device
        experiment = next((exp for exp in config_data['EXPERIMENTS'] 
                         if exp['NAME'] == args.experiment and exp['NAME'].startswith(args.device)), None)
        if experiment:
            print(f"Running specific experiment: {args.experiment} on device: {args.device} (Run #{args.run})")
            run_experiment(experiment, device=args.device, problems_to_run=problems_to_run, run_number=args.run, enable_wandb=enable_wandb)
        else:
            print(f"Experiment '{args.experiment}' not found for device '{args.device}'!")
            device_experiments = [exp['NAME'] for exp in config_data['EXPERIMENTS'] 
                                if exp['NAME'].startswith(args.device)]
            print(f"Available experiments for {args.device}: {', '.join(device_experiments)}")
            sys.exit(1)
            
    elif args.device:
        # Run all experiments for specific device
        device_experiments = [exp for exp in config_data['EXPERIMENTS'] 
                            if exp['NAME'].startswith(args.device)]
        if not device_experiments:
            print(f"No experiments found for device '{args.device}'!")
            print("Available devices: odin, nyquist, laptop, minipc")
            sys.exit(1)
        
        print(f"Running {len(device_experiments)} experiments for device: {args.device} (Run #{args.run})")
        for experiment in device_experiments:
            run_experiment(experiment, device=args.device, problems_to_run=problems_to_run, run_number=args.run, enable_wandb=enable_wandb)
            
    elif args.experiment:
        # Run specific experiment
        experiment = next((exp for exp in config_data['EXPERIMENTS'] if exp['NAME'] == args.experiment), None)
        if experiment:
            # Try to detect device from experiment name
            device = None
            for dev in ['odin', 'nyquist', 'laptop', 'minipc']:
                if experiment['NAME'].startswith(dev):
                    device = dev
                    break
            run_experiment(experiment, device=device, problems_to_run=problems_to_run, run_number=args.run, enable_wandb=enable_wandb)
        else:
            print(f"Experiment '{args.experiment}' not found!")
            available_experiments = [exp['NAME'] for exp in config_data['EXPERIMENTS']]
            print(f"Available experiments: {', '.join(available_experiments)}")
            sys.exit(1)
    else:
        # Run all experiments
        print("Running ALL experiments. This may take a very long time!")
        print("Consider using --device flag for device-specific experiments.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
            
        for experiment in config_data['EXPERIMENTS']:
            # Try to detect device from experiment name
            device = None
            for dev in ['odin', 'nyquist', 'laptop', 'minipc']:
                if experiment['NAME'].startswith(dev):
                    device = dev
                    break
            run_experiment(experiment, device=device, problems_to_run=problems_to_run, run_number=args.run, enable_wandb=enable_wandb)

if __name__ == "__main__":
    main()