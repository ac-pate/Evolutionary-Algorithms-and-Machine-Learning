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
                result['avg_fitness_final'], result['avg_diversity_final'], 
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

def run_single_problem(problem_id, problem_config, experiment_config, device, max_workers, problem_dir, wandb_run, generation_count):
    """Run EA for a single problem and return results"""
    print(f"\n{'='*60}")
    print(f"Running Problem {problem_id}")
    print(f"Structure: {problem_config['STRUCTURE_CONSTRAINT'][:50]}...")
    print(f"Sequence:  {problem_config['SEQUENCE_CONSTRAINT'][:50]}...")
    print(f"{'='*60}")
    
    # Create EA instance
    ea = RNAFoldingEA(
        population_size=experiment_config['POPULATION_SIZE'],
        generations=experiment_config['GENERATIONS'],
        sequence_constraint=problem_config['SEQUENCE_CONSTRAINT'],
        structure_constraint=problem_config['STRUCTURE_CONSTRAINT'],
        max_workers=max_workers
    )
    
    # Override default parameters if specified
    if 'CROSSOVER_RATE' in experiment_config:
        ea.crossover_rate = experiment_config['CROSSOVER_RATE']
    if 'MUTATION_RATE' in experiment_config:
        ea.mutation_rate = experiment_config['MUTATION_RATE']
    if 'TOURNAMENT_SIZE' in experiment_config:
        ea.tournament_size = experiment_config['TOURNAMENT_SIZE']
    
    # Add progress monitoring
    try:
        from progress_monitor import add_progress_monitoring
        ea = add_progress_monitoring(ea)
        print("Enhanced progress monitoring enabled")
    except ImportError:
        print("Progress monitoring not available, running standard EA...")
    
    # Log experiment configuration as metadata (not time-series data)
    if wandb_run:
        # Use wandb.config for static configuration parameters
        wandb_run.config.update({
            f"problem_{problem_id}_population_size": experiment_config['POPULATION_SIZE'],
            f"problem_{problem_id}_generations": experiment_config['GENERATIONS'],
            f"problem_{problem_id}_crossover_rate": experiment_config.get('CROSSOVER_RATE', 0.8),
            f"problem_{problem_id}_mutation_rate": experiment_config.get('MUTATION_RATE', 0.01),
            f"problem_{problem_id}_tournament_size": experiment_config.get('TOURNAMENT_SIZE', 3),
            f"problem_{problem_id}_sequence_length": len(problem_config['SEQUENCE_CONSTRAINT']),
            f"problem_{problem_id}_sequence_constraint": problem_config['SEQUENCE_CONSTRAINT'],
            f"problem_{problem_id}_structure_constraint": problem_config['STRUCTURE_CONSTRAINT']
        })
    
    # Track start time
    import time
    start_time = time.time()
    
    # Optimized wandb callback - log every 10 generations for performance
    log_interval = max(1, experiment_config['GENERATIONS'] // 10)  # Log at most 10 times during evolution
    
    if wandb_run:
        def optimized_wandb_callback(generation, best_fitness, avg_fitness, diversity):
            # Only log every log_interval generations to reduce overhead
            if generation % log_interval == 0 or generation == experiment_config['GENERATIONS'] - 1:
                print(f"Logging to wandb: Gen {generation}, Best: {best_fitness:.4f}")  # Debug print
                wandb_run.log({
                    f"problem_{problem_id}/generation": generation,
                    f"problem_{problem_id}/best_fitness": best_fitness,
                    f"problem_{problem_id}/avg_fitness": avg_fitness,
                    f"problem_{problem_id}/diversity": diversity
                })
        
        # Add callback to EA - verify it's actually added
        if hasattr(ea, 'add_callback'):
            ea.add_callback(optimized_wandb_callback)
            print(f"Wandb callback registered for problem {problem_id}")
        else:
            print(f"ERROR: EA does not have add_callback method!")
    else:
        print(f"ERROR: No wandb run initialized for problem {problem_id}")
    
    ea.run_evolution()
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Get results
    best_individuals = ea.get_best_individuals()
    fitness_history = ea.fitness_history
    
    # Find generation where best fitness was first achieved
    best_fitness = max(fitness_history) if fitness_history else 0
    generation_found = next((i for i, f in enumerate(fitness_history) if f == best_fitness), 0)
    
    # Get top 5 valid sequences above fitness threshold
    valid_sequences = [ind for ind in best_individuals if ind['fitness'] > 0.5]  # threshold
    valid_sequences.sort(key=lambda x: x['fitness'], reverse=True)
    top_sequences = [seq['sequence'] for seq in valid_sequences[:5]]
    
    # Save problem-specific results
    problem_results_file = problem_dir / f"problem_{problem_id}_results.txt"
    ea.save_results(str(problem_results_file))
    
    # Save problem-specific stats
    problem_stats = {
        'problem_id': problem_id,
        'best_fitness': best_fitness,
        'generation_found': generation_found,
        'avg_fitness_final': sum(fitness_history[-10:]) / 10 if len(fitness_history) >= 10 else (fitness_history[-1] if fitness_history else 0),
        'total_time': runtime,
        'top_sequences': top_sequences,
        'fitness_history': fitness_history
    }
    
    problem_stats_file = problem_dir / f"problem_{problem_id}_stats.json"
    with open(problem_stats_file, 'w') as f:
        json.dump(problem_stats, f, indent=2)
    
    # Log final results to wandb with enhanced tracking
    if wandb_run:
        # Create wandb Table for this problem's results
        problem_table = wandb.Table(columns=[
            "Problem_ID", "Best_Fitness", "Avg_Fitness_Final", "Generation_Found", 
            "Runtime_Seconds", "Valid_Sequences", "Top_Sequence", "Diversity_Final"
        ])
        
        # Calculate final diversity from last generation
        final_diversity = ea.calculate_diversity(ea.population) if hasattr(ea, 'calculate_diversity') and ea.population else 0
        
        problem_table.add_data(
            problem_id,
            best_fitness,
            problem_stats['avg_fitness_final'],
            generation_found,
            runtime,
            len(valid_sequences),
            top_sequences[0] if top_sequences else "None",
            final_diversity
        )
        
        # Log the table and final metrics
        wandb_run.log({
            f"problem_{problem_id}/final_best_fitness": best_fitness,
            f"problem_{problem_id}/generation_found": generation_found,
            f"problem_{problem_id}/runtime_seconds": runtime,
            f"problem_{problem_id}/valid_sequences_count": len(valid_sequences),
            f"problem_{problem_id}/final_diversity": final_diversity,
            f"problem_{problem_id}_results_table": problem_table
        })
    
    print(f"Problem {problem_id} completed!")
    print(f"Best fitness: {best_fitness:.4f} (found at generation {generation_found})")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Valid sequences found: {len(valid_sequences)}")
    
    return problem_stats
def run_experiment(experiment_config, device=None, problems_to_run=None, run_number=1):
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
    
    # Initialize wandb run with enhanced organization
    wandb_run = None
    try:
        # Create meaningful experiment grouping
        experiment_group = f"{device}_{experiment_config['NAME'].split('_')[1]}"  # e.g., "odin_intensive", "nyquist_standard"
        job_type = "multi-problem-ea"
        
        wandb_run = wandb.init(
            project="rna-folding-evolutionary-algorithms",  # More descriptive project name
            group=experiment_group,  # Group by device and experiment type
            job_type=job_type,
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
                "problems_count": len(problems_to_run),
                "max_workers": max_workers,
                "assignment": "COEN432_Assignment1",
                "algorithm": "evolutionary_algorithm",
                "problem_type": "inverse_rna_folding"
            },
            tags=[
                device, 
                experiment_config['NAME'].split('_')[1],  # e.g., "intensive", "standard", "mobile"
                "multi-problem", 
                "rna-folding", 
                f"run-{run_number}",
                f"pop-{experiment_config['POPULATION_SIZE']}",
                f"gen-{experiment_config['GENERATIONS']}",
                "COEN432"
            ],
            notes=f"Multi-problem RNA folding experiment on {device.upper()} - {len(problems_to_run)} problems with {experiment_config['POPULATION_SIZE']} population over {experiment_config['GENERATIONS']} generations"
        )
        print(f"Wandb tracking initialized: {wandb_run.url}")
        print(f"Project: rna-folding-evolutionary-algorithms")
        print(f"Group: {experiment_group}")
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
        print("Continuing with local logging only...")
    
    # Load problem configurations
    config_file = "config/device_experiments.yml"
    with open(config_file, 'r') as f:
        full_config = yaml.safe_load(f)
    problem_configs = full_config['PROBLEMS']
    
    # Track overall experiment time
    import time
    experiment_start_time = time.time()
    
    all_results = {}
    total_generation_count = experiment_config['GENERATIONS'] * len(problems_to_run)
    
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
                device, max_workers, problem_dir, wandb_run, total_generation_count
            )
            all_results[problem_id] = result
            
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
    
    # Save experiment summary and output files
    summary_file, output_file = save_experiment_summary(
        results_dir, experiment_config['NAME'], device, experiment_config, all_results
    )
    
    # Update master data file
    master_file = update_master_data_file(
        data_dir, experiment_config['NAME'], device, experiment_config, 
        all_results, total_experiment_time, run_number
    )
    
    # Log summary to wandb with comprehensive tables
    if wandb_run:
        # Create experiment summary table
        experiment_table = wandb.Table(columns=[
            "Device", "Experiment_Name", "Run_Number", "Problem_ID", "Best_Fitness", 
            "Generation_Found", "Runtime_Seconds", "Population_Size", "Generations",
            "Mutation_Rate", "Crossover_Rate", "Tournament_Size", "Valid_Solutions"
        ])
        
        # Add each problem result to the table
        for problem_id, result in all_results.items():
            experiment_table.add_data(
                device or "unknown",
                experiment_config['NAME'],
                run_number,
                problem_id,
                result['best_fitness'],
                result['generation_found'],
                result['total_time'],
                experiment_config['POPULATION_SIZE'],
                experiment_config['GENERATIONS'],
                experiment_config.get('MUTATION_RATE', 0.01),
                experiment_config.get('CROSSOVER_RATE', 0.8),
                experiment_config.get('TOURNAMENT_SIZE', 3),
                len(result.get('top_sequences', []))
            )
        
        # Calculate and log comprehensive experiment metrics
        avg_best_fitness = sum(r['best_fitness'] for r in all_results.values()) / len(all_results) if all_results else 0
        best_overall_fitness = max(r['best_fitness'] for r in all_results.values()) if all_results else 0
        total_valid_solutions = sum(len(r.get('top_sequences', [])) for r in all_results.values())
        avg_generation_found = sum(r['generation_found'] for r in all_results.values()) / len(all_results) if all_results else 0
        
        wandb_run.log({
            "experiment/total_runtime_minutes": total_experiment_time / 60,
            "experiment/problems_completed": len(all_results),
            "experiment/avg_best_fitness": avg_best_fitness,
            "experiment/best_overall_fitness": best_overall_fitness,
            "experiment/total_valid_solutions": total_valid_solutions,
            "experiment/avg_generation_found": avg_generation_found,
            "experiment/efficiency_score": (avg_best_fitness * len(all_results)) / (total_experiment_time / 60),  # Fitness per minute
            "experiment_summary_table": experiment_table
        })
        
        # Upload result files and master CSV as artifacts
        artifact = wandb.Artifact(f"experiment_results_{experiment_config['NAME']}_run{run_number:03d}", type="results")
        artifact.add_file(str(summary_file))
        artifact.add_file(str(output_file))
        
        # Add master CSV if it exists
        if master_file.exists():
            artifact.add_file(str(master_file))
            
            # Also create a master CSV table in wandb for easy viewing
            try:
                import pandas as pd
                master_df = pd.read_csv(master_file)
                master_table = wandb.Table(dataframe=master_df)
                wandb_run.log({"master_experiments_table": master_table})
                print(f"Master experiments table uploaded to wandb")
            except Exception as e:
                print(f"Warning: Could not create master table: {e}")
        
        wandb_run.log_artifact(artifact)
        
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
    
    if args.device and args.experiment:
        # Run specific experiment for specific device
        experiment = next((exp for exp in config_data['EXPERIMENTS'] 
                         if exp['NAME'] == args.experiment and exp['NAME'].startswith(args.device)), None)
        if experiment:
            print(f"Running specific experiment: {args.experiment} on device: {args.device} (Run #{args.run})")
            run_experiment(experiment, device=args.device, problems_to_run=problems_to_run, run_number=args.run)
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
            run_experiment(experiment, device=args.device, problems_to_run=problems_to_run, run_number=args.run)
            
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
            run_experiment(experiment, device=device, problems_to_run=problems_to_run, run_number=args.run)
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
            run_experiment(experiment, device=device, problems_to_run=problems_to_run, run_number=args.run)

if __name__ == "__main__":
    main()