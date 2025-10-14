#!/usr/bin/env python3
"""
RNA Folding EA - Assignment Runner
Processes CSV input file and produces CSV output file as required
Achal Patel - 40227663
"""

import json
import sys
import os
import argparse
import datetime
import csv
from pathlib import Path
import time

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration and modules
import config
from src.rna_folding_ea import RNAFoldingEA
from src.csv_processor import CSVProcessor

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Running without experiment tracking.")
    WANDB_AVAILABLE = False

def run_single_problem(problem_instance: dict, run_id: str = None, device_config: dict = None) -> dict:
    """
    Run EA for a single problem instance
    
    Args:
        problem_instance: Dict with 'id', 'structure', 'iupac'
        run_id: Optional run identifier
        device_config: Optional device configuration override
        
    Returns:
        Dict with results: {'id': str, 'sequences': List[str], 'fitness_scores': List[float]}
    """
    # Use device config if provided, otherwise use default config
    if device_config is None:
        device_config = {
            'population_size': config.POPULATION_SIZE,
            'generations': config.GENERATIONS,
            'max_workers': config.MAX_WORKERS
        }
    problem_id = problem_instance['id']
    structure_constraint = problem_instance['structure']
    sequence_constraint = problem_instance['iupac']
    
    print(f"\n{'='*60}")
    print(f"SOLVING PROBLEM: {problem_id}")
    print(f"Structure: {structure_constraint}")
    print(f"IUPAC:     {sequence_constraint}")
    print(f"Length:    {len(structure_constraint)}")
    print(f"{'='*60}")
    
    # Create EA instance with device-specific parameters
    ea = RNAFoldingEA(
        population_size=device_config['population_size'],
        generations=device_config['generations'],
        sequence_constraint=sequence_constraint,
        structure_constraint=structure_constraint,
        max_workers=device_config['max_workers'],
        elite_percentage=config.ELITE_PERCENTAGE,
        enable_cache_preloading=config.ENABLE_CACHE_PRELOADING
    )
    
    # Set additional parameters from config
    ea.mutation_rate = config.MUTATION_RATE
    ea.early_termination_fitness = config.EARLY_TERMINATION_FITNESS
    ea.high_fitness_streak_threshold = config.HIGH_FITNESS_STREAK_THRESHOLD
    ea.mutation_rate_boost_factor = config.MUTATION_RATE_BOOST_FACTOR
    ea.mutation_boost_generations = config.MUTATION_BOOST_GENERATIONS
    ea.fitness_threshold_for_boost = config.FITNESS_THRESHOLD_FOR_BOOST
    
    # Initialize wandb if available and desired
    wandb_run = None
    if WANDB_AVAILABLE and config.VERBOSE_OUTPUT:
        try:
            wandb_run = wandb.init(
                project="rna-folding-ea-assignment",
                name=f"{problem_id}_{run_id or 'run'}",
                config={
                    "problem_id": problem_id,
                    "structure_constraint": structure_constraint,
                    "sequence_constraint": sequence_constraint,
                    "population_size": device_config['population_size'],
                    "generations": device_config['generations'],
                    "mutation_rate": config.MUTATION_RATE,
                    "max_workers": device_config['max_workers']
                }
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            wandb_run = None
    
    try:
        # Run the evolutionary algorithm
        start_time = time.time()
        ea.run_evolution()
        end_time = time.time()
        
        # Get diverse results
        diverse_results = ea.get_diverse_top_sequences(
            num_sequences=config.NUM_DIVERSE_SEQUENCES,
            min_diversity_threshold=config.MIN_DIVERSITY_THRESHOLD,
            verbose=False
        )
        
        if not diverse_results:
            print(f"Warning: No valid results found for problem {problem_id}")
            return {
                'id': problem_id,
                'sequences': [],
                'fitness_scores': []
            }
        
        # Extract sequences and fitness scores
        sequences = [seq for seq, _ in diverse_results]
        fitness_scores = [fitness for _, fitness in diverse_results]
        
        # Log results
        runtime = end_time - start_time
        print(f"\nPROBLEM {problem_id} COMPLETED:")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Found {len(sequences)} diverse sequences")
        print(f"Best fitness: {max(fitness_scores):.4f}")
        print(f"Diversity: {ea.calculate_diversity(sequences):.4f}")
        
        # Log to wandb if available
        if wandb_run:
            wandb.log({
                "runtime": runtime,
                "best_fitness": max(fitness_scores),
                "num_sequences": len(sequences),
                "diversity": ea.calculate_diversity(sequences),
                "termination_reason": getattr(ea, 'termination_reason', 'Normal completion')
            })
            wandb.finish()
        
        return {
            'id': problem_id,
            'sequences': sequences,
            'fitness_scores': fitness_scores,
            'runtime': runtime,
            'termination_reason': getattr(ea, 'termination_reason', 'Normal completion')
        }
        
    except Exception as e:
        print(f"Error solving problem {problem_id}: {e}")
        if wandb_run:
            wandb.finish()
        return {
            'id': problem_id,
            'sequences': [],
            'fitness_scores': [],
            'error': str(e)
        }

def main():
    """Main function to process CSV input and produce CSV output"""
    parser = argparse.ArgumentParser(description='RNA Folding EA - Assignment Runner')
    parser.add_argument('--input', '-i', 
                       default=config.INPUT_CSV_FILE,
                       help=f'Input CSV file (default: {config.INPUT_CSV_FILE})')
    parser.add_argument('--experiment', '-e',
                       default="test_run",
                       help='Experiment name for output file naming (default: test_run)')
    parser.add_argument('--run-id', '--id',
                       default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                       help='Run identifier for tracking')
    parser.add_argument('--problems', '-p',
                       help='Comma-separated list of problem IDs to solve (default: all)')
    parser.add_argument('--device', '-d',
                       choices=['odin', 'nyquist', 'laptop', 'minipc'],
                       help='Device configuration to use (overrides config settings)')
    
    args = parser.parse_args()
    
    # Apply device-specific configuration if specified
    current_config = {
        'population_size': config.POPULATION_SIZE,
        'generations': config.GENERATIONS,
        'max_workers': config.MAX_WORKERS
    }
    
    if args.device and args.device in config.DEVICE_CONFIGURATIONS:
        device_config = config.DEVICE_CONFIGURATIONS[args.device]
        current_config.update(device_config)
        print(f"Using device configuration for '{args.device.upper()}':")
        print(f"  - Max workers: {current_config['max_workers']}")
        print(f"  - Population size: {current_config['population_size']}")
        print(f"  - Generations: {current_config['generations']}")
    
    # Auto-generate output filename based on experiment name and timestamp
    output_filename = f"EA_Assignment_1_output-sheet1_{args.experiment}_{args.run_id}.csv"
    output_path = os.path.join("output", output_filename)
    
    print("="*80)
    print("RNA FOLDING EVOLUTIONARY ALGORITHM - ASSIGNMENT RUNNER")
    print("Achal Patel - 40227663")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Output file: {output_path}")
    print(f"Experiment: {args.experiment}")
    print(f"Run ID: {args.run_id}")
    print(f"Configuration loaded from: config.py")
    print("="*80)
    
    # Initialize CSV processor
    processor = CSVProcessor(args.input, output_path)
    
    try:
        # Read problem instances
        problem_instances = processor.read_problem_instances()
        
        # Filter problems if specified
        if args.problems:
            requested_ids = [pid.strip() for pid in args.problems.split(',')]
            problem_instances = [p for p in problem_instances if p['id'] in requested_ids]
            print(f"Filtering to {len(problem_instances)} requested problems: {requested_ids}")
        
        if not problem_instances:
            print("No problems to solve!")
            return
        
        # Validate all problems first
        print(f"\nValidating {len(problem_instances)} problem instances...")
        valid_problems = []
        for problem in problem_instances:
            is_valid, error_msg = processor.validate_problem_instance(problem)
            if is_valid:
                valid_problems.append(problem)
                print(f"✓ Problem {problem['id']}: Valid")
            else:
                print(f"✗ Problem {problem['id']}: {error_msg}")
        
        if not valid_problems:
            print("No valid problems found!")
            return
        
        print(f"\nSolving {len(valid_problems)} valid problems...")
        
        # Solve each problem
        all_results = []
        for i, problem in enumerate(valid_problems, 1):
            print(f"\n[{i}/{len(valid_problems)}] Processing problem {problem['id']}...")
            
            result = run_single_problem(problem, f"{args.run_id}_{problem['id']}", current_config)
            all_results.append(result)
            
            # Show progress
            sequences_found = len(result.get('sequences', []))
            if sequences_found > 0:
                print(f"✓ Problem {problem['id']}: Found {sequences_found} sequences")
            else:
                print(f"✗ Problem {problem['id']}: No sequences found")
        
        # Write results to CSV
        print(f"\nWriting results to {output_path}...")
        processor.write_results(all_results)
        
        # Summary
        total_sequences = sum(len(r.get('sequences', [])) for r in all_results)
        successful_problems = sum(1 for r in all_results if len(r.get('sequences', [])) > 0)
        
        print(f"\n{'='*80}")
        print("ASSIGNMENT COMPLETED")
        print(f"{'='*80}")
        print(f"Problems processed: {len(all_results)}")
        print(f"Problems solved successfully: {successful_problems}")
        print(f"Total sequences found: {total_sequences}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()