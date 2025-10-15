#!/usr/bin/env python3
"""
Generate Final Output Script
Processes all result folders to extract optimal solutions and clean up empty runs.

Tasks:
1. Scan all result folders in results/
2. Extract valid solutions from each run
3. Compare and select top 5 solutions per problem
4. Update final_output_solutions.csv in output/
5. Delete empty/failed run folders
6. Update all_experiments_master.csv with relevant summary data only
"""

import os
import json
import csv
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd


class SolutionProcessor:
    def __init__(self, base_dir="/home/odin/achal/Evolutionary-Algorithms-and-Machine-Learning/Assignment_1"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.output_dir = self.base_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Storage for all solutions per problem
        self.all_solutions = defaultdict(list)  # problem_id -> [(sequence, fitness, run_info), ...]
        self.run_summaries = []  # For master CSV
        self.deleted_folders = []
        self.processed_folders = []
        
    def process_all_runs(self):
        """Main processing function"""
        print(f"[SCAN] Scanning results directory: {self.results_dir}")
        
        if not self.results_dir.exists():
            print("[ERROR] Results directory not found!")
            return
            
        run_folders = [d for d in self.results_dir.iterdir() if d.is_dir() and not d.name.endswith('.log')]
        print(f"[INFO] Found {len(run_folders)} run folders to process")
        
        for run_folder in sorted(run_folders):
            self.process_single_run(run_folder)
            
        print(f"\n[SUMMARY] Processing Results:")
        print(f"   [PROCESSED] {len(self.processed_folders)} folders")
        print(f"   [DELETED] {len(self.deleted_folders)} empty folders")
        print(f"   [SOLUTIONS] Found for problems: {list(self.all_solutions.keys())}")
        
        # Generate outputs
        self.generate_final_output()
        self.update_master_csv()
        
    def process_single_run(self, run_folder):
        """Process a single run folder"""
        try:
            run_info = self.extract_run_info(run_folder.name)
            has_valid_data = False
            run_solutions = {}
            
            # Check for output_results.csv
            output_csv = run_folder / "output_results.csv"
            if output_csv.exists():
                solutions_from_csv = self.extract_solutions_from_csv(output_csv)
                if solutions_from_csv:
                    has_valid_data = True
                    run_solutions.update(solutions_from_csv)
            
            # Check problem folders for additional data
            problem_folders = [d for d in run_folder.iterdir() if d.is_dir() and d.name.startswith('problem_')]
            
            for problem_folder in problem_folders:
                problem_id = problem_folder.name.replace('problem_', '')
                stats_file = problem_folder / f"problem_{problem_id}_stats.json"
                
                if stats_file.exists():
                    stats = self.load_stats(stats_file)
                    if stats and (stats.get('best_fitness', 0) > 0 or stats.get('top_sequences')):
                        has_valid_data = True
                        
                        # Extract sequences from stats if not in CSV
                        if problem_id not in run_solutions and stats.get('top_sequences'):
                            run_solutions[problem_id] = stats['top_sequences']
                        
                        # Update run_info with problem-specific data
                        run_info[f'problem_{problem_id}_fitness'] = stats.get('best_fitness', 0)
                        run_info[f'problem_{problem_id}_generation'] = stats.get('generation_found', 0)
                        run_info[f'problem_{problem_id}_runtime'] = stats.get('total_time', 0)
            
            if has_valid_data:
                self.processed_folders.append(run_folder.name)
                
                # Add solutions to global collection
                for problem_id, sequences in run_solutions.items():
                    if sequences:  # Only if we have actual sequences
                        for seq_data in sequences:
                            if isinstance(seq_data, dict):
                                sequence = seq_data.get('sequence', '')
                                fitness = seq_data.get('fitness', 0)
                            elif isinstance(seq_data, (list, tuple)) and len(seq_data) >= 2:
                                sequence, fitness = seq_data[0], seq_data[1]
                            else:
                                sequence = str(seq_data)
                                fitness = 1.0  # Default fitness for valid sequences
                            
                            if sequence and len(sequence.strip()) > 0:
                                self.all_solutions[problem_id].append({
                                    'sequence': sequence,
                                    'fitness': fitness,
                                    'run_folder': run_folder.name,
                                    'timestamp': run_info.get('timestamp', ''),
                                    'device': run_info.get('device', ''),
                                    'experiment': run_info.get('experiment_name', '')
                                })
                
                # Store run summary for master CSV
                self.run_summaries.append(run_info)
                print(f"[VALID] {run_folder.name}: Valid data found")
                
            else:
                # Delete empty folder
                try:
                    shutil.rmtree(run_folder)
                    self.deleted_folders.append(run_folder.name)
                    print(f"[DELETE] {run_folder.name}: Deleted (no valid data)")
                except Exception as e:
                    print(f"[WARNING] {run_folder.name}: Failed to delete - {e}")
                    
        except Exception as e:
            print(f"[ERROR] {run_folder.name}: Error processing - {e}")
    
    def extract_run_info(self, folder_name):
        """Extract run information from folder name"""
        # Format: 2025-10-13_08-43-04_device_experiment_runXXX
        parts = folder_name.split('_')
        
        info = {
            'timestamp': '',
            'device': '',
            'experiment_name': '',
            'run_number': 1,
            'folder_name': folder_name
        }
        
        if len(parts) >= 4:
            # Extract timestamp
            if len(parts) >= 2:
                info['timestamp'] = f"{parts[0]}_{parts[1]}"
            
            # Extract device (usually 3rd part)
            if len(parts) >= 3:
                info['device'] = parts[2]
            
            # Extract experiment name and run number
            experiment_parts = parts[3:]
            run_num = 1
            
            # Check if last part is runXXX
            if experiment_parts and experiment_parts[-1].startswith('run'):
                try:
                    run_num = int(experiment_parts[-1].replace('run', ''))
                    experiment_parts = experiment_parts[:-1]
                except:
                    pass
            
            info['experiment_name'] = '_'.join(experiment_parts)
            info['run_number'] = run_num
        
        return info
    
    def extract_solutions_from_csv(self, csv_path):
        """Extract solutions from output_results.csv"""
        solutions = {}
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    problem_id = row.get('id', '').strip()
                    if problem_id:
                        sequences = []
                        for i in range(1, 6):  # result_1 to result_5
                            seq = row.get(f'result_{i}', '').strip()
                            if seq:
                                sequences.append({
                                    'sequence': seq,
                                    'fitness': 1.0 - (i-1) * 0.05,  # Estimate fitness based on rank
                                    'rank': i
                                })
                        if sequences:
                            solutions[problem_id] = sequences
        except Exception as e:
            print(f"[WARNING] Error reading CSV {csv_path}: {e}")
        
        return solutions
    
    def load_stats(self, stats_path):
        """Load statistics from JSON file"""
        try:
            with open(stats_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Error reading stats {stats_path}: {e}")
            return None
    
    def generate_final_output(self):
        """Generate final_output_solutions.csv with top 5 solutions per problem"""
        output_file = self.output_dir / "final_output_solutions.csv"
        
        # Define all possible problems
        all_problems = ['1.1', '1.2', '2.1', '2.2', '3.1', '3.2']
        
        final_results = []
        
        for problem_id in all_problems:
            if problem_id in self.all_solutions:
                # Remove duplicates by sequence and sort by fitness (descending)
                unique_solutions = {}
                for sol in self.all_solutions[problem_id]:
                    seq = sol['sequence'].strip()
                    if seq and (seq not in unique_solutions or sol['fitness'] > unique_solutions[seq]['fitness']):
                        unique_solutions[seq] = sol
                
                sorted_solutions = sorted(
                    unique_solutions.values(), 
                    key=lambda x: x['fitness'], 
                    reverse=True
                )
                
                top_5 = sorted_solutions[:5]
                result_row = {'id': problem_id}
                
                for i, solution in enumerate(top_5, 1):
                    result_row[f'result_{i}'] = solution['sequence']
                
                # Fill empty slots
                for i in range(len(top_5) + 1, 6):
                    result_row[f'result_{i}'] = ''
                    
                final_results.append(result_row)
                print(f"[OUTPUT] Problem {problem_id}: {len(top_5)} unique solutions (best fitness: {top_5[0]['fitness']:.4f})")
                
            else:
                # No solutions found for this problem
                result_row = {'id': problem_id}
                for i in range(1, 6):
                    result_row[f'result_{i}'] = ''
                final_results.append(result_row)
                print(f"[OUTPUT] Problem {problem_id}: No solutions found")
        
        # Write final output
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['id'] + [f'result_{i}' for i in range(1, 6)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_results)
        
        print(f"[SAVE] Final output saved to: {output_file}")
    
    def update_master_csv(self):
        """Update all_experiments_master.csv with relevant summary data only"""
        master_file = self.output_dir / "all_experiments_master.csv"
        
        # Define relevant columns only
        columns = [
            'timestamp', 'experiment_name', 'device', 'run_number',
            'total_runtime_minutes', 'problems_with_solutions', 
            'best_overall_fitness', 'total_solutions_found',
            'problem_1.1_fitness', 'problem_1.2_fitness', 
            'problem_2.1_fitness', 'problem_2.2_fitness',
            'problem_3.1_fitness', 'problem_3.2_fitness'
        ]
        
        # Prepare data
        master_data = []
        for run_info in self.run_summaries:
            # Calculate summary metrics
            problem_fitnesses = []
            problems_with_solutions = 0
            total_solutions = 0
            
            for pid in ['1.1', '1.2', '2.1', '2.2', '3.1', '3.2']:
                fitness = run_info.get(f'problem_{pid}_fitness', 0)
                if fitness > 0:
                    problems_with_solutions += 1
                    problem_fitnesses.append(fitness)
                
                if pid in self.all_solutions:
                    # Count solutions from this specific run
                    run_solutions = [s for s in self.all_solutions[pid] 
                                   if s['run_folder'] == run_info['folder_name']]
                    total_solutions += len(run_solutions)
            
            # Calculate runtime (estimate from folder name if not available)
            runtime = run_info.get('total_runtime_minutes', 0)
            if runtime == 0:
                # Estimate from problem runtimes
                runtime = sum(run_info.get(f'problem_{pid}_runtime', 0) for pid in ['1.1', '1.2', '2.1', '2.2', '3.1', '3.2']) / 60
            
            row = {
                'timestamp': run_info.get('timestamp', ''),
                'experiment_name': run_info.get('experiment_name', ''),
                'device': run_info.get('device', ''),
                'run_number': run_info.get('run_number', 1),
                'total_runtime_minutes': round(runtime, 2),
                'problems_with_solutions': problems_with_solutions,
                'best_overall_fitness': max(problem_fitnesses) if problem_fitnesses else 0,
                'total_solutions_found': total_solutions,
                'problem_1.1_fitness': run_info.get('problem_1.1_fitness', 0),
                'problem_1.2_fitness': run_info.get('problem_1.2_fitness', 0),
                'problem_2.1_fitness': run_info.get('problem_2.1_fitness', 0),
                'problem_2.2_fitness': run_info.get('problem_2.2_fitness', 0),
                'problem_3.1_fitness': run_info.get('problem_3.1_fitness', 0),
                'problem_3.2_fitness': run_info.get('problem_3.2_fitness', 0),
            }
            
            master_data.append(row)
        
        # Sort by timestamp
        master_data.sort(key=lambda x: x['timestamp'])
        
        # Write master CSV
        if master_data:
            df = pd.DataFrame(master_data)
            df.to_csv(master_file, index=False)
            print(f"[SAVE] Master CSV updated: {master_file}")
            print(f"   [INFO] {len(master_data)} experiments summarized")
        else:
            print("[WARNING] No valid experiment data to write to master CSV")


def main():
    """Main execution function"""
    print("[START] Starting Output Generation Process")
    print("=" * 60)
    
    processor = SolutionProcessor()
    processor.process_all_runs()
    
    print("\n" + "=" * 60)
    print("[COMPLETE] Output generation completed!")
    print(f"[INFO] Check output folder: {processor.output_dir}")
    print("[FILES] Files updated:")
    print("   - final_output_solutions.csv")
    print("   - all_experiments_master.csv")


if __name__ == "__main__":
    main()