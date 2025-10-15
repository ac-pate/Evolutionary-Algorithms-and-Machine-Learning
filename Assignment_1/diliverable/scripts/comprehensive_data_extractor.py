#!/usr/bin/env python3
"""
Comprehensive Data Extractor and Visualizer
Extracts ALL useful information from results/, wandb/, and other sources
Creates comprehensive 6-constraint grid plots and cleans up useless files.
"""

import os
import json
import csv
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import glob
import re

class ComprehensiveDataExtractor:
    def __init__(self, base_dir="/home/odin/achal/Evolutionary-Algorithms-and-Machine-Learning/Assignment_1"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.wandb_dir = self.base_dir / "wandb"
        self.output_dir = self.base_dir / "output"
        self.plots_dir = self.base_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Storage for all extracted data
        self.all_experiment_data = {}  # experiment_id -> {problem_data, metadata}
        self.deleted_folders = []
        self.processed_folders = []
        self.wandb_data = {}
        
    def extract_all_data(self):
        """Main extraction function - gets data from ALL sources"""
        print(f"[SCAN] Starting comprehensive data extraction...")
        print(f"  - Results dir: {self.results_dir}")
        print(f"  - Wandb dir: {self.wandb_dir}")
        print(f"  - Output dir: {self.output_dir}")
        
        # Extract from results folders
        self.extract_from_results()
        
        # Extract from wandb logs
        self.extract_from_wandb()
        
        # Extract from output CSVs
        self.extract_from_output_files()
        
        # Clean up useless folders
        self.cleanup_useless_data()
        
        # Generate comprehensive visualizations
        self.create_comprehensive_plots()
        
        print(f"\n[SUMMARY] Extraction Results:")
        print(f"  - Processed folders: {len(self.processed_folders)}")
        print(f"  - Deleted empty folders: {len(self.deleted_folders)}")
        print(f"  - Experiments with data: {len(self.all_experiment_data)}")
        print(f"  - Wandb runs found: {len(self.wandb_data)}")
        
    def extract_from_results(self):
        """Extract data from results/ folders"""
        if not self.results_dir.exists():
            print("[WARNING] Results directory not found!")
            return
            
        run_folders = [d for d in self.results_dir.iterdir() if d.is_dir()]
        print(f"[INFO] Found {len(run_folders)} result folders")
        
        for run_folder in sorted(run_folders):
            try:
                experiment_id = self.extract_experiment_id(run_folder.name)
                has_data = False
                experiment_data = {
                    'source': 'results',
                    'folder_path': str(run_folder),
                    'timestamp': self.extract_timestamp(run_folder.name),
                    'problems': {}
                }
                
                # Look for stats files in problem folders
                problem_folders = [d for d in run_folder.iterdir() if d.is_dir() and d.name.startswith('problem_')]
                
                for problem_folder in problem_folders:
                    problem_id = problem_folder.name.replace('problem_', '')
                    stats_file = problem_folder / f"problem_{problem_id}_stats.json"
                    
                    if stats_file.exists():
                        stats = self.load_json(stats_file)
                        if stats and self.is_useful_stats(stats):
                            has_data = True
                            experiment_data['problems'][problem_id] = {
                                'stats': stats,
                                'stats_file': str(stats_file)
                            }
                
                # Check for CSV output files
                csv_files = list(run_folder.glob("*.csv"))
                if csv_files:
                    for csv_file in csv_files:
                        csv_data = self.extract_csv_data(csv_file)
                        if csv_data:
                            has_data = True
                            experiment_data['csv_data'] = csv_data
                
                if has_data:
                    self.all_experiment_data[experiment_id] = experiment_data
                    self.processed_folders.append(run_folder.name)
                    print(f"[VALID] {run_folder.name}: Found useful data")
                else:
                    # Mark for deletion
                    self.schedule_folder_deletion(run_folder)
                    
            except Exception as e:
                print(f"[ERROR] {run_folder.name}: {e}")
    
    def extract_from_wandb(self):
        """Extract evolution data from wandb logs"""
        if not self.wandb_dir.exists():
            print("[WARNING] Wandb directory not found!")
            return
            
        wandb_runs = [d for d in self.wandb_dir.iterdir() if d.is_dir() and d.name.startswith('run-')]
        print(f"[INFO] Found {len(wandb_runs)} wandb runs")
        
        for run_dir in wandb_runs:
            try:
                # Parse wandb run data
                config_file = run_dir / "files" / "config.yaml"
                log_file = run_dir / "files" / "output.log"
                summary_file = run_dir / "files" / "wandb-summary.json"
                
                wandb_data = {}
                
                # Extract config
                if config_file.exists():
                    config = self.parse_wandb_config(config_file)
                    wandb_data['config'] = config
                
                # Extract evolution data from logs
                if log_file.exists():
                    evolution_data = self.parse_wandb_logs(log_file)
                    if evolution_data:
                        wandb_data['evolution'] = evolution_data
                
                # Extract summary
                if summary_file.exists():
                    summary = self.load_json(summary_file)
                    if summary:
                        wandb_data['summary'] = summary
                
                if wandb_data:
                    run_id = run_dir.name
                    self.wandb_data[run_id] = wandb_data
                    
                    # Try to match with existing experiment data
                    self.match_wandb_to_experiment(run_id, wandb_data)
                    print(f"[WANDB] {run_id}: Extracted evolution data")
                    
            except Exception as e:
                print(f"[ERROR] Wandb {run_dir.name}: {e}")
    
    def extract_from_output_files(self):
        """Extract data from output CSV files"""
        if not self.output_dir.exists():
            return
            
        csv_files = list(self.output_dir.glob("*.csv"))
        for csv_file in csv_files:
            if "Assignment_1_output" in csv_file.name:
                csv_data = self.extract_csv_data(csv_file)
                if csv_data:
                    # Create experiment entry for CSV data
                    experiment_id = f"csv_{csv_file.stem}"
                    self.all_experiment_data[experiment_id] = {
                        'source': 'output_csv',
                        'csv_data': csv_data,
                        'file_path': str(csv_file),
                        'timestamp': self.extract_timestamp_from_filename(csv_file.name)
                    }
                    print(f"[CSV] {csv_file.name}: Extracted solution data")
    
    def parse_wandb_logs(self, log_file):
        """Parse wandb logs to extract real-time evolution data"""
        evolution_data = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'problem_id': None
        }
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
                # Look for logged data patterns
                # Pattern: generation: X, best_fitness: Y, avg_fitness: Z, diversity: W
                lines = content.split('\n')
                for line in lines:
                    if 'generation' in line and 'fitness' in line:
                        # Extract generation data
                        gen_match = re.search(r'generation["\s]*:\s*([\d.]+)', line)
                        best_match = re.search(r'best_fitness["\s]*:\s*([\d.]+)', line)
                        avg_match = re.search(r'avg_fitness["\s]*:\s*([\d.]+)', line)
                        div_match = re.search(r'diversity["\s]*:\s*([\d.]+)', line)
                        
                        if gen_match and best_match:
                            generation = float(gen_match.group(1))
                            best_fitness = float(best_match.group(1))
                            avg_fitness = float(avg_match.group(1)) if avg_match else best_fitness
                            diversity = float(div_match.group(1)) if div_match else 0.0
                            
                            evolution_data['generations'].append(generation)
                            evolution_data['best_fitness'].append(best_fitness)
                            evolution_data['avg_fitness'].append(avg_fitness)
                            evolution_data['diversity'].append(diversity)
                    
                    # Extract problem ID
                    if 'problem_id' in line:
                        prob_match = re.search(r'problem_id["\s]*:\s*["\']?([0-9.]+)["\']?', line)
                        if prob_match:
                            evolution_data['problem_id'] = prob_match.group(1)
                
                return evolution_data if evolution_data['generations'] else None
                
        except Exception as e:
            print(f"[WARNING] Error parsing wandb log: {e}")
            return None
    
    def create_comprehensive_plots(self):
        """Create comprehensive 6-constraint grid plots with ALL available data"""
        print(f"[PLOT] Creating comprehensive 6-constraint visualization...")
        
        # Organize all data by problem ID
        problem_data = defaultdict(list)
        
        for exp_id, exp_data in self.all_experiment_data.items():
            # From results stats
            if 'problems' in exp_data:
                for problem_id, prob_data in exp_data['problems'].items():
                    stats = prob_data['stats']
                    fitness_history = stats.get('fitness_history', [])
                    diversity_history = stats.get('diversity_history', [])
                    
                    if fitness_history:
                        problem_data[problem_id].append({
                            'type': 'results_stats',
                            'experiment_id': exp_id,
                            'fitness_history': fitness_history,
                            'diversity_history': diversity_history,
                            'best_fitness': stats.get('best_fitness', 0),
                            'metadata': exp_data
                        })
            
            # From wandb data
            if 'wandb_evolution' in exp_data:
                wandb_data = exp_data['wandb_evolution']
                problem_id = wandb_data.get('problem_id')
                if problem_id and wandb_data.get('generations'):
                    # Convert wandb data to standard format
                    fitness_history = [(gen, fit, avg) for gen, fit, avg in 
                                     zip(wandb_data['generations'], 
                                         wandb_data['best_fitness'],
                                         wandb_data['avg_fitness'])]
                    
                    problem_data[problem_id].append({
                        'type': 'wandb_evolution',
                        'experiment_id': exp_id,
                        'fitness_history': fitness_history,
                        'diversity_history': wandb_data.get('diversity', []),
                        'best_fitness': max(wandb_data['best_fitness']) if wandb_data['best_fitness'] else 0,
                        'metadata': exp_data
                    })
        
        # Create the 6-constraint grid plot
        self.plot_six_constraint_comprehensive(problem_data)
        
        # Create additional analysis plots
        self.plot_experiment_comparison(problem_data)
        self.plot_performance_summary(problem_data)
    
    def plot_six_constraint_comprehensive(self, problem_data):
        """Create comprehensive 6-constraint grid with fitness and diversity"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
        
        problem_ids = ['1.1', '1.2', '2.1', '2.2', '3.1', '3.2']
        colors = plt.cm.Set1(np.linspace(0, 1, 10))
        
        for i, problem_id in enumerate(problem_ids):
            ax = axes[i]
            ax2 = ax.twinx()  # Secondary y-axis for diversity
            
            if problem_id in problem_data and problem_data[problem_id]:
                experiments = problem_data[problem_id]
                
                max_fitness = 0
                max_diversity = 0
                
                for j, exp_data in enumerate(experiments):
                    color = colors[j % len(colors)]
                    exp_id = exp_data['experiment_id']
                    
                    # Plot fitness
                    fitness_history = exp_data['fitness_history']
                    if fitness_history:
                        if isinstance(fitness_history[0], (list, tuple)):
                            # Format: [(gen, max_fit, avg_fit), ...]
                            generations = [entry[0] for entry in fitness_history]
                            max_fitness_vals = [entry[1] for entry in fitness_history]
                            avg_fitness_vals = [entry[2] if len(entry) > 2 else entry[1] for entry in fitness_history]
                        else:
                            # Simple list format
                            generations = list(range(len(fitness_history)))
                            max_fitness_vals = fitness_history
                            avg_fitness_vals = fitness_history
                        
                        # Plot fitness lines
                        ax.plot(generations, max_fitness_vals, 
                               color=color, linewidth=2, alpha=0.8,
                               label=f'Max Fitness ({exp_data["type"]})' if j == 0 else '', linestyle='-')
                        
                        ax.plot(generations, avg_fitness_vals, 
                               color=color, linewidth=1, alpha=0.6,
                               linestyle=':', label=f'Avg Fitness' if j == 0 else '')
                        
                        max_fitness = max(max_fitness, max(max_fitness_vals))
                    
                    # Plot diversity
                    diversity_history = exp_data['diversity_history']
                    if diversity_history:
                        div_generations = list(range(len(diversity_history)))
                        ax2.plot(div_generations, diversity_history,
                                color=color, linewidth=2, alpha=0.5,
                                linestyle='--', label=f'Diversity' if j == 0 else '')
                        
                        max_diversity = max(max_diversity, max(diversity_history))
                
                # Formatting
                ax.set_title(f'Problem {problem_id} ({len(experiments)} runs)', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Generation')
                ax.set_ylabel('Fitness', color='blue')
                ax2.set_ylabel('Diversity', color='red')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, min(1.05, max_fitness * 1.1))
                ax2.set_ylim(0, min(1.05, max_diversity * 1.1))
                
                # Color y-axis labels
                ax.tick_params(axis='y', labelcolor='blue')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Add statistics
                best_overall = max(exp['best_fitness'] for exp in experiments)
                ax.text(0.02, 0.95, f'Best: {best_overall:.3f}', 
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                
                if i == 0:  # Add legend to first subplot
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, 
                             loc='upper right', fontsize=10)
                    
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=16, color='red')
                ax.set_title(f'Problem {problem_id}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Generation')
                ax.set_ylabel('Fitness', color='blue')
                ax2.set_ylabel('Diversity', color='red')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive RNA Folding EA Performance: All Experiments', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save plot
        output_file = self.plots_dir / 'comprehensive_six_constraint_grid.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def cleanup_useless_data(self):
        """Delete folders and files with no useful information"""
        print(f"[CLEANUP] Starting cleanup of useless data...")
        
        # Clean up empty result folders
        if self.results_dir.exists():
            for run_folder in self.results_dir.iterdir():
                if run_folder.is_dir() and run_folder.name not in self.processed_folders:
                    self.schedule_folder_deletion(run_folder)
        
        # Clean up empty wandb runs
        if self.wandb_dir.exists():
            for run_dir in self.wandb_dir.iterdir():
                if run_dir.is_dir() and run_dir.name not in self.wandb_data:
                    self.schedule_folder_deletion(run_dir)
        
        print(f"[CLEANUP] Scheduled {len(self.deleted_folders)} folders for deletion")
    
    def schedule_folder_deletion(self, folder_path):
        """Schedule a folder for deletion (safely)"""
        try:
            # Check if folder is really empty or useless
            has_useful_content = False
            
            if folder_path.is_dir():
                for item in folder_path.rglob('*'):
                    if item.is_file():
                        # Check if file contains useful data
                        if item.suffix in ['.json', '.csv'] and item.stat().st_size > 100:
                            has_useful_content = True
                            break
                        elif item.suffix in ['.png', '.jpg'] and item.stat().st_size > 1000:
                            has_useful_content = True
                            break
            
            if not has_useful_content:
                shutil.rmtree(folder_path)
                self.deleted_folders.append(folder_path.name)
                print(f"[DELETE] {folder_path.name}: Removed (no useful content)")
            else:
                print(f"[KEEP] {folder_path.name}: Contains useful data")
                
        except Exception as e:
            print(f"[WARNING] Failed to delete {folder_path.name}: {e}")
    
    # Helper methods
    def extract_experiment_id(self, folder_name):
        """Extract a unique experiment ID from folder name"""
        # Remove timestamp prefix to get experiment identifier
        parts = folder_name.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[2:])  # Skip date and time parts
        return folder_name
    
    def extract_timestamp(self, folder_name):
        """Extract timestamp from folder name"""
        parts = folder_name.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return ""
    
    def extract_timestamp_from_filename(self, filename):
        """Extract timestamp from filename"""
        match = re.search(r'(\d{8}_\d{6})', filename)
        return match.group(1) if match else ""
    
    def load_json(self, file_path):
        """Safely load JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Error loading {file_path}: {e}")
            return None
    
    def is_useful_stats(self, stats):
        """Check if stats contain useful information"""
        return (stats.get('best_fitness', 0) > 0 or 
                stats.get('fitness_history') or 
                stats.get('top_sequences'))
    
    def extract_csv_data(self, csv_file):
        """Extract data from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            if not df.empty and 'id' in df.columns:
                return df.to_dict('records')
        except Exception as e:
            print(f"[WARNING] Error reading CSV {csv_file}: {e}")
        return None
    
    def parse_wandb_config(self, config_file):
        """Parse wandb config.yaml file"""
        try:
            import yaml
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[WARNING] Error parsing wandb config: {e}")
            return {}
    
    def match_wandb_to_experiment(self, run_id, wandb_data):
        """Match wandb data to existing experiment data"""
        # Try to find matching experiment by timestamp or problem ID
        config = wandb_data.get('config', {})
        problem_id = config.get('problem_id')
        
        for exp_id, exp_data in self.all_experiment_data.items():
            if problem_id and 'problems' in exp_data and problem_id in exp_data['problems']:
                # Add wandb data to existing experiment
                exp_data['wandb_evolution'] = wandb_data.get('evolution', {})
                exp_data['wandb_config'] = config
                return
        
        # Create new experiment entry for wandb data
        if wandb_data.get('evolution'):
            new_exp_id = f"wandb_{run_id}"
            self.all_experiment_data[new_exp_id] = {
                'source': 'wandb',
                'wandb_evolution': wandb_data.get('evolution', {}),
                'wandb_config': config,
                'run_id': run_id
            }
    
    def plot_experiment_comparison(self, problem_data):
        """Create experiment comparison plots"""
        # Implementation for additional comparison plots
        pass
    
    def plot_performance_summary(self, problem_data):
        """Create performance summary plots"""
        # Implementation for performance summary
        pass


def main():
    """Main execution function"""
    print("=" * 80)
    print("COMPREHENSIVE DATA EXTRACTION AND VISUALIZATION")
    print("=" * 80)
    
    extractor = ComprehensiveDataExtractor()
    extractor.extract_all_data()
    
    print("\n" + "=" * 80)
    print("EXTRACTION AND VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"Check plots directory: {extractor.plots_dir}")


if __name__ == "__main__":
    main()