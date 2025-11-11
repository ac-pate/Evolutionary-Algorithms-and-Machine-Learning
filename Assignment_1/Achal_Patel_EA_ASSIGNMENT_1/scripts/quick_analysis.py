#!/usr/bin/env python3
"""
Quick Problem-Specific Visualization
Usage: python3 quick_analysis.py [problem_id]
Example: python3 quick_analysis.py 1.1
"""

import sys
import os
from glob import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_problem(problem_id=None):
    """Analyze a specific problem or all problems"""
    
    if problem_id:
        print(f"📊 Analyzing Problem {problem_id}")
        pattern = f"results/**/problem_{problem_id}/*stats.json"
    else:
        print(f"📊 Analyzing All Problems")
        pattern = f"results/**/*stats.json"
    
    stats_files = glob(pattern, recursive=True)
    print(f"Found {len(stats_files)} experiment files")
    
    if not stats_files:
        print(f"❌ No data found for problem {problem_id}")
        return
    
    # Create plots directory
    Path("plots").mkdir(exist_ok=True)
    
    # Load and plot data
    plt.figure(figsize=(15, 10))
    
    best_experiments = []
    
    for i, stats_file in enumerate(stats_files):
        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
            
            if 'fitness_history' not in data:
                continue
                
            fitness_history = data['fitness_history']
            generations = [entry[0] for entry in fitness_history]
            max_fitness = [entry[1] for entry in fitness_history]
            
            # Extract experiment info from path
            exp_name = Path(stats_file).parent.parent.name
            
            # Plot with transparency
            alpha = 0.3 if len(stats_files) > 10 else 0.7
            plt.plot(generations, max_fitness, alpha=alpha, linewidth=1)
            
            # Track best experiments
            final_fitness = max_fitness[-1] if max_fitness else 0
            best_experiments.append((exp_name, final_fitness, max_fitness, generations))
            
        except Exception as e:
            print(f"⚠️  Skipped {stats_file}: {e}")
    
    if not best_experiments:
        print("❌ No valid fitness data found")
        return
    
    # Highlight top 3 experiments
    best_experiments.sort(key=lambda x: x[1], reverse=True)
    
    for i, (exp_name, final_fitness, max_fitness, generations) in enumerate(best_experiments[:3]):
        plt.plot(generations, max_fitness, linewidth=3, 
                label=f"#{i+1}: {exp_name[:20]}... (fitness: {final_fitness:.3f})")
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Problem {problem_id} - Fitness Evolution ({len(stats_files)} experiments)' if problem_id else f'All Problems - Fitness Evolution ({len(stats_files)} experiments)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Save plot
    filename = f"plots/problem_{problem_id}_analysis.png" if problem_id else "plots/all_problems_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    
    # Print summary
    print(f"\n📈 Results Summary:")
    print(f"   📊 Total experiments: {len(best_experiments)}")
    print(f"   🏆 Best fitness: {best_experiments[0][1]:.4f}")
    print(f"   📊 Average best fitness: {np.mean([x[1] for x in best_experiments]):.4f}")
    print(f"   📊 Experiments with fitness > 0.9: {len([x for x in best_experiments if x[1] > 0.9])}")
    
    plt.show()

def main():
    if len(sys.argv) > 1:
        problem_id = sys.argv[1]
        analyze_problem(problem_id)
    else:
        print("Available problems based on your data:")
        
        # Find all problem IDs
        stats_files = glob("results/**/*stats.json", recursive=True)
        problem_ids = set()
        
        for stats_file in stats_files:
            if "problem_" in stats_file:
                problem_id = stats_file.split("problem_")[1].split("/")[0]
                problem_ids.add(problem_id)
        
        for pid in sorted(problem_ids):
            count = len(glob(f"results/**/problem_{pid}/*stats.json", recursive=True))
            print(f"   🧬 Problem {pid}: {count} experiments")
        
        print(f"\nUsage:")
        print(f"   python3 quick_analysis.py 1.1    # Analyze specific problem")
        print(f"   python3 quick_analysis.py all    # Analyze all problems")
        
        choice = input(f"\nEnter problem ID (or 'all'): ").strip()
        if choice.lower() == 'all':
            analyze_problem()
        elif choice in problem_ids:
            analyze_problem(choice)
        else:
            print(f"❌ Unknown problem ID: {choice}")

if __name__ == "__main__":
    main()