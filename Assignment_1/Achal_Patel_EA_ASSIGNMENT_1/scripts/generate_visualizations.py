#!/usr/bin/env python3
"""
Generate comprehensive visualizations from EA experiment results
Usage: python3 generate_visualizations.py
"""

import sys
import os
from glob import glob
from pathlib import Path

# Add scripts to path for imports
sys.path.append('scripts')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/scripts')

try:
    from ea_visualization_and_plots import EAVisualizer
except ImportError as e:
    print(f"❌ Could not import visualization module: {e}")
    print(f"Make sure ea_visualization_and_plots.py is in the scripts/ folder")
    sys.exit(1)

def main():
    print("RNA Folding EA - Comprehensive Visualization Generator")
    print("="*60)
    
    # Find all stats files
    stats_files = glob("results/**/*stats.json", recursive=True)
    results_files = glob("results/**/*results.txt", recursive=True)
    
    print(f"Found {len(stats_files)} stats files")
    print(f"Found {len(results_files)} results files")
    
    if not stats_files:
        print("❌ No stats files found!")
        print("Make sure you have run experiments that generate stats.json files")
        return
    
    # Create output directory
    output_dir = "plots"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create visualizer
    print(f"\n📊 Creating visualizations...")
    visualizer = EAVisualizer(stats_files, output_dir)
    
    if not visualizer.data:
        print("❌ No data loaded from stats files!")
        return
    
    print(f"✅ Loaded data from {len(visualizer.data)} experiments")
    
    # Generate all plots
    print(f"\n🎨 Generating plots...")
    
    try:
        # 1. Fitness Evolution Plots
        print("   📈 Generating fitness evolution plots...")
        visualizer.plot_fitness_evolution(show=False)
        
        # 2. Parameter Comparison
        print("   🔬 Generating parameter comparison plots...")
        visualizer.plot_parameter_comparison(show=False)
        
        # 3. Convergence Analysis
        print("   📊 Generating convergence analysis...")
        visualizer.plot_convergence_analysis(show=False)
        
        # 4. Performance Summary
        print("   📋 Generating performance summary dashboard...")
        visualizer.plot_performance_summary(show=False)
        
        # 5. Six Constraint Grid (NEW!)
        print("   🧬 Generating 6-constraint comparison grid...")
        visualizer.plot_six_constraint_grid(show=False)
        
        # 6. Sequence Diversity Analysis (if results files available)
        if results_files:
            print("   🧬 Generating sequence diversity analysis...")
            # Sample some results files to avoid overloading
            sample_results = results_files[:50]  # Use more files for better analysis
            visualizer.plot_sequence_diversity_analysis(sample_results, show=False)
        
        print(f"\n✅ All visualizations completed!")
        print(f"📁 Plots saved to: {output_dir}/")
        print(f"\nGenerated files:")
        
        # List generated files
        plot_files = glob(f"{output_dir}/*.png")
        for plot_file in sorted(plot_files):
            filename = Path(plot_file).name
            print(f"   📊 {filename}")
        
        print(f"\n🎯 Usage tips:")
        print(f"   • Open plots in an image viewer or web browser")
        print(f"   • Use these plots in your assignment report")
        print(f"   • fitness_evolution.png shows learning curves")
        print(f"   • parameter_comparison.png shows which settings work best")
        print(f"   • performance_summary.png gives overall comparison")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        print(f"🔧 Try running: pip install matplotlib seaborn pandas")

def analyze_specific_problems():
    """Analyze results for specific problem types"""
    print(f"\n🔍 Problem-specific analysis...")
    
    # Group stats files by problem
    problem_data = {}
    stats_files = glob("results/**/*stats.json", recursive=True)
    
    for stats_file in stats_files:
        # Extract problem ID from path
        if "problem_" in stats_file:
            problem_id = stats_file.split("problem_")[1].split("/")[0]
            if problem_id not in problem_data:
                problem_data[problem_id] = []
            problem_data[problem_id].append(stats_file)
    
    print(f"Found data for problems: {list(problem_data.keys())}")
    
    # Create problem-specific visualization
    if problem_data:
        visualizer = EAVisualizer([], "plots")
        
        # Load data for each problem
        for problem_id, files in problem_data.items():
            print(f"   📊 Problem {problem_id}: {len(files)} experiments")
        
        # You can extend this to create problem-specific plots
        print(f"✅ Problem analysis complete")

if __name__ == "__main__":
    main()
    analyze_specific_problems()