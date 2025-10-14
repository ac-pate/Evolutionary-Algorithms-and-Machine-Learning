# EA Visualization and Analysis Tools

import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict
import re

class EAVisualizer:
    """Comprehensive visualization toolkit for RNA folding EA"""
    
    def __init__(self, stats_files=None, output_dir="plots"):
        self.stats_files = stats_files or []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.data = {}
        if self.stats_files:
            self.load_data()
    
    def load_data(self):
        """Load data from stats files"""
        print("Loading experiment data...")
        
        for stats_file in self.stats_files:
            try:
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                    experiment_name = data.get('name', Path(stats_file).stem)
                    self.data[experiment_name] = data
                    print(f"✓ Loaded {experiment_name}")
            except Exception as e:
                print(f"✗ Failed to load {stats_file}: {e}")
    
    def plot_fitness_evolution(self, save=True, show=True):
        """Plot fitness evolution over generations for all experiments"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for exp_name, data in self.data.items():
            if 'fitness_history' in data:
                fitness_history = data['fitness_history']
                generations = [entry[0] for entry in fitness_history]
                max_fitness = [entry[1] for entry in fitness_history]
                avg_fitness = [entry[2] for entry in fitness_history]
                
                ax1.plot(generations, max_fitness, label=f'{exp_name} (Max)', linewidth=2, marker='o', markersize=3)
                ax2.plot(generations, avg_fitness, label=f'{exp_name} (Avg)', linewidth=2, marker='s', markersize=3)
        
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Maximum Fitness')
        ax1.set_title('Evolution of Best Solutions', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1.05)
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Fitness')
        ax2.set_title('Population Average Fitness', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'fitness_evolution.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}/fitness_evolution.png")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_parameter_comparison(self, save=True, show=True):
        """
        Compare final results across different parameter settings
        """
        if len(self.data) < 2:
            print("Need at least 2 experiments for parameter comparison")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        experiments = []
        max_fitness_values = []
        avg_fitness_values = []
        population_sizes = []
        mutation_rates = []
        crossover_rates = []
        final_generations = []
        
        # Extract data for comparison
        for exp_name, data in self.data.items():
            experiments.append(exp_name)
            
            # Get final fitness values
            if 'fitness_history' in data and data['fitness_history']:
                final_entry = data['fitness_history'][-1]
                max_fitness_values.append(final_entry[1])
                avg_fitness_values.append(final_entry[2])
                final_generations.append(final_entry[0])
            else:
                max_fitness_values.append(0)
                avg_fitness_values.append(0)
                final_generations.append(0)
            
            # Get parameters
            params = data.get('parameters', {})
            population_sizes.append(params.get('population_size', 0))
            mutation_rates.append(params.get('mutation_rate', 0))
            crossover_rates.append(params.get('crossover_rate', 0))
        
        # Plot 1: Final Fitness Comparison
        x_pos = np.arange(len(experiments))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, max_fitness_values, width, label='Max Fitness', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, avg_fitness_values, width, label='Avg Fitness', alpha=0.8)
        
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Fitness Score')
        ax1.set_title('Final Fitness Comparison', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(experiments, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Population Size vs Final Max Fitness
        ax2.scatter(population_sizes, max_fitness_values, s=100, alpha=0.7, c=range(len(experiments)), cmap='viridis')
        for i, exp in enumerate(experiments):
            ax2.annotate(exp, (population_sizes[i], max_fitness_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('Population Size')
        ax2.set_ylabel('Final Max Fitness')
        ax2.set_title('Population Size vs Performance', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mutation Rate vs Final Max Fitness
        ax3.scatter(mutation_rates, max_fitness_values, s=100, alpha=0.7, c=range(len(experiments)), cmap='plasma')
        for i, exp in enumerate(experiments):
            ax3.annotate(exp, (mutation_rates[i], max_fitness_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Mutation Rate')
        ax3.set_ylabel('Final Max Fitness')
        ax3.set_title('Mutation Rate vs Performance', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Parameter Summary Table
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for i, exp in enumerate(experiments):
            table_data.append([
                exp[:15] + '...' if len(exp) > 15 else exp,
                f"{population_sizes[i]}",
                f"{mutation_rates[i]:.3f}",
                f"{crossover_rates[i]:.2f}",
                f"{max_fitness_values[i]:.3f}"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Experiment', 'Pop Size', 'Mut Rate', 'Cross Rate', 'Best Fitness'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.set_title('Parameter Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'parameter_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}/parameter_comparison.png")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_convergence_analysis(self, save=True, show=True):
        """
        Analyze convergence patterns and diversity over time
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        for exp_name, data in self.data.items():
            if 'fitness_history' not in data:
                continue
            
            fitness_history = data['fitness_history']
            generations = [entry[0] for entry in fitness_history]
            max_fitness = [entry[1] for entry in fitness_history]
            avg_fitness = [entry[2] for entry in fitness_history]
            
            # Calculate fitness improvement rate
            improvement_rate = []
            for i in range(1, len(max_fitness)):
                rate = max_fitness[i] - max_fitness[i-1]
                improvement_rate.append(rate)
            
            # Calculate fitness diversity (max - avg)
            fitness_diversity = [max_fit - avg_fit for max_fit, avg_fit in zip(max_fitness, avg_fitness)]
            
            # Plot 1: Fitness Improvement Rate
            if improvement_rate:
                ax1.plot(generations[1:], improvement_rate, label=exp_name, linewidth=2, alpha=0.7)
            
            # Plot 2: Fitness Diversity (Population Spread)
            ax2.plot(generations, fitness_diversity, label=exp_name, linewidth=2, alpha=0.7)
            
            # Plot 3: Convergence Speed (Generations to reach 90% of final fitness)
            final_fitness = max_fitness[-1] if max_fitness else 0
            convergence_threshold = 0.9 * final_fitness
            convergence_gen = None
            for i, fitness in enumerate(max_fitness):
                if fitness >= convergence_threshold:
                    convergence_gen = generations[i]
                    break
            
            if convergence_gen is not None:
                ax3.bar(exp_name, convergence_gen, alpha=0.7)
            
            # Plot 4: Learning Curve Smoothness
            if len(max_fitness) > 5:
                # Calculate variance in improvement
                smoothness = np.var(improvement_rate) if improvement_rate else 0
                ax4.bar(exp_name, smoothness, alpha=0.7)
        
        # Customize plots
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness Improvement Rate')
        ax1.set_title('Fitness Improvement Rate Over Time', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Max - Average Fitness')
        ax2.set_title('Population Fitness Diversity', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Experiment')
        ax3.set_ylabel('Generations to 90% Final Fitness')
        ax3.set_title('Convergence Speed', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        ax4.set_xlabel('Experiment')
        ax4.set_ylabel('Variance in Improvement Rate')
        ax4.set_title('Learning Curve Smoothness', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}/convergence_analysis.png")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_sequence_diversity_analysis(self, results_files=None, save=True, show=True):
        """
        Analyze diversity of final sequences
        
        Args:
            results_files (list): List of results text files containing sequences
        """
        if not results_files:
            print("No results files provided for diversity analysis")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        all_sequences = {}
        
        # Load sequences from results files
        for results_file in results_files:
            try:
                exp_name = Path(results_file).stem.replace('results_', '')
                with open(results_file, 'r') as f:
                    sequences = [line.strip() for line in f if line.strip()]
                all_sequences[exp_name] = sequences
                print(f"✓ Loaded {len(sequences)} sequences from {exp_name}")
            except Exception as e:
                print(f"✗ Failed to load {results_file}: {e}")
        
        if not all_sequences:
            print("No sequences loaded for diversity analysis")
            return
        
        # Calculate diversity metrics
        diversity_scores = {}
        sequence_lengths = {}
        gc_content = {}
        unique_counts = {}
        
        for exp_name, sequences in all_sequences.items():
            if not sequences:
                continue
            
            # Calculate pairwise Hamming distances
            distances = []
            for i in range(len(sequences)):
                for j in range(i + 1, len(sequences)):
                    if len(sequences[i]) == len(sequences[j]):
                        hamming_dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                        normalized_dist = hamming_dist / len(sequences[i])
                        distances.append(normalized_dist)
            
            diversity_scores[exp_name] = np.mean(distances) if distances else 0
            
            # Sequence length distribution
            lengths = [len(seq) for seq in sequences]
            sequence_lengths[exp_name] = lengths
            
            # GC content analysis
            gc_contents = []
            for seq in sequences:
                gc_count = seq.count('G') + seq.count('C')
                gc_contents.append(gc_count / len(seq) if len(seq) > 0 else 0)
            gc_content[exp_name] = gc_contents
            
            # Unique sequence count
            unique_counts[exp_name] = len(set(sequences))
        
        # Plot 1: Diversity Scores Comparison
        exp_names = list(diversity_scores.keys())
        diversity_values = list(diversity_scores.values())
        
        bars = ax1.bar(exp_names, diversity_values, alpha=0.8, color=plt.cm.viridis(np.linspace(0, 1, len(exp_names))))
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Average Normalized Hamming Distance')
        ax1.set_title('Sequence Diversity Comparison', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, diversity_values):
            ax1.annotate(f'{value:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        # Plot 2: Sequence Length Distribution
        for exp_name, lengths in sequence_lengths.items():
            ax2.hist(lengths, alpha=0.7, label=exp_name, bins=20)
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Sequence Length Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: GC Content Distribution
        for exp_name, gc_values in gc_content.items():
            ax3.hist(gc_values, alpha=0.7, label=exp_name, bins=20)
        ax3.set_xlabel('GC Content')
        ax3.set_ylabel('Frequency')
        ax3.set_title('GC Content Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Unique Sequences Count
        unique_names = list(unique_counts.keys())
        unique_values = list(unique_counts.values())
        total_values = [len(all_sequences[name]) for name in unique_names]
        
        x_pos = np.arange(len(unique_names))
        width = 0.35
        
        ax4.bar(x_pos - width/2, total_values, width, label='Total Sequences', alpha=0.8)
        ax4.bar(x_pos + width/2, unique_values, width, label='Unique Sequences', alpha=0.8)
        
        ax4.set_xlabel('Experiment')
        ax4.set_ylabel('Number of Sequences')
        ax4.set_title('Sequence Uniqueness', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(unique_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'sequence_diversity.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}/sequence_diversity.png")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_performance_summary(self, save=True, show=True):
        """
        Create a comprehensive performance summary dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Extract summary statistics
        summary_data = []
        for exp_name, data in self.data.items():
            params = data.get('parameters', {})
            fitness_history = data.get('fitness_history', [])
            
            if fitness_history:
                final_fitness = fitness_history[-1][1]
                avg_final_fitness = fitness_history[-1][2]
                generations = len(fitness_history)
                
                # Calculate improvement metrics
                initial_fitness = fitness_history[0][1] if fitness_history else 0
                total_improvement = final_fitness - initial_fitness
                avg_improvement_per_gen = total_improvement / generations if generations > 0 else 0
                
                summary_data.append({
                    'experiment': exp_name,
                    'final_fitness': final_fitness,
                    'avg_final_fitness': avg_final_fitness,
                    'total_improvement': total_improvement,
                    'avg_improvement_per_gen': avg_improvement_per_gen,
                    'generations': generations,
                    'population_size': params.get('population_size', 0),
                    'mutation_rate': params.get('mutation_rate', 0),
                    'crossover_rate': params.get('crossover_rate', 0)
                })
        
        if not summary_data:
            print("No data available for performance summary")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(summary_data)
        
        # Plot 1: Final Fitness Ranking
        ax1 = fig.add_subplot(gs[0, :2])
        sorted_df = df.sort_values('final_fitness', ascending=True)
        bars = ax1.barh(sorted_df['experiment'], sorted_df['final_fitness'], alpha=0.8)
        ax1.set_xlabel('Final Best Fitness')
        ax1.set_title('Experiment Performance Ranking', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Color bars by performance
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn(sorted_df.iloc[i]['final_fitness']))
        
        # Plot 2: Efficiency (Improvement per Generation)
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.scatter(df['generations'], df['avg_improvement_per_gen'], 
                   s=df['population_size']/2, alpha=0.7, c=df['final_fitness'], cmap='viridis')
        for _, row in df.iterrows():
            ax2.annotate(row['experiment'][:10], 
                        (row['generations'], row['avg_improvement_per_gen']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('Generations')
        ax2.set_ylabel('Average Improvement per Generation')
        ax2.set_title('Learning Efficiency', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Parameter vs Performance Heatmap
        ax3 = fig.add_subplot(gs[1, :2])
        
        # Create parameter performance matrix
        param_perf = df.pivot_table(index='mutation_rate', columns='crossover_rate', 
                                   values='final_fitness', aggfunc='mean')
        
        if not param_perf.empty:
            sns.heatmap(param_perf, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3)
            ax3.set_title('Parameter Performance Heatmap', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for heatmap', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Parameter Performance Heatmap', fontweight='bold')
        
        # Plot 4: Resource Efficiency
        ax4 = fig.add_subplot(gs[1, 2:])
        efficiency = df['final_fitness'] / (df['population_size'] * df['generations'] / 1000)
        ax4.bar(df['experiment'], efficiency, alpha=0.8)
        ax4.set_xlabel('Experiment')
        ax4.set_ylabel('Fitness per 1K Evaluations')
        ax4.set_title('Computational Efficiency', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Statistical Summary Table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create summary statistics
        stats_summary = [
            ['Metric', 'Best', 'Worst', 'Average', 'Std Dev'],
            ['Final Fitness', f"{df['final_fitness'].max():.3f}", 
             f"{df['final_fitness'].min():.3f}", f"{df['final_fitness'].mean():.3f}", 
             f"{df['final_fitness'].std():.3f}"],
            ['Total Improvement', f"{df['total_improvement'].max():.3f}", 
             f"{df['total_improvement'].min():.3f}", f"{df['total_improvement'].mean():.3f}", 
             f"{df['total_improvement'].std():.3f}"],
            ['Generations', f"{df['generations'].max():.0f}", 
             f"{df['generations'].min():.0f}", f"{df['generations'].mean():.1f}", 
             f"{df['generations'].std():.1f}"],
            ['Population Size', f"{df['population_size'].max():.0f}", 
             f"{df['population_size'].min():.0f}", f"{df['population_size'].mean():.1f}", 
             f"{df['population_size'].std():.1f}"]
        ]
        
        table = ax5.table(cellText=stats_summary[1:], colLabels=stats_summary[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax5.set_title('Performance Statistics Summary', fontweight='bold', pad=20)
        
        if save:
            plt.savefig(self.output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}/performance_summary.png")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_report_plots(self, results_files=None):
        """
        Generate all plots for the assignment report
        
        Args:
            results_files (list): List of results text files for diversity analysis
        """
        print("Generating comprehensive EA visualization report...")
        print("=" * 50)
        
        # Create all plots
        self.plot_fitness_evolution(show=False)
        self.plot_parameter_comparison(show=False)
        self.plot_convergence_analysis(show=False)
        self.plot_performance_summary(show=False)
        
        if results_files:
            self.plot_sequence_diversity_analysis(results_files, show=False)
        
        print(f"\n✓ All plots saved to: {self.output_dir}")
        print("Ready for report integration!")


    def plot_six_constraint_grid(self, problem_results, save=True, show=True):
        """Create 2x3 grid showing results for all 6 constraint problems"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        problem_ids = ['1.1', '1.2', '2.1', '2.2', '3.1', '3.2']
        
        for i, problem_id in enumerate(problem_ids):
            ax = axes[i]
            
            if problem_id in problem_results:
                data = problem_results[problem_id]
                fitness_history = data.get('fitness_history', [])
                
                if fitness_history:
                    generations = [entry[0] for entry in fitness_history]
                    max_fitness = [entry[1] for entry in fitness_history]
                    avg_fitness = [entry[2] for entry in fitness_history]
                    
                    ax.plot(generations, max_fitness, 'b-', linewidth=2, label='Max Fitness')
                    ax.plot(generations, avg_fitness, 'r--', linewidth=1.5, label='Avg Fitness')
                    
                    ax.set_title(f'Problem {problem_id}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Fitness')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1.05)
                    ax.legend()
                    
                    # Add final fitness annotation
                    final_fitness = max_fitness[-1] if max_fitness else 0
                    ax.text(0.02, 0.95, f'Final: {final_fitness:.3f}', 
                           transform=ax.transAxes, fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'Problem {problem_id}', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'Problem {problem_id}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'six_constraint_grid.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}/six_constraint_grid.png")
        
        if show:
            plt.show()
        else:
            plt.close()

def load_and_visualize(results_dir="results", stats_pattern="**/*stats.json"):
    """Convenience function to load experiment data and create visualizations"""
    from glob import glob
    import os
    
    # Find all stats files
    if os.path.isdir(results_dir):
        full_pattern = os.path.join(results_dir, stats_pattern)
        stats_files = glob(full_pattern, recursive=True)
    else:
        stats_files = glob(stats_pattern, recursive=True)
    
    print(f"Found {len(stats_files)} stats files")
    
    if not stats_files:
        print("No stats files found! Make sure to run experiments first.")
        return None
    
    # Create visualizer and generate plots
    visualizer = EAVisualizer(stats_files)
    visualizer.plot_fitness_evolution()
    visualizer.plot_parameter_comparison()
    visualizer.plot_convergence_analysis()
    visualizer.plot_sequence_diversity()
    visualizer.plot_performance_summary()
    
    return visualizer


if __name__ == "__main__":
    # Example usage
    visualizer = load_and_visualize()
    
    if visualizer:
        print("\nVisualization complete!")
        print("Check the 'plots' directory for all generated images.")
        print("\nPlots generated:")
        print("- fitness_evolution.png: Fitness progression over generations")
        print("- parameter_comparison.png: Performance vs parameter settings")
        print("- convergence_analysis.png: Convergence patterns and speed")
        print("- sequence_diversity.png: Diversity analysis of final sequences")
        print("- performance_summary.png: Comprehensive performance dashboard")