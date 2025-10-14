# Simple progress monitoring for EA

import time

class SimpleProgressMonitor:
    """Lightweight progress monitoring for EA"""
    
    def __init__(self):
        self.start_time = None
        self.generation_times = []
        self.best_fitness_progress = []
        
    def start_monitoring(self):
        """Start monitoring session"""
        self.start_time = time.time()
        print("EA Progress Monitoring Started")
        print("=" * 50)
    
    def log_generation(self, generation, max_generations, max_fitness, avg_fitness, diversity=None, streak_info=""):
        """Log generation progress"""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        
        progress = (generation + 1) / max_generations * 100
        
        if generation > 0:
            avg_gen_time = elapsed / (generation + 1)
            remaining_gens = max_generations - (generation + 1)
            eta = avg_gen_time * remaining_gens
            eta_str = f"ETA: {eta/60:.1f}min" if eta > 60 else f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: calculating..."
        
        self.best_fitness_progress.append(max_fitness)
        
        diversity_str = f", Diversity: {diversity:.4f}" if diversity is not None else ""
        streak_display = f" ({streak_info})" if streak_info else ""
        
        print(f"Gen {generation+1:3d}/{max_generations} ({progress:5.1f}%) | "
              f"Best: {max_fitness:.4f}, Avg: {avg_fitness:.4f}{diversity_str}{streak_display} | "
              f"Time: {elapsed/60:.1f}min, {eta_str}")
        
        # Show milestone achievements
        if max_fitness > 0.9 and len(self.best_fitness_progress) > 1 and self.best_fitness_progress[-2] <= 0.9:
            print(f"Milestone: First high-quality solution (fitness > 0.9) at generation {generation+1}!")
        
        if max_fitness > 0.95 and len(self.best_fitness_progress) > 1 and self.best_fitness_progress[-2] <= 0.95:
            print(f"Excellent: Near-perfect solution (fitness > 0.95) at generation {generation+1}!")
    
    def finish_monitoring(self, final_results_count=0):
        """Finish monitoring and show summary"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "=" * 50)
        print("EA Execution Completed!")
        print(f"Total Runtime: {total_time/60:.2f} minutes")
        
        if self.best_fitness_progress:
            final_fitness = self.best_fitness_progress[-1]
            initial_fitness = self.best_fitness_progress[0]
            improvement = final_fitness - initial_fitness
            
            print(f"Final Best Fitness: {final_fitness:.4f}")
            print(f"Total Improvement: {improvement:.4f}")
            print(f"Solutions Found: {final_results_count}")
        
        print("Generate plots with: ./generate_plots.sh")
        print("=" * 50)


def add_progress_monitoring(ea_instance):
    """Add lightweight progress monitoring to EA"""
    ea_instance.progress_monitor = SimpleProgressMonitor()
    
    def monitoring_callback(generation, max_fitness, avg_fitness, diversity):
        """Callback function for progress monitoring"""
        streak_info = ea_instance.high_fitness_streak if hasattr(ea_instance, 'high_fitness_streak') else 0
        fitness_stag = ea_instance.fitness_stagnation_counter if hasattr(ea_instance, 'fitness_stagnation_counter') else 0
        diversity_stag = ea_instance.diversity_stagnation_counter if hasattr(ea_instance, 'diversity_stagnation_counter') else 0
        
        info_parts = []
        if streak_info > 0:
            info_parts.append(f"streak: {streak_info}")
        if fitness_stag > 5:
            info_parts.append(f"f_stag: {fitness_stag}")
        if diversity_stag > 5:
            info_parts.append(f"d_stag: {diversity_stag}")
        
        combined_info = ", ".join(info_parts) if info_parts else ""
        
        ea_instance.progress_monitor.log_generation(
            generation, ea_instance.generations, max_fitness, avg_fitness, diversity, combined_info
        )
    
    ea_instance.add_callback(monitoring_callback)
    
    original_run_evolution = ea_instance.run_evolution
    
    def enhanced_run_evolution():
        """Enhanced run_evolution with progress monitoring"""
        print(f"Progress monitoring enabled for {ea_instance.population_size} individuals, {ea_instance.generations} generations")
        
        ea_instance.progress_monitor.start_monitoring()
        original_run_evolution()
        
        results = ea_instance.get_best_individuals() if hasattr(ea_instance, 'get_best_individuals') else []
        final_count = len(results)
        ea_instance.progress_monitor.finish_monitoring(final_count)
    
    ea_instance.run_evolution = enhanced_run_evolution
    
    return ea_instance


def create_simple_demo():
    """Create simple demonstration"""
    import sys
    sys.path.append('src')
    from rna_folding_ea import RNAFoldingEA
    
    ea = RNAFoldingEA(
        population_size=20,
        generations=10,
        sequence_constraint="NNNNNNNNNN",
        structure_constraint="((((...))))"
    )
    
    ea = add_progress_monitoring(ea)
    ea.run_evolution()
    
    print("Demo completed! Use ./generate_plots.sh to create visualizations.")


if __name__ == "__main__":
    create_simple_demo()