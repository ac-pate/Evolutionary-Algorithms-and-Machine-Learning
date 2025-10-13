# Achal Patel - 40227663
# Simple progress monitoring for EA (no real-time plots)

import time

class SimpleProgressMonitor:
    """
    lightweight progress monitoring for ea
    no real-time plots - just enhanced console output and data collection
    """
    
    def __init__(self):
        """initialize progress monitor"""
        self.start_time = None
        self.generation_times = []
        self.best_fitness_progress = []
        
    def start_monitoring(self):
        """Start monitoring session"""
        self.start_time = time.time()
        print("EA Progress Monitoring Started")
        print("=" * 50)
    
    def log_generation(self, generation, max_generations, max_fitness, avg_fitness, diversity=None):
        """
        Log generation progress with enhanced output
        
        Args:
            generation (int): Current generation
            max_generations (int): Total generations
            max_fitness (float): Best fitness this generation
            avg_fitness (float): Average fitness
            diversity (float): Population diversity
        """
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        
        # calculate progress percentage
        progress = (generation + 1) / max_generations * 100
        
        # estimate time remaining
        if generation > 0:
            avg_gen_time = elapsed / (generation + 1)
            remaining_gens = max_generations - (generation + 1)
            eta = avg_gen_time * remaining_gens
            eta_str = f"ETA: {eta/60:.1f}min" if eta > 60 else f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: calculating..."
        
        # store progress data
        self.best_fitness_progress.append(max_fitness)
        
        # enhanced console output
        diversity_str = f", Diversity: {diversity:.4f}" if diversity is not None else ""
        
        print(f"Gen {generation+1:3d}/{max_generations} ({progress:5.1f}%) | "
              f"Best: {max_fitness:.4f}, Avg: {avg_fitness:.4f}{diversity_str} | "
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


# Simple integration function (no real-time visualization)
def add_progress_monitoring(ea_instance):
    """
    Add lightweight progress monitoring to EA (no performance impact)
    
    Args:
        ea_instance: Instance of RNAFoldingEA
    
    Returns:
        Modified EA instance with progress monitoring
    """
    # Create progress monitor
    ea_instance.progress_monitor = SimpleProgressMonitor()
    
    # Store original run_evolution method
    original_run_evolution = ea_instance.run_evolution
    
    def enhanced_run_evolution():
        """Enhanced run_evolution with progress monitoring (no visualization overhead)"""
        print(f"Starting evolution: {ea_instance.population_size} individuals, {ea_instance.generations} generations")
        print(f"Sequence constraint length: {len(ea_instance.sequence_constraint)}")
        print(f"Structure constraint: {ea_instance.structure_constraint[:50]}..." if len(ea_instance.structure_constraint) > 50 else f"Structure constraint: {ea_instance.structure_constraint}")
        
        # Start monitoring
        ea_instance.progress_monitor.start_monitoring()
        
        # Initialize population
        ea_instance.initialize_population()
        
        for generation in range(ea_instance.generations):
            # Evaluate fitness
            fitness_scores = ea_instance.evaluate_fitness(ea_instance.population)
            
            # Track statistics
            max_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            ea_instance.fitness_history.append((generation, max_fitness, avg_fitness))
            
            # Calculate diversity (lightweight calculation)
            diversity = ea_instance.calculate_diversity(ea_instance.population)
            
            # Log progress (no plotting overhead)
            ea_instance.progress_monitor.log_generation(
                generation, ea_instance.generations, max_fitness, avg_fitness, diversity
            )
            
            # Store best individuals
            for i, (seq, fitness) in enumerate(zip(ea_instance.population, fitness_scores)):
                if fitness > 0.9:  # High-quality threshold
                    ea_instance.best_individuals.append((seq, fitness))
            
            # Create next generation (same as original)
            new_population = []
            
            # Elitism: Keep best individuals
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:ea_instance.elitism_count]
            for idx in elite_indices:
                new_population.append(ea_instance.population[idx])
            
            # Generate rest of population through crossover and mutation
            import random
            while len(new_population) < ea_instance.population_size:
                if random.random() < ea_instance.crossover_rate:
                    # Crossover
                    parent1 = ea_instance.tournament_selection(fitness_scores)
                    parent2 = ea_instance.tournament_selection(fitness_scores)
                    child1, child2 = ea_instance.two_point_crossover(parent1, parent2)
                    
                    # Mutation
                    child1 = ea_instance.mutate(child1)
                    child2 = ea_instance.mutate(child2)
                    
                    new_population.extend([child1, child2])
                else:
                    # Just mutation
                    parent = ea_instance.tournament_selection(fitness_scores)
                    child = ea_instance.mutate(parent)
                    new_population.append(child)
            
            # Trim to exact population size
            ea_instance.population = new_population[:ea_instance.population_size]
        
        # Analyze results
        results = ea_instance.analyze_results()
        final_count = len(results) if results else 0
        
        # Finish monitoring
        ea_instance.progress_monitor.finish_monitoring(final_count)
    
    # Replace the method
    ea_instance.run_evolution = enhanced_run_evolution
    
    return ea_instance


def create_simple_demo():
    """Create a simple demonstration without real-time visualization"""
    import sys
    sys.path.append('src')
    from rna_folding_ea import RNAFoldingEA
    
    # Create a small EA for demo
    ea = RNAFoldingEA(
        population_size=20,
        generations=10,
        sequence_constraint="NNNNNNNNNN",
        structure_constraint="((((...))))"
    )
    
    # Add lightweight progress monitoring
    ea = add_progress_monitoring(ea)
    
    # Run the demo
    ea.run_evolution()
    
    print("Demo completed! Use ./generate_plots.sh to create visualizations.")


if __name__ == "__main__":
    # Run a simple demo
    create_simple_demo()