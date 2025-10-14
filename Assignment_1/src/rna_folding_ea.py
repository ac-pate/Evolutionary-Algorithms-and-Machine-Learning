# Achal Patel - 40227663

import random
import numpy as np
import subprocess
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json

class RNAFoldingEA:
    """
    Evolutionary Algorithm for Inverse RNA Folding Problem
    
    this ea finds rna sequences that:
    1. Satisfy IUPAC sequence constraints (HARD constraint)
    2. fold into target secondary structure (fitness optimization)
    3. maximize diversity of solutions
    """
    
    def __init__(self, population_size, generations, sequence_constraint, structure_constraint, max_workers=8, elite_percentage=0.01):
        """
        Initialize the Evolutionary Algorithm
        
        Args:
            population_size (int): Number of individuals in population
            generations (int): Number of generations to evolve
            sequence_constraint (str): IUPAC notation string
            structure_constraint (str): Dot-bracket notation string
            max_workers (int): Number of parallel workers for fitness evaluation
            elite_percentage (float): Percentage of population to keep as elites (default: 0.01 = 1%)
        """
        self.population_size = population_size
        self.generations = generations
        self.sequence_constraint = sequence_constraint
        self.structure_constraint = structure_constraint
        self.sequence_length = len(sequence_constraint)
        self.max_workers = max_workers
        
        # evolutionary params (will be optimized based on hardware)
        self.crossover_rate = 0.8
        self.mutation_rate = 1.0 / self.sequence_length  # adaptive mutation rate
        self.tournament_size = 3
        self.elite_percentage = elite_percentage  # Use provided value
        self.elitism_count = max(1, int(population_size * self.elite_percentage))
        
        # Early termination settings
        self.early_termination_fitness = 0.95
        self.high_fitness_streak_threshold = 5
        self.high_fitness_streak = 0
        
        # Fitness cache for optimization
        self.fitness_cache = {}
        
        # Stagnation settings - separated fitness and diversity
        self.fitness_stagnation_threshold = 10
        self.diversity_stagnation_threshold = 15
        self.fitness_stagnation_counter = 0
        self.diversity_stagnation_counter = 0
        self.last_best_fitness = 0.0
        self.last_diversity = 1.0
        
        self.diversity_threshold = 0.2
        self.restart_rate = 0.3
        self.base_mutation_rate = self.mutation_rate
        self.mutation_boost_factor = 3.0
        
        # iupac code mapping for constraint validation
        self.iupac_codes = {
            'N': ['A', 'U', 'C', 'G'],  # any nucleotide
            'A': ['A'], 'U': ['U'], 'C': ['C'], 'G': ['G'],  # specific bases
            'W': ['A', 'U'],  # weak bonds
            'S': ['C', 'G'],  # Strong bonds  
            'R': ['A', 'G'],  # purines
            'Y': ['C', 'U'],  # pyrimidines
            'M': ['A', 'C'],  # amino group
            'K': ['G', 'U'],  # Keto group
            'H': ['A', 'C', 'U'],  # not g
            'B': ['C', 'G', 'U'],  # Not A
            'V': ['A', 'C', 'G'],  # Not U
            'D': ['A', 'G', 'U']   # Not C
        }
        
        # results storage
        self.best_individuals = []
        self.population = []
        self.fitness_history = []
        self.diversity_history = []
        self.callbacks = []  # For wandb or other external logging
        self.early_terminated = False
        self.termination_reason = None
        
    def add_callback(self, callback_func):
        """
        Add a callback function that will be called each generation
        
        Args:
            callback_func: Function with signature (generation, best_fitness, avg_fitness, diversity)
        """
        self.callbacks.append(callback_func)
        
    def is_valid_sequence(self, sequence):
        """Check if sequence satisfies IUPAC constraints"""
        if len(sequence) != self.sequence_length:
            return False
            
        for i, (seq_base, constraint) in enumerate(zip(sequence, self.sequence_constraint)):
            if constraint in self.iupac_codes:
                if seq_base not in self.iupac_codes[constraint]:
                    return False
            else:
                if seq_base not in ['A', 'U', 'C', 'G']:
                    return False
        return True
    
    def generate_random_sequence(self):
        """Generate random RNA sequence satisfying IUPAC constraints"""
        sequence = []
        for constraint in self.sequence_constraint:
            if constraint in self.iupac_codes:
                allowed_bases = self.iupac_codes[constraint]
            else:
                allowed_bases = ['A', 'U', 'C', 'G']
            sequence.append(random.choice(allowed_bases))
        return ''.join(sequence)
    
    def initialize_population(self):
        """Initialize population with valid random sequences"""
        print("Initializing population...")
        population = []
        seen_sequences = set()
        
        attempts = 0
        max_attempts = self.population_size * 10
        
        while len(population) < self.population_size and attempts < max_attempts:
            sequence = self.generate_random_sequence()
            if sequence not in seen_sequences and self.is_valid_sequence(sequence):
                population.append(sequence)
                seen_sequences.add(sequence)
            attempts += 1
            
        while len(population) < self.population_size:
            sequence = self.generate_random_sequence()
            population.append(sequence)
            
        self.population = population
        print(f"Initialized population with {len(population)} individuals")
    
    def fold_rna_ipknot(self, sequence):
        """Use IPknot via Docker to predict RNA secondary structure"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
                f.write(f">sequence\n{sequence}\n")
                temp_file = f.name
            
            cmd = ["sudo", "docker", "exec", "ipknot_runner", "ipknot", f"/work/{os.path.basename(temp_file)}"]
            copy_cmd = ["sudo", "docker", "cp", temp_file, f"ipknot_runner:/work/{os.path.basename(temp_file)}"]
            subprocess.run(copy_cmd, check=True, capture_output=True)
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                if line.startswith('>'):
                    if i + 2 < len(lines):
                        structure = lines[i + 2].strip()
                        os.unlink(temp_file)
                        return structure
            
            os.unlink(temp_file)
            return '.' * len(sequence)
            
        except Exception as e:
            print(f"Error folding sequence {sequence}: {e}")
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)
            return '.' * len(sequence)
    
    def calculate_structure_fitness(self, predicted_structure, target_structure):
        """Calculate fitness based on structure similarity"""
        if len(predicted_structure) != len(target_structure):
            return 0.0
        
        matches = sum(1 for p, t in zip(predicted_structure, target_structure) if p == t)
        return matches / len(target_structure)
    
    def evaluate_fitness(self, sequences):
        """Evaluate fitness for batch of sequences with caching"""
        def evaluate_single(sequence):
            if not self.is_valid_sequence(sequence):
                return 0.0
            
            # Check cache first
            if sequence in self.fitness_cache:
                return self.fitness_cache[sequence]
            
            predicted_structure = self.fold_rna_ipknot(sequence)
            fitness = self.calculate_structure_fitness(predicted_structure, self.structure_constraint)
            
            # Cache result
            self.fitness_cache[sequence] = fitness
            return fitness
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fitness_scores = list(executor.map(evaluate_single, sequences))
        
        return fitness_scores
    
    def tournament_selection(self, fitness_scores):
        """Tournament selection for parent selection"""
        try:
            if len(self.population) == 0 or len(fitness_scores) == 0:
                raise ValueError("Empty population or fitness scores")
            
            if len(self.population) != len(fitness_scores):
                raise ValueError(f"Population size ({len(self.population)}) != fitness scores size ({len(fitness_scores)})")
            
            actual_tournament_size = min(self.tournament_size, len(self.population))
            tournament_indices = random.sample(range(len(self.population)), actual_tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            
            if 0 <= winner_index < len(self.population):
                return self.population[winner_index]
            else:
                raise IndexError(f"Winner index {winner_index} out of bounds for population size {len(self.population)}")
                
        except Exception as e:
            print(f"Warning: Tournament selection failed: {e}, using fallback")
            if len(self.population) > 0:
                return random.choice(self.population)
            else:
                return self.generate_random_sequence()
    
    def multi_point_crossover(self, parent1, parent2):
        """Multi-point crossover optimized for long sequences"""
        if len(parent1) != len(parent2):
            return parent1, parent2
        
        seq_len = len(parent1)
        if seq_len < 50:
            return self.two_point_crossover(parent1, parent2)
        
        # For long sequences, use 3-4 crossover points
        num_points = 4 if seq_len > 150 else 3
        points = sorted(random.sample(range(1, seq_len), num_points))
        
        offspring1, offspring2 = list(parent1), list(parent2)
        
        # Alternate segments between parents
        for i in range(0, len(points), 2):
            start = points[i]
            end = points[i + 1] if i + 1 < len(points) else seq_len
            offspring1[start:end], offspring2[start:end] = offspring2[start:end], offspring1[start:end]
        
        return self.repair_sequence(''.join(offspring1)), self.repair_sequence(''.join(offspring2))

    def two_point_crossover(self, parent1, parent2):
        """Two-point crossover for shorter sequences"""
        if len(parent1) != len(parent2):
            return parent1, parent2
        
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)
        
        offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return self.repair_sequence(offspring1), self.repair_sequence(offspring2)
    
    def mutate(self, sequence):
        """Point mutation with constraint validation and adaptive cooling"""
        sequence_list = list(sequence)
        
        # Adaptive mutation based on generation (cooling)
        generation_ratio = getattr(self, 'current_generation', 0) / max(self.generations, 1)
        cooling_factor = 1.0 - (generation_ratio * 0.5)  # Reduce mutation over time
        current_rate = self.mutation_rate * cooling_factor
        
        for i in range(len(sequence_list)):
            if random.random() < current_rate:
                constraint = self.sequence_constraint[i]
                if constraint in self.iupac_codes:
                    allowed_bases = self.iupac_codes[constraint]
                else:
                    allowed_bases = ['A', 'U', 'C', 'G']
                
                current_base = sequence_list[i]
                possible_mutations = [base for base in allowed_bases if base != current_base]
                if possible_mutations:
                    sequence_list[i] = random.choice(possible_mutations)
        
        return ''.join(sequence_list)
    
    def repair_sequence(self, sequence):
        """Repair sequence to satisfy IUPAC constraints"""
        sequence_list = list(sequence)
        
        for i, (base, constraint) in enumerate(zip(sequence_list, self.sequence_constraint)):
            if constraint in self.iupac_codes:
                allowed_bases = self.iupac_codes[constraint]
                if base not in allowed_bases:
                    sequence_list[i] = random.choice(allowed_bases)
        
        return ''.join(sequence_list)
    
    def calculate_diversity(self, sequences, sample_size=None):
        """Calculate diversity with optional sampling for large populations"""
        if len(sequences) < 2:
            return 0.0
        
        # Use sampling for large populations to improve performance
        if sample_size and len(sequences) > sample_size:
            sampled = random.sample(sequences, sample_size)
        else:
            sampled = sequences
        
        def calculate_distances_chunk(chunk_indices):
            total_distance = 0.0
            comparisons = 0
            
            for i in chunk_indices:
                if i >= len(sampled):
                    continue
                for j in range(i + 1, len(sampled)):
                    if j >= len(sampled):
                        continue
                    hamming_dist = sum(1 for a, b in zip(sampled[i], sampled[j]) if a != b)
                    normalized_dist = hamming_dist / len(sampled[i])
                    total_distance += normalized_dist
                    comparisons += 1
            
            return total_distance, comparisons
        
        num_sequences = len(sampled)
        chunk_size = max(1, num_sequences // self.max_workers)
        chunks = []
        
        for i in range(0, num_sequences, chunk_size):
            chunk_end = min(i + chunk_size, num_sequences)
            chunk_indices = list(range(i, chunk_end))
            if chunk_indices:
                chunks.append(chunk_indices)
        
        if not chunks:
            return 0.0
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
            results = list(executor.map(calculate_distances_chunk, chunks))
        
        total_distance = sum(result[0] for result in results)
        total_comparisons = sum(result[1] for result in results)
        
        return total_distance / total_comparisons if total_comparisons > 0 else 0.0
    
    def adaptive_mutation_rate(self, diversity):
        """Adaptive mutation with gradual cooling and stagnation response"""
        generation_ratio = getattr(self, 'current_generation', 0) / max(self.generations, 1)
        cooling_factor = 1.0 - (generation_ratio * 0.3)  # Gradual cooling
        
        mutation_rate = self.base_mutation_rate * cooling_factor
        
        # Boost for fitness stagnation
        if self.fitness_stagnation_counter > self.fitness_stagnation_threshold // 2:
            mutation_rate *= self.mutation_boost_factor
        
        # Boost for diversity stagnation
        if diversity < self.diversity_threshold:
            mutation_rate *= 2.0
        
        return min(mutation_rate, 0.1)

    def crowding_selection(self, fitness_scores, offspring, offspring_fitness):
        """Crowding selection to maintain diversity"""
        new_population = self.population[:]
        
        for i, (child, child_fitness) in enumerate(zip(offspring, offspring_fitness)):
            # Find most similar individual in current population
            similarities = []
            for j, individual in enumerate(self.population):
                hamming_dist = sum(1 for a, b in zip(child, individual) if a != b)
                similarity = 1.0 - (hamming_dist / len(child))
                similarities.append((similarity, j))
            
            # Sort by similarity (most similar first)
            similarities.sort(reverse=True)
            
            # Replace most similar individual if child is better
            most_similar_idx = similarities[0][1]
            if child_fitness > fitness_scores[most_similar_idx]:
                new_population[most_similar_idx] = child
                fitness_scores[most_similar_idx] = child_fitness
        
        self.population = new_population
    
    def diversity_aware_selection(self, fitness_scores):
        """Lightweight selection with diversity consideration when stagnating"""
        try:
            if len(self.population) == 0:
                return self.generate_random_sequence()
            
            # Normal tournament when not stagnating
            if self.fitness_stagnation_counter <= self.fitness_stagnation_threshold // 2:
                return self.tournament_selection(fitness_scores)
            
            # Add randomness when stagnating (30% random, 70% tournament)
            if random.random() < 0.3:
                return random.choice(self.population)
            else:
                return self.tournament_selection(fitness_scores)
                
        except Exception as e:
            print(f"Warning: Diversity-aware selection failed: {e}, using fallback")
            if len(self.population) > 0:
                return random.choice(self.population)
            else:
                return self.generate_random_sequence()
    
    def restart_population(self, fitness_scores):
        """Restart portion of population when stagnating"""
        if len(fitness_scores) != len(self.population):
            print("Warning: Fitness scores and population size mismatch, skipping restart")
            return
            
        num_to_restart = int(self.population_size * self.restart_rate)
        if num_to_restart <= 0:
            return
        
        elite_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], reverse=True)
        keep_indices = set(elite_indices[:self.population_size - num_to_restart])
        
        def generate_sequences_chunk(count):
            return [self.generate_random_sequence() for _ in range(count)]
        
        if num_to_restart <= self.max_workers:
            new_sequences = [self.generate_random_sequence() for _ in range(num_to_restart)]
        else:
            chunk_size = max(1, num_to_restart // self.max_workers)
            chunks = []
            remaining = num_to_restart
            
            while remaining > 0:
                chunk = min(chunk_size, remaining)
                chunks.append(chunk)
                remaining -= chunk
            
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
                new_sequences_chunks = list(executor.map(generate_sequences_chunk, chunks))
            
            new_sequences = [seq for chunk in new_sequences_chunks for seq in chunk]
        
        new_sequences = new_sequences[:num_to_restart]
        
        new_population = []
        new_seq_index = 0
        for i in range(self.population_size):
            if i in keep_indices and i < len(self.population):
                new_population.append(self.population[i])
            elif new_seq_index < len(new_sequences):
                new_population.append(new_sequences[new_seq_index])
                new_seq_index += 1
            else:
                new_population.append(self.generate_random_sequence())
        
        self.population = new_population[:self.population_size]
        print(f"Anti-stagnation: Restarted {num_to_restart} individuals")
    
    def get_best_individuals(self):
        """Get best individuals as list of dictionaries"""
        return [{'sequence': seq, 'fitness': fitness} for seq, fitness in self.best_individuals]
    
    def run_evolution(self):
        """Main evolutionary algorithm loop"""
        print(f"Starting evolution: {self.population_size} individuals, {self.generations} generations")
        print(f"Using {self.max_workers} parallel workers for fitness evaluation")
        print(f"Sequence constraint: {self.sequence_constraint}")
        print(f"Structure constraint: {self.structure_constraint}")
        
        self.initialize_population()
        
        for generation in range(self.generations):
            self.current_generation = generation
                        
            # Evaluate fitness
            fitness_scores = self.evaluate_fitness(self.population)
            
            # Track statistics
            max_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.fitness_history.append((generation, max_fitness, avg_fitness))
            
            # Calculate diversity with sampling for large populations
            try:
                sample_size = 100 if len(self.population) > 200 else None
                diversity = self.calculate_diversity(self.population, sample_size)
            except Exception as e:
                print(f"Warning: Diversity calculation failed: {e}, using fallback")
                diversity = 0.0
            
            # Separate stagnation tracking for fitness and diversity
            if max_fitness > 0 and max_fitness <= self.last_best_fitness + 1e-6:
                self.fitness_stagnation_counter += 1
            else:
                self.fitness_stagnation_counter = 0
                self.last_best_fitness = max_fitness
            
            if diversity <= self.last_diversity + 1e-6:
                self.diversity_stagnation_counter += 1
            else:
                self.diversity_stagnation_counter = 0
                self.last_diversity = diversity
            
            # Early termination check
            if max_fitness >= self.early_termination_fitness:
                self.high_fitness_streak += 1
                if self.high_fitness_streak >= self.high_fitness_streak_threshold:
                    self.early_terminated = True
                    self.termination_reason = f"Early termination: fitness >={self.early_termination_fitness} for {self.high_fitness_streak} generations"
                    print(f"\n{self.termination_reason}")
                    break
            else:
                self.high_fitness_streak = 0
            
            # Anti-stagnation measures
            should_apply_anti_stagnation = (
                (max_fitness == 0.0 or max_fitness < self.early_termination_fitness) and
                (self.fitness_stagnation_counter >= self.fitness_stagnation_threshold or 
                self.diversity_stagnation_counter >= self.diversity_stagnation_threshold)
            )

            
            # Adaptive mutation
            self.mutation_rate = self.adaptive_mutation_rate(diversity)
            
            # Population restart if severely stagnant
            if should_apply_anti_stagnation:
                try:
                    self.restart_population(fitness_scores)
                    self.fitness_stagnation_counter = 0
                    self.diversity_stagnation_counter = 0
                    self.last_best_fitness = 0.0
                    self.last_diversity = 1.0
                    
                    # Re-evaluate after restart
                    fitness_scores = self.evaluate_fitness(self.population)
                    max_fitness = max(fitness_scores)
                    avg_fitness = sum(fitness_scores) / len(fitness_scores)
                    self.last_best_fitness = max_fitness
                    
                    try:
                        diversity = self.calculate_diversity(self.population, sample_size)
                        self.last_diversity = diversity
                    except Exception as e:
                        print(f"Warning: Post-restart diversity calculation failed: {e}")
                        diversity = 0.0
                except Exception as e:
                    print(f"Warning: Population restart failed: {e}, continuing with current population")
                    self.fitness_stagnation_counter = 0
                    self.diversity_stagnation_counter = 0
            
            # Call external callbacks
            for callback in self.callbacks:
                try:
                    callback(generation, max_fitness, avg_fitness, diversity)
                except Exception as e:
                    print(f"Warning: Callback failed: {e}")
            
            # Store high-quality individuals
            for i, (seq, fitness) in enumerate(zip(self.population, fitness_scores)):
                if fitness > 0.9:
                    self.best_individuals.append((seq, fitness))
            
            # Create next generation with 1% elitism and crowding
            new_population = []
            elite_count = max(1, int(self.population_size * 0.01))  # 1% elitism
            
            # Select elites
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # Generate offspring
            offspring = []
            offspring_fitness = []
            
            while len(offspring) < self.population_size - elite_count:
                try:
                    if random.random() < self.crossover_rate:
                        parent1 = self.tournament_selection(fitness_scores)
                        parent2 = self.tournament_selection(fitness_scores)
                        
                        # Use multi-point crossover for long sequences
                        if len(parent1) > 100:
                            child1, child2 = self.multi_point_crossover(parent1, parent2)
                        else:
                            child1, child2 = self.two_point_crossover(parent1, parent2)
                        
                        child1 = self.mutate(child1)
                        child2 = self.mutate(child2)
                        
                        offspring.extend([child1, child2])
                    else:
                        parent = self.tournament_selection(fitness_scores)
                        child = self.mutate(parent)
                        offspring.append(child)
                        
                except Exception as e:
                    print(f"Warning: Error during offspring generation: {e}, using fallback")
                    offspring.append(self.generate_random_sequence())
            
            # Trim offspring to exact size needed
            offspring = offspring[:self.population_size - elite_count]
            
            # Evaluate offspring fitness
            if offspring:
                offspring_fitness = self.evaluate_fitness(offspring)
                
                # Apply crowding selection instead of fitness sharing
                self.crowding_selection(fitness_scores, offspring, offspring_fitness)
            else:
                # Just keep current population if no offspring generated
                self.population = new_population[:self.population_size]
        
        if self.early_terminated:
            print(f"\nEvolution terminated early!")
            print(f"Reason: {self.termination_reason}")
        else:
            print("\nEvolution completed!")
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze final results and prepare output"""
        print("\n=== FINAL ANALYSIS ===")
        
        if self.early_terminated:
            print(f"EARLY TERMINATION: {self.termination_reason}")
        
        # Remove duplicates and sort by fitness
        unique_individuals = list(set(seq for seq, _ in self.best_individuals))
        
        # Re-evaluate fitness for unique individuals
        final_fitness = self.evaluate_fitness(unique_individuals)
        
        # Sort by fitness
        sorted_results = sorted(zip(unique_individuals, final_fitness), key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(sorted_results)} unique high-quality sequences")
        print(f"Elite percentage: 1.0% (fixed)")
        
        if sorted_results:
            # Calculate diversity
            sequences = [seq for seq, _ in sorted_results]
            diversity = self.calculate_diversity(sequences)
            print(f"Average sequence diversity: {diversity:.4f}")
            
            # Show top results
            print("\nTop 10 sequences:")
            for i, (seq, fitness) in enumerate(sorted_results[:10]):
                print(f"{i+1:2d}. {seq} (fitness: {fitness:.4f})")
        
        return sorted_results
    
    def save_results(self, filename="assignment1_results.txt"):
        """Save results to output file"""
        results = self.analyze_results()
        
        with open(filename, 'w') as f:
            for seq, fitness in results:
                if fitness > 0.5:
                    f.write(f"{seq}\n")
        
        print(f"\nResults saved to {filename}")
        print(f"Saved {len([r for r in results if r[1] > 0.5])} sequences")


def main():
    """
    Main function with user input as required by assignment
    """
    print("RNA Folding Evolutionary Algorithm")
    print("==================================")
    
    # Get user input as required
    try:
        population_size = int(input("Please enter the population size: "))
        num_generations = int(input("Please enter the number of generations: "))
    except ValueError:
        print("Invalid input. Using default values.")
        population_size = 100
        num_generations = 100
    
    # Load constraints from CSV (using the first constraint set for testing)
    # In practice, you would specify which constraint set to use
    sequence_constraint = "GARMUWYMNKKSSGMUCCKCYAGCNCMMNGAGKNCWAUKSKRUNCGNMYCNMNSCKCNCNCKUKKSWSAACSSSAMCN"
    structure_constraint = "((((....))))....(((((.......)))))............(((((.......[[[[[[))))).]]]]]]..."
    
    print("Running EA...")
    
    # Create and run EA
    ea = RNAFoldingEA(population_size, num_generations, sequence_constraint, structure_constraint, max_workers=8)
    
    # Add lightweight progress monitoring (no performance impact)
    try:
        from progress_monitor import add_progress_monitoring
        ea = add_progress_monitoring(ea)
    except ImportError:
        print("Progress monitoring not available, running standard EA...")
    
    ea.run_evolution()
    ea.save_results()

if __name__ == "__main__":
    main()