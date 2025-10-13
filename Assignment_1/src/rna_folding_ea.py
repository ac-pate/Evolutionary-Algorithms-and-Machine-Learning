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
    
    def __init__(self, population_size, generations, sequence_constraint, structure_constraint, max_workers=8):
        """
        Initialize the Evolutionary Algorithm
        
        Args:
            population_size (int): Number of individuals in population
            generations (int): Number of generations to evolve
            sequence_constraint (str): IUPAC notation string
            structure_constraint (str): Dot-bracket notation string
            max_workers (int): Number of parallel workers for fitness evaluation
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
        self.elite_percentage = 0.1  # Keep top 10% by default
        self.elitism_count = max(1, int(population_size * self.elite_percentage))
        
        # Early termination settings
        self.early_termination_fitness = 0.95
        self.high_fitness_streak_threshold = 25
        self.high_fitness_streak = 0
        
        # Anti-stagnation settings (using existing pattern)
        self.stagnation_threshold = 10  # Generations without improvement
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0
        
        self.diversity_threshold = 0.2  # Minimum diversity to maintain
        self.restart_rate = 0.4  # Percentage of population to restart
        self.base_mutation_rate = self.mutation_rate  # Store original rate
        self.mutation_boost_factor = 3.0  # Boost factor when stagnant
        
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
        """
        check if sequence satisfys iupac constraints
        
        Args:
            sequence (str): RNA sequence to validate
            
        Returns:
            bool: True if sequence is valid, False otherwise
        """
        if len(sequence) != self.sequence_length:
            return False
            
        for i, (seq_base, constraint) in enumerate(zip(sequence, self.sequence_constraint)):
            if constraint in self.iupac_codes:
                if seq_base not in self.iupac_codes[constraint]:
                    return False
            else:
                # unknown iupac code - treat as any nucleotide allowed
                if seq_base not in ['A', 'U', 'C', 'G']:
                    return False
        return True
    
    def generate_random_sequence(self):
        """
        Generate a random RNA sequence that satisfies IUPAC constraints
        
        Returns:
            str: Valid RNA sequence
        """
        sequence = []
        for constraint in self.sequence_constraint:
            if constraint in self.iupac_codes:
                allowed_bases = self.iupac_codes[constraint]
            else:
                allowed_bases = ['A', 'U', 'C', 'G']  # Default to any nucleotide
            sequence.append(random.choice(allowed_bases))
        return ''.join(sequence)
    
    def initialize_population(self):
        """
        initialize pop with valid random sequences
        ensures diversity by avoiding duplicates
        """
        print("Initializing population...")
        population = []
        seen_sequences = set()
        
        attempts = 0
        max_attempts = self.population_size * 10  # Prevent infinite loops
        
        while len(population) < self.population_size and attempts < max_attempts:
            sequence = self.generate_random_sequence()
            if sequence not in seen_sequences and self.is_valid_sequence(sequence):
                population.append(sequence)
                seen_sequences.add(sequence)
            attempts += 1
            
        # fill remaining slots if needed (should not happen with good constraints)
        while len(population) < self.population_size:
            sequence = self.generate_random_sequence()
            population.append(sequence)
            
        self.population = population
        print(f"Initialized population with {len(population)} individuals")
    
    def fold_rna_ipknot(self, sequence):
        """
        use ipknot via docker to predict rna secondary structure
        
        Args:
            sequence (str): RNA sequence to fold
            
        Returns:
            str: predicted secondary structure in dot-bracket notation
        """
        try:
            # create temporary FASTA file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
                f.write(f">sequence\n{sequence}\n")
                temp_file = f.name
            
            # run ipknot via docker
            cmd = ["sudo", "docker", "exec", "ipknot_runner", "ipknot", f"/work/{os.path.basename(temp_file)}"]
            
            # copy file to Docker container working directory
            copy_cmd = ["sudo", "docker", "cp", temp_file, f"ipknot_runner:/work/{os.path.basename(temp_file)}"]
            subprocess.run(copy_cmd, check=True, capture_output=True)
            
            # Run IPknot
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            
            # Parse output to extract structure
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                if line.startswith('>'):
                    # Next line should be sequence, line after that should be structure
                    if i + 2 < len(lines):
                        structure = lines[i + 2].strip()
                        # Clean up temporary file
                        os.unlink(temp_file)
                        return structure
            
            # Cleanup and return empty structure if parsing failed
            os.unlink(temp_file)
            return '.' * len(sequence)  # All unpaired as fallback
            
        except Exception as e:
            print(f"Error folding sequence {sequence}: {e}")
            # Cleanup temp file if it exists
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)
            return '.' * len(sequence)  # All unpaired as fallback
    
    def calculate_structure_fitness(self, predicted_structure, target_structure):
        """
        Calculate fitness based on structure similarity
        
        Args:
            predicted_structure (str): Predicted dot-bracket structure
            target_structure (str): Target dot-bracket structure
            
        Returns:
            float: Fitness score (0.0 to 1.0, higher is better)
        """
        if len(predicted_structure) != len(target_structure):
            return 0.0
        
        # Count matching positions
        matches = sum(1 for p, t in zip(predicted_structure, target_structure) if p == t)
        return matches / len(target_structure)
    
    def evaluate_fitness(self, sequences):
        """
        Evaluate fitness for a batch of sequences using parallel processing
        
        Args:
            sequences (list): List of RNA sequences
            
        Returns:
            list: List of fitness scores
        """
        def evaluate_single(sequence):
            if not self.is_valid_sequence(sequence):
                return 0.0  # Invalid sequences get zero fitness
            
            predicted_structure = self.fold_rna_ipknot(sequence)
            return self.calculate_structure_fitness(predicted_structure, self.structure_constraint)
        
        # use parallel processing for fitness evaluation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fitness_scores = list(executor.map(evaluate_single, sequences))
        
        return fitness_scores
    
    def tournament_selection(self, fitness_scores):
        """
        Tournament selection for parent selection
        
        Args:
            fitness_scores (list): Fitness scores for population
            
        Returns:
            str: Selected parent sequence
        """
        try:
            # Validate inputs
            if len(self.population) == 0 or len(fitness_scores) == 0:
                raise ValueError("Empty population or fitness scores")
            
            if len(self.population) != len(fitness_scores):
                raise ValueError(f"Population size ({len(self.population)}) != fitness scores size ({len(fitness_scores)})")
            
            # Ensure tournament size doesn't exceed population size
            actual_tournament_size = min(self.tournament_size, len(self.population))
            
            tournament_indices = random.sample(range(len(self.population)), actual_tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            
            # Validate winner index
            if 0 <= winner_index < len(self.population):
                return self.population[winner_index]
            else:
                raise IndexError(f"Winner index {winner_index} out of bounds for population size {len(self.population)}")
                
        except Exception as e:
            print(f"Warning: Tournament selection failed: {e}, using fallback")
            # Safe fallback - return a random individual
            if len(self.population) > 0:
                return random.choice(self.population)
            else:
                # Emergency fallback - generate new sequence
                return self.generate_random_sequence()
    
    def two_point_crossover(self, parent1, parent2):
        """
        Two-point crossover with constraint validation
        
        Args:
            parent1 (str): First parent sequence
            parent2 (str): Second parent sequence
            
        Returns:
            tuple: Two offspring sequences
        """
        if len(parent1) != len(parent2):
            return parent1, parent2
        
        # Choose two crossover points
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)
        
        # Create offspring
        offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        # Validate and repair if necessary
        offspring1 = self.repair_sequence(offspring1)
        offspring2 = self.repair_sequence(offspring2)
        
        return offspring1, offspring2
    
    def mutate(self, sequence):
        """
        Point mutation with constraint validation
        
        Args:
            sequence (str): Sequence to mutate
            
        Returns:
            str: Mutated sequence
        """
        sequence_list = list(sequence)
        
        for i in range(len(sequence_list)):
            if random.random() < self.mutation_rate:
                # Get allowed bases for this position
                constraint = self.sequence_constraint[i]
                if constraint in self.iupac_codes:
                    allowed_bases = self.iupac_codes[constraint]
                else:
                    allowed_bases = ['A', 'U', 'C', 'G']
                
                # Mutate to a different valid base
                current_base = sequence_list[i]
                possible_mutations = [base for base in allowed_bases if base != current_base]
                if possible_mutations:
                    sequence_list[i] = random.choice(possible_mutations)
        
        return ''.join(sequence_list)
    
    def repair_sequence(self, sequence):
        """
        Repair sequence to satisfy IUPAC constraints
        
        Args:
            sequence (str): Potentially invalid sequence
            
        Returns:
            str: Valid sequence
        """
        sequence_list = list(sequence)
        
        for i, (base, constraint) in enumerate(zip(sequence_list, self.sequence_constraint)):
            if constraint in self.iupac_codes:
                allowed_bases = self.iupac_codes[constraint]
                if base not in allowed_bases:
                    sequence_list[i] = random.choice(allowed_bases)
        
        return ''.join(sequence_list)
    
    def calculate_diversity(self, sequences):
        """
        Calculate average normalized Hamming distance for diversity measurement
        PARALLELIZED VERSION - much faster than O(N²) single-threaded
        
        Args:
            sequences (list): List of sequences
            
        Returns:
            float: Average diversity score
        """
        if len(sequences) < 2:
            return 0.0
        
        def calculate_distances_chunk(chunk_indices):
            """Calculate distances for a chunk of sequence pairs"""
            total_distance = 0.0
            comparisons = 0
            
            for i in chunk_indices:
                if i >= len(sequences):  # Safety check
                    continue
                for j in range(i + 1, len(sequences)):
                    if j >= len(sequences):  # Safety check
                        continue
                    hamming_dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                    normalized_dist = hamming_dist / len(sequences[i])
                    total_distance += normalized_dist
                    comparisons += 1
            
            return total_distance, comparisons
        
        # Split work into chunks for parallel processing - Fixed chunking
        num_sequences = len(sequences)
        chunk_size = max(1, num_sequences // self.max_workers)
        chunks = []
        
        for i in range(0, num_sequences, chunk_size):
            chunk_end = min(i + chunk_size, num_sequences)
            chunk_indices = list(range(i, chunk_end))
            if chunk_indices:  # Only add non-empty chunks
                chunks.append(chunk_indices)
        
        if not chunks:  # Fallback for edge cases
            return 0.0
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
            results = list(executor.map(calculate_distances_chunk, chunks))
        
        # Combine results
        total_distance = sum(result[0] for result in results)
        total_comparisons = sum(result[1] for result in results)
        
        return total_distance / total_comparisons if total_comparisons > 0 else 0.0
    
    def adaptive_mutation_rate(self, diversity):
        """
        Adapt mutation rate based on stagnation and diversity
        
        Args:
            diversity (float): Current population diversity
            
        Returns:
            float: Adjusted mutation rate
        """
        mutation_rate = self.base_mutation_rate
        
        # Increase mutation if stagnating
        if self.stagnation_counter > self.stagnation_threshold // 2:
            mutation_rate *= self.mutation_boost_factor
        
        # Increase mutation if diversity is too low
        if diversity < self.diversity_threshold:
            mutation_rate *= 2.0
        
        # Cap mutation rate to prevent chaos
        return min(mutation_rate, 0.1)
    
    def diversity_aware_selection(self, fitness_scores):
        """
        Lightweight selection that adds randomness when stagnating
        Much faster than calculating diversity for every parent selection
        
        Args:
            fitness_scores (list): Fitness scores for population
            
        Returns:
            str: Selected parent sequence
        """
        try:
            # Validate inputs
            if len(self.population) == 0:
                return self.generate_random_sequence()
            
            # When not stagnating, use normal tournament selection
            if self.stagnation_counter <= self.stagnation_threshold // 2:
                return self.tournament_selection(fitness_scores)
            
            # When stagnating, add some randomness to encourage diversity
            # 30% random selection to inject diversity, 70% tournament selection
            if random.random() < 0.3:
                # Random selection for diversity
                return random.choice(self.population)
            else:
                # Tournament selection for quality
                return self.tournament_selection(fitness_scores)
                
        except Exception as e:
            print(f"Warning: Diversity-aware selection failed: {e}, using fallback")
            # Safe fallback
            if len(self.population) > 0:
                return random.choice(self.population)
            else:
                return self.generate_random_sequence()
    
    def restart_population(self, fitness_scores):
        """
        Restart portion of population when stagnating
        PARALLELIZED VERSION - generates new sequences in parallel
        
        Args:
            fitness_scores (list): Current fitness scores
        """
        if len(fitness_scores) != len(self.population):
            print("Warning: Fitness scores and population size mismatch, skipping restart")
            return
            
        num_to_restart = int(self.population_size * self.restart_rate)
        if num_to_restart <= 0:
            return
        
        # Keep the best individuals
        elite_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], reverse=True)
        keep_indices = set(elite_indices[:self.population_size - num_to_restart])
        
        # Generate new random individuals in parallel
        def generate_sequences_chunk(count):
            """Generate multiple random sequences"""
            return [self.generate_random_sequence() for _ in range(count)]
        
        # Parallel generation of new sequences - Fixed chunking
        if num_to_restart <= self.max_workers:
            # If few sequences to generate, just do it directly
            new_sequences = [self.generate_random_sequence() for _ in range(num_to_restart)]
        else:
            # Split into chunks for parallel generation
            chunk_size = max(1, num_to_restart // self.max_workers)
            chunks = []
            remaining = num_to_restart
            
            while remaining > 0:
                chunk = min(chunk_size, remaining)
                chunks.append(chunk)
                remaining -= chunk
            
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
                new_sequences_chunks = list(executor.map(generate_sequences_chunk, chunks))
            
            # Flatten the results
            new_sequences = [seq for chunk in new_sequences_chunks for seq in chunk]
        
        # Ensure we have exactly the right number of new sequences
        new_sequences = new_sequences[:num_to_restart]
        
        # Build new population
        new_population = []
        new_seq_index = 0
        for i in range(self.population_size):
            if i in keep_indices and i < len(self.population):
                new_population.append(self.population[i])
            elif new_seq_index < len(new_sequences):
                new_population.append(new_sequences[new_seq_index])
                new_seq_index += 1
            else:
                # Fallback: generate a new sequence
                new_population.append(self.generate_random_sequence())
        
        # Ensure population is exactly the right size
        self.population = new_population[:self.population_size]
        print(f"Anti-stagnation: Restarted {num_to_restart} individuals (parallelized)")
    
    def get_best_individuals(self):
        """
        Get best individuals as list of dictionaries for compatibility
        
        Returns:
            list: List of dicts with 'sequence' and 'fitness' keys
        """
        return [{'sequence': seq, 'fitness': fitness} for seq, fitness in self.best_individuals]
    
    def run_evolution(self):
        """
        Main evolutionary algorithm loop
        """
        print(f"Starting evolution: {self.population_size} individuals, {self.generations} generations")
        print(f"Using {self.max_workers} parallel workers for fitness evaluation")
        print(f"Sequence constraint: {self.sequence_constraint}")
        print(f"Structure constraint: {self.structure_constraint}")
        
        # Initialize population
        self.initialize_population()
        
        for generation in range(self.generations):
                        
            # Evaluate fitness
            fitness_scores = self.evaluate_fitness(self.population)
            
            # Track statistics
            max_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.fitness_history.append((generation, max_fitness, avg_fitness))
            
            # Calculate diversity with error handling
            try:
                diversity = self.calculate_diversity(self.population)
            except Exception as e:
                print(f"Warning: Diversity calculation failed: {e}, using fallback")
                diversity = 0.0
            
            # Anti-stagnation: Check for stagnation
            if max_fitness > 0 and max_fitness <= self.last_best_fitness + 1e-6:  # No significant improvement
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                self.last_best_fitness = max_fitness
            
            # PRIORITY 1: Check early termination condition FIRST
            if max_fitness >= self.early_termination_fitness:
                self.high_fitness_streak += 1
                if self.high_fitness_streak >= self.high_fitness_streak_threshold:
                    self.early_terminated = True
                    self.termination_reason = f"Early termination: fitness >={self.early_termination_fitness} for {self.high_fitness_streak} generations"
                    print(f"\n{self.termination_reason}")
                    break
            else:
                self.high_fitness_streak = 0
            
            # PRIORITY 2: Anti-stagnation measures (but NOT when close to early termination)
            # Prevent anti-stagnation interference when we're close to early termination
            near_termination_threshold = self.high_fitness_streak_threshold - 15  
            should_apply_anti_stagnation = (
                max_fitness > 0 and max_fitness < self.early_termination_fitness and
                (self.stagnation_counter >= self.stagnation_threshold or diversity < self.diversity_threshold)
            )
            
            # Anti-stagnation: Adaptive mutation rate
            self.mutation_rate = self.adaptive_mutation_rate(diversity)
            
            # Anti-stagnation: Population restart if severely stagnant (but not near termination)
            if should_apply_anti_stagnation:
                try:
                    self.restart_population(fitness_scores)
                    self.stagnation_counter = 0
                    # Reset baseline after restart to prevent immediate re-triggering
                    self.last_best_fitness = 0.0
                    # Re-evaluate after restart
                    fitness_scores = self.evaluate_fitness(self.population)
                    max_fitness = max(fitness_scores)
                    avg_fitness = sum(fitness_scores) / len(fitness_scores)
                    # Update baseline to new best after restart
                    self.last_best_fitness = max_fitness
                    try:
                        diversity = self.calculate_diversity(self.population)
                    except Exception as e:
                        print(f"Warning: Post-restart diversity calculation failed: {e}")
                        diversity = 0.0
                except Exception as e:
                    print(f"Warning: Population restart failed: {e}, continuing with current population")
                    self.stagnation_counter = 0
            
            # Call external callbacks (e.g., wandb logging, progress monitoring)
            for callback in self.callbacks:
                try:
                    callback(generation, max_fitness, avg_fitness, diversity)
                except Exception as e:
                    print(f"Warning: Callback failed: {e}")
            
            # Store best individuals
            for i, (seq, fitness) in enumerate(zip(self.population, fitness_scores)):
                if fitness > 0.9:  # High-quality threshold
                    self.best_individuals.append((seq, fitness))
            
            # Create next generation
            new_population = []

            # Anti-stagnation: Dynamic elitism (reduce when stagnating)
            current_elite_percentage = self.elite_percentage
            if self.stagnation_counter > 20:
                current_elite_percentage = 0.01  # Keep only 1% when severely stuck
            elif self.stagnation_counter > 15:
                current_elite_percentage = 0.02  # Keep only 2% when very stuck
            elif self.stagnation_counter > 10:
                current_elite_percentage = max(0.03, self.elite_percentage / 3)
            elif self.stagnation_counter > 5:
                current_elite_percentage = max(0.05, self.elite_percentage / 2)

            elite_count = max(1, int(self.population_size * current_elite_percentage))

            # MODIFIED: Elite-biased selection instead of direct copying
            # Create a selection pool with higher probability for elites
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
            elite_bias_pool = []

            # Add elites multiple times to increase their selection probability
            for idx in elite_indices:
                elite_bias_pool.extend([self.population[idx]] * 3)  # 3x selection probability

            # Add rest of population once
            for i, individual in enumerate(self.population):
                if i not in elite_indices:
                    elite_bias_pool.append(individual)

            # Now generate offspring using elite-biased selection
            while len(new_population) < self.population_size:
                try:
                    if random.random() < self.crossover_rate:
                        # RANDOMIZED PARENT SELECTION with multiple strategies
                        selection_strategy = random.random()
                        
                        if selection_strategy < 0.6:  # 60% - Elite-biased selection (current method)
                            parent1 = random.choice(elite_bias_pool)
                            parent2 = random.choice(elite_bias_pool)
                          
                        else:  # 40% - Completely random selection (pure exploration)
                            parent1 = random.choice(self.population)
                            parent2 = random.choice(self.population)
                        
                        child1, child2 = self.two_point_crossover(parent1, parent2)
                        
                        # Apply mutation to ALL offspring (including elite-derived)
                        child1 = self.mutate(child1)
                        child2 = self.mutate(child2)
                        
                        new_population.extend([child1, child2])
                        
                    else:
                        # RANDOMIZED MUTATION-ONLY PARENT SELECTION
                        mutation_strategy = random.random()

                        if mutation_strategy < 0.6:  # 60% - Elite-biased
                            parent = random.choice(elite_bias_pool)
                        else:  # 40% - Random
                            parent = random.choice(self.population)
                        
                        child = self.mutate(parent)
                        new_population.append(child)
                        
                except Exception as e:
                    print(f"Warning: Error during offspring generation: {e}, using fallback")
                    new_population.append(self.generate_random_sequence())

            # Trim to exact population size
            self.population = new_population[:self.population_size]
        
        if self.early_terminated:
            print(f"\nEvolution terminated early!")
            print(f"Reason: {self.termination_reason}")
        else:
            print("\nEvolution completed!")
        self.analyze_results()
    
    def analyze_results(self):
        """
        Analyze final results and prepare output
        """
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
        print(f"Elite percentage: {self.elite_percentage*100:.1f}% ({self.elitism_count} individuals)")
        
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
        """
        Save results to output file
        
        Args:
            filename (str): Output filename
        """
        # Get final results
        results = self.analyze_results()
        
        # Save sequences to file
        with open(filename, 'w') as f:
            for seq, fitness in results:
                if fitness > 0.5:  # Only save decent quality sequences
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