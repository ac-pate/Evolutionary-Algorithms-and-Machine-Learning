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
        tournament_indices = random.sample(range(len(self.population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return self.population[winner_index]
    
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
        
        Args:
            sequences (list): List of sequences
            
        Returns:
            float: Average diversity score
        """
        if len(sequences) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                hamming_dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                normalized_dist = hamming_dist / len(sequences[i])
                total_distance += normalized_dist
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
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
            
            # Calculate diversity
            diversity = self.calculate_diversity(self.population)
            
            # Check early termination condition
            if max_fitness >= self.early_termination_fitness:
                self.high_fitness_streak += 1
                if self.high_fitness_streak >= self.high_fitness_streak_threshold:
                    self.early_terminated = True
                    self.termination_reason = f"Early termination: fitness >={self.early_termination_fitness} for {self.high_fitness_streak} generations"
                    print(f"\n{self.termination_reason}")
                    break
            else:
                self.high_fitness_streak = 0
            
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
            
            # Elitism: Keep best individuals (using configurable elite percentage)
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elitism_count]
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # Generate rest of population through crossover and mutation
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1 = self.tournament_selection(fitness_scores)
                    parent2 = self.tournament_selection(fitness_scores)
                    child1, child2 = self.two_point_crossover(parent1, parent2)
                    
                    # Mutation
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    new_population.extend([child1, child2])
                else:
                    # Just mutation
                    parent = self.tournament_selection(fitness_scores)
                    child = self.mutate(parent)
                    new_population.append(child)
            
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