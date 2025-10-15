# Achal Patel - 40227663

import random
import numpy as np
import subprocess
import tempfile
import os
import csv
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
    
    def __init__(self, population_size, generations, sequence_constraint, structure_constraint, max_workers=8, elite_percentage=0.01, enable_cache_preloading=False):
        """
        Initialize the Evolutionary Algorithm
        
        Args:
            population_size (int): Number of individuals in population
            generations (int): Number of generations to evolve
            sequence_constraint (str): IUPAC notation string
            structuyre_constraint (str): Dot-bracket notation string
            max_workers (int): Number of parallel workers for fitness evaluation
            elite_percentage (float): Percentage of pipulation to keep as elites (default: 0.01 = 1%)
            enable_cache_preloading (bool): Whether to load fitness cache from previous runs (default: False)
        """
        self.population_size = population_size
        self.generations = generations
        self.sequence_constraint = sequence_constraint
        self.structure_constraint = structure_constraint
        self.sequence_length = len(sequence_constraint)
        self.max_workers = max_workers
        self.enable_cache_preloading = enable_cache_preloading
        
        # === PROBLEM TYPE DETECTION ===
        self.is_unknown_problem = len([c for c in sequence_constraint if c == 'N']) > len(sequence_constraint) * 0.7

        
        # evolutionary params (will be optimized based on hardware)
        self.crossover_rate = 0.8
        self.mutation_rate = 0.02  
        self.tournament_size = 3
        # Increase elitism for .2 problems to preserve good solutions
        if self.is_unknown_problem:
            self.elite_percentage = max(elite_percentage, 0.06)  # At least 6% for .2 problems cause of huge possible sample space
        else:
            self.elite_percentage = elite_percentage  # Keep default for .1 problems
        
        self.elitism_count = max(1, int(population_size * self.elite_percentage))
        
        # Early termination settings
        self.early_termination_fitness = 0.95
        self.high_fitness_streak_threshold = 5
        self.high_fitness_streak = 0
        
        # Fitness cache for optimization
        self.fitness_cache = {}
        
        # Load previous cache if enabled
        if self.enable_cache_preloading:
            self.preload_cache_from_previous_runs()
        
        # Stagnation settings - separated fitness and diversity
        self.fitness_stagnation_threshold = 10
        self.diversity_stagnation_threshold = 15
        self.fitness_stagnation_couter = 0
        self.diversity_stagnation_counter = 0
        self.last_best_fitness = 0.0
        self.last_real_fitness = 0.0  # Track real fitness separately from boosted
        self.last_diversity = 1.0
        
        # Diversity-aware termination cache (performance optimization)
        self.last_diversity_check_generation = -1
        self.cached_diversity_score = 0.0
        self.cached_min_diverse_fitness = 0.0
        
        # Diversity stagnation parameters
        self.diversity_threshold = 0.3
        self.restart_rate = self.restart_rate = 0.3 if self.is_unknown_problem else 0.4
        
        # Fitness stagnation parameters - mutation rate boost instead of fitness manipulation
        self.mutation_rate_boost_factor = 4.0  # Increase mutation rate by 4x when stagnating
        self.mutation_boost_generations = 5  # How long boost lasts
        self.mutation_boost_active = 0  # Counter for active boost
        self.fitness_threshold_for_boost = 0.85  # Only boost when fitness > 0.85
        
        self.base_mutation_rate = self.mutation_rate
        
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
    
    def fold_rna_ipknot(self, sequence_or_list):
        """Use IPknot via Docker to predict RNA secondary structure - supports both single and batch"""
        # Handle single sequence (backward compatibility)
        if isinstance(sequence_or_list, str):
            return self._fold_single_sequence(sequence_or_list)
        
        # Handle batch of sequences
        sequences = sequence_or_list
        if not sequences:
            return []
        
        try:
            # Create single batch file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
                for i, seq in enumerate(sequences):
                    f.write(f">seq_{i}\n{seq}\n")
                batch_file = f.name
            
            # Copy to container and execute
            base_name = os.path.basename(batch_file)
            copy_cmd = ["docker", "cp", batch_file, f"ipknot_runner:/work/{base_name}"]
            subprocess.run(copy_cmd, check=True, capture_output=True)
            
            cmd = ["docker", "exec", "ipknot_runner", "ipknot", f"/work/{base_name}"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30*len(sequences))

            # Parse batch results
            structures = []
            lines = result.stdout.strip().split('\n')
            current_structure = None
            skip_next = False
            
            for line in lines:
                if line.startswith('>'):
                    if current_structure:
                        structures.append(current_structure)
                    current_structure = None
                    skip_next = True
                elif skip_next and line.strip():
                    # Skip sequence line
                    skip_next = False
                    continue
                elif not skip_next and line.strip():
                    current_structure = line.strip()
            
            if current_structure:
                structures.append(current_structure)
            
            os.unlink(batch_file)
            
            # Ensure we return exactly the right number of structures
            while len(structures) < len(sequences):
                structures.append('.' * len(sequences[len(structures)]))
                
            return structures[:len(sequences)]
            
        except Exception as e:
            if 'batch_file' in locals() and os.path.exists(batch_file):
                os.unlink(batch_file)
            return ['.' * len(seq) for seq in sequences]
    
    def _fold_single_sequence(self, sequence):
        """Original single sequence folding logic"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
                f.write(f">sequence\n{sequence}\n")
                temp_file = f.name
            
            cmd = ["docker", "exec", "ipknot_runner", "ipknot", f"/work/{os.path.basename(temp_file)}"]
            copy_cmd = ["docker", "cp", temp_file, f"ipknot_runner:/work/{os.path.basename(temp_file)}"]
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
        """Evaluate fitness for batch of sequences with caching and parallel processing"""
        def evaluate_single(sequence):
            if not self.is_valid_sequence(sequence):
                return 0.0
            
            # Check cache first
            if sequence in self.fitness_cache:
                return self.fitness_cache[sequence]
            
            # Use single sequence folding for parallel processing
            predicted_structure = self.fold_rna_ipknot(sequence)
            fitness = self.calculate_structure_fitness(predicted_structure, self.structure_constraint)
            
            # Cache result
            self.fitness_cache[sequence] = fitness
            return fitness
        
        # Separate cached vs uncached for potential future batch optimization
        cached_results = {}
        uncached_sequences = []
        uncached_indices = []
        
        for i, seq in enumerate(sequences):
            if not self.is_valid_sequence(seq):
                cached_results[i] = 0.0
            elif seq in self.fitness_cache:
                cached_results[i] = self.fitness_cache[seq]
            else:
                uncached_sequences.append(seq)
                uncached_indices.append(i)
        
        # Process uncached sequences in parallel (restore original threading)
        if uncached_sequences:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                uncached_fitness = list(executor.map(evaluate_single, uncached_sequences))
            
            for fitness, idx in zip(uncached_fitness, uncached_indices):
                cached_results[idx] = fitness
        
        # Return results in original order
        return [cached_results[i] for i in range(len(sequences))]
    
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
        generation_ratio = getattr(self, 'current_generation', 0) / (2 * max(self.generations, 1)) # Slower cooling 2x
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
    
    def preload_cache_from_previous_runs(self):
        """Load fitness cache from previous experiments (if enabled)"""
        cache_file = "cache/fitness_cache.json"
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    self.fitness_cache.update(cached_data)
        except Exception as e:
            pass
            
    def save_cache(self):
        """Save fitness cache for future runs (if cache preloading enabled)"""
        if not self.enable_cache_preloading:
            return
            
        try:
            os.makedirs("cache", exist_ok=True)
            cache_file = "cache/fitness_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(self.fitness_cache, f)
        except Exception as e:
            pass
    
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
        cooling_factor = max(0.1, 1.0 - (generation_ratio * 0.3))  # Ensure minimum 10% of base rate
        
        mutation_rate = self.base_mutation_rate * cooling_factor
        
        # Apply mutation boost if active (replaces old fitness stagnation boost)
        if self.mutation_boost_active > 0:
            mutation_rate *= self.mutation_rate_boost_factor
            # No need to print every generation - already logged when activated
        
        # Boost for diversity stagnation
        if diversity < self.diversity_threshold:
            mutation_rate *= 2.0
        
        return min(mutation_rate, 0.1)

    def crowding_selection(self, fitness_scores, offspring, offspring_fitness, protected_indices=None):
        """Crowding selection to maintain diversity - optimized with parallel processing
        
        Args:
            protected_indices: Set of indices that should not be replaced (e.g., elites)
        """
        new_population = self.population[:]
        protected_indices = protected_indices or set()
        
        def find_most_similar(child):
            """Find most similar individual to child in current population"""
            best_similarity = -1
            best_idx = 0
            for j, individual in enumerate(self.population):
                # Skip protected individuals (elites)
                if j in protected_indices:
                    continue
                    
                # Fast Hamming distance calculation
                hamming_dist = sum(1 for a, b in zip(child, individual) if a != b)
                similarity = 1.0 - (hamming_dist / len(child))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_idx = j
            return best_similarity, best_idx
        
        # Parallel similarity computation for all offspring
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(offspring))) as executor:
            most_similar_results = list(executor.map(find_most_similar, offspring))
        
        # Apply replacements (avoiding protected individuals)
        for i, (child, child_fitness) in enumerate(zip(offspring, offspring_fitness)):
            similarity, most_similar_idx = most_similar_results[i]
            # Only replace if target is not protected and child is better
            if (most_similar_idx not in protected_indices and 
                child_fitness > fitness_scores[most_similar_idx]):
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
            if len(self.population) > 0:
                return random.choice(self.population)
            else:
                return self.generate_random_sequence()
    
    def restart_population(self, fitness_scores):
        """Restart portion of population when stagnating (diversity focused)"""
        if len(fitness_scores) != len(self.population):
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
    
    def apply_mutation_boost(self, real_max_fitness):
        """Apply mutation rate boost when fitness stagnates at high levels (>0.75)"""
        if real_max_fitness <= self.fitness_threshold_for_boost:
            return False
        
        # Activate mutation boost
        self.mutation_boost_active = self.mutation_boost_generations
        return True
    
    def get_best_individuals(self):
        """Get best individuals as list of dictionaries"""
        return [{'sequence': seq, 'fitness': fitness} for seq, fitness in self.best_individuals]
    
    def get_diverse_top_sequences(self, num_sequences=5, min_diversity_threshold=0.15, verbose=False):
        """
        Get top sequences with STRONG diversity enforcement for grading
        
        Args:
            num_sequences: Number of sequences to select (default 5 for assignment)
            min_diversity_threshold: Minimum required diversity between sequences (0.15 = 15% different)
        
        Returns:
            List of (sequence, fitness) tuples that are both high-fitness AND diverse
        """
        # Get all unique results sorted by fitness
        unique_individuals = list(set(seq for seq, _ in self.best_individuals))
        
        # Fallback: if no best_individuals collected, use current population
        if not unique_individuals and hasattr(self, 'population') and self.population:
            unique_individuals = list(set(self.population))
        
        if not unique_individuals:
            return []
        
        final_fitness = self.evaluate_fitness(unique_individuals)
        all_results = sorted(zip(unique_individuals, final_fitness), key=lambda x: x[1], reverse=True)
        
        if len(all_results) <= 1:
            return all_results
        
        # Header only if verbose
        if verbose:
            print(f"\nSelecting {num_sequences} DIVERSE sequences (min {min_diversity_threshold:.1%} different):")
        
        # Always start with the best fitness sequence
        selected = [all_results[0]]
        
        # For remaining sequences, enforce strong diversity requirement
        remaining_candidates = all_results[1:]
        
        for position in range(2, num_sequences + 1):
            best_candidate = None
            best_score = -1
            best_min_diversity = 0
            
            for candidate_seq, candidate_fitness in remaining_candidates:
                # Calculate minimum diversity from ALL already selected sequences
                diversities = []
                for selected_seq, _ in selected:
                    hamming_dist = sum(1 for a, b in zip(candidate_seq, selected_seq) if a != b)
                    diversity = hamming_dist / len(candidate_seq)
                    diversities.append(diversity)
                
                min_diversity = min(diversities)
                
                # STRONG diversity requirement: reject if too similar to ANY selected sequence
                if min_diversity < min_diversity_threshold:
                    continue  # Skip this candidate - too similar
                
                # For candidates that pass diversity threshold, balance fitness and diversity
                # But prioritize passing the diversity threshold first
                avg_diversity = sum(diversities) / len(diversities)
                combined_score = candidate_fitness * 0.6 + avg_diversity * 0.4
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = (candidate_seq, candidate_fitness)
                    best_min_diversity = min_diversity
            
            if best_candidate:
                selected.append(best_candidate)
                remaining_candidates.remove(best_candidate)
            else:
                # No candidates meet diversity threshold - lower the bar slightly and try again
                min_diversity_threshold *= 0.8  # Lower threshold by 20%
                if min_diversity_threshold < 0.05:  # Don't go below 5%
                    # Just add the next best fitness sequences if diversity can't be achieved
                    for remaining_seq, remaining_fitness in remaining_candidates:
                        if len(selected) < num_sequences:
                            selected.append((remaining_seq, remaining_fitness))
                    break
        
        return selected
    
    def run_evolution(self):
        """Main evolutionary algorithm loop"""
        # Add progress monitoring if available
        try:
            from src.progress_monitor import add_progress_monitoring
            add_progress_monitoring(self)
        except ImportError:
            pass
        
        self.initialize_population()
        
        for generation in range(self.generations):
            self.current_generation = generation
                        
            # Evaluate fitness
            fitness_scores = self.evaluate_fitness(self.population)
            
            # Track REAL fitness (before any boosting) for stagnation detection
            real_max_fitness = max(fitness_scores)
            
            # Track statistics
            max_fitness = real_max_fitness  # Will be updated if boost is applied
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.fitness_history.append((generation, max_fitness, avg_fitness))
            
            # Calculate diversity with sampling for large populations
            try:
                sample_size = 100 if len(self.population) > 200 else None
                diversity = self.calculate_diversity(self.population, sample_size)
            except Exception as e:
                print(f"Warning: Diversity calculation failed: {e}, using fallback")
                diversity = 0.0
            
            # Store diversity history for plotting
            self.diversity_history.append(diversity)
            
            # Separate stagnation tracking for fitness and diversity
            # Use REAL fitness for stagnation detection (not boosted fitness)
            if real_max_fitness > 0 and real_max_fitness <= self.last_real_fitness + 1e-6:
                self.fitness_stagnation_counter += 1
            else:
                self.fitness_stagnation_counter = 0
                self.last_real_fitness = real_max_fitness
            
            if diversity <= self.last_diversity + 1e-6:
                self.diversity_stagnation_counter += 1
            else:
                self.diversity_stagnation_counter = 0
                self.last_diversity = diversity
            
            # Decrease mutation boost counter if active
            if self.mutation_boost_active > 0:
                self.mutation_boost_active -= 1
            
            # Early termination check (use real fitness, not boosted)
            if real_max_fitness >= self.early_termination_fitness:
                self.high_fitness_streak += 1
                
                # Enhanced termination: check diversity when close to termination
                if self.high_fitness_streak >= self.high_fitness_streak_threshold:
                    
                    # Only check diversity if we have enough unique individuals (performance optimization)
                    unique_count = len(set(seq for seq, _ in self.best_individuals))
                    
                    if unique_count >= 5:  # Only check if we have enough candidates
                        
                        # Use cached diversity if checked recently (performance optimization)
                        if generation == self.last_diversity_check_generation:
                            diversity_score = self.cached_diversity_score
                            min_diverse_fitness = self.cached_min_diverse_fitness
                        else:
                            # Get current diverse candidates (lightweight check)
                            current_diverse = self.get_diverse_top_sequences(num_sequences=5, min_diversity_threshold=0.15, verbose=False)
                            
                            if current_diverse and len(current_diverse) >= 3:  # Need at least 3 diverse sequences
                                diverse_sequences = [seq for seq, _ in current_diverse]
                                diverse_fitness = [fit for _, fit in current_diverse]
                                
                                min_diverse_fitness = min(diverse_fitness)
                                diversity_score = self.calculate_diversity(diverse_sequences)
                                
                                # Cache results
                                self.last_diversity_check_generation = generation
                                self.cached_diversity_score = diversity_score
                                self.cached_min_diverse_fitness = min_diverse_fitness
                            else:
                                diversity_score = 0.0
                                min_diverse_fitness = 0.0
                        
                        # Terminate if we have both high fitness AND good diversity
                        if (min_diverse_fitness >= 0.85 and diversity_score >= 0.20):
                            self.early_terminated = True
                            self.termination_reason = f"Early termination: High fitness + diverse solutions (diversity: {diversity_score:.3f}, min_fit: {min_diverse_fitness:.3f})"
                            print(f"\n{self.termination_reason}")
                            break
                        else:
                            # High fitness but low diversity - continue to improve diversity
                            print(f"Generation {generation}: High fitness achieved, but diversity too low ({diversity_score:.3f}). Continuing...")
                            # Reset streak to allow more generations for diversity improvement
                            self.high_fitness_streak = max(0, self.high_fitness_streak - 5)
                    else:
                        # Standard termination when we don't have enough unique individuals
                        self.early_terminated = True
                        self.termination_reason = f"Early termination: fitness >={self.early_termination_fitness} for {self.high_fitness_streak} generations"
                        print(f"\n{self.termination_reason}")
                        break
            else:
                self.high_fitness_streak = 0
            
            # === SEPARATED STAGNATION HANDLING ===
            
            # Handle FITNESS stagnation with mutation rate boost (only when fitness >0.75)
            if (self.fitness_stagnation_counter >= self.fitness_stagnation_threshold and
                self.mutation_boost_active == 0):  # Only if boost not already active
                
                # Apply mutation rate boost (only if fitness is high enough)
                boost_applied = self.apply_mutation_boost(real_max_fitness)
                if boost_applied:
                    self.fitness_stagnation_counter = 0  # Reset counter only if boost was applied
                else:
                    # If fitness too low, just reduce stagnation counter to prevent spam
                    self.fitness_stagnation_counter = max(0, self.fitness_stagnation_counter - 2)
            
            # Handle DIVERSITY stagnation with population restart
            if (self.diversity_stagnation_counter >= self.diversity_stagnation_threshold):               
                try:
                    self.restart_population(fitness_scores)
                    self.diversity_stagnation_counter = 0  # Reset counter
                    self.last_diversity = 1.0
                    
                    # Re-evaluate after restart
                    fitness_scores = self.evaluate_fitness(self.population)
                    real_max_fitness = max(fitness_scores)
                    max_fitness = real_max_fitness
                    avg_fitness = sum(fitness_scores) / len(fitness_scores)
                    self.last_real_fitness = real_max_fitness
                    
                    try:
                        diversity = self.calculate_diversity(self.population, sample_size)
                        self.last_diversity = diversity
                    except Exception as e:
                        print(f"Warning: Post-restart diversity calculation failed: {e}")
                        diversity = 0.0
                        
                except Exception as e:
                    print(f"Warning: Population restart failed: {e}, continuing with current population")
                    self.diversity_stagnation_counter = 0
            
            # Adaptive mutation rate adjustment
            self.mutation_rate = self.adaptive_mutation_rate(diversity)
            
            # Call external callbacks
            for callback in self.callbacks:
                try:
                    callback(generation, max_fitness, avg_fitness, diversity)
                except Exception as e:
                    print(f"Warning: Callback failed: {e}")
            
            # Store high-quality individuals (threshold set to capture good sequences)
            for i, (seq, fitness) in enumerate(zip(self.population, fitness_scores)):
                if fitness > 0.8:  # Increased from 0.7 to 0.8 for better quality
                    self.best_individuals.append((seq, fitness))
            
            # Create next generation with elite preservation + mutation and protected crowding
            new_population = []
            elite_count = max(1, int(self.population_size * self.elite_percentage))  # 1%-5% elitism
            
            # Select and preserve elites (keep originals + add mutated versions)
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
            
            # Always preserve the absolute best individual unchanged
            best_idx = elite_indices[0]
            new_population.append(self.population[best_idx])
            
            # Add mutated versions of remaining elites (no protection)
            for idx in elite_indices[1:]:
                mutated_elite = self.mutate(self.population[idx])
                new_population.append(mutated_elite)

            # Generating offspring
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
            
            # Triming offspring to exact size needed
            offspring = offspring[:self.population_size - elite_count]
            
            # Evaluate offspring fitness
            if offspring:
                offspring_fitness = self.evaluate_fitness(offspring)
                
                # Apply crowding selection with elite protection
                elite_indices_set = set(elite_indices)
                self.crowding_selection(fitness_scores, offspring, offspring_fitness, elite_indices_set)
            else:
                # Just keep current population if no offspring generated
                self.population = new_population[:self.population_size]
        
        if self.early_terminated:
            print(f"\nEvolution terminated early!")
            print(f"Reason: {self.termination_reason}")
        else:
            print("\nEvolution completed!")
        
        # Ensure we have final population in best_individuals for analysis
        if hasattr(self, 'population') and self.population:
            final_fitness = self.evaluate_fitness(self.population)
            for seq, fitness in zip(self.population, final_fitness):
                if fitness > 0.6:  # Lower threshold for final collection to ensure output
                    self.best_individuals.append((seq, fitness))
            
        # Saving cache for future runs
        self.save_cache()
        
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze final results and prepare output"""
        print("\n=== FINAL ANALYSIS ===")
        
        if self.early_terminated:
            print(f"TERMINATION: {self.termination_reason}")
        else:
            print("COMPLETED: Full generation count reached")
        
        # Get the final diverse sequences for output
        diverse_results = self.get_diverse_top_sequences(num_sequences=5, verbose=False)
        
        if diverse_results:
            print(f"SELECTED SEQUENCES: {len(diverse_results)} diverse solutions")
            sequences = [seq for seq, _ in diverse_results]
            fitness_scores = [fitness for _, fitness in diverse_results]
            diversity = self.calculate_diversity(sequences)
            
            print(f"Best fitness: {max(fitness_scores):.4f}")
            print(f"Avg fitness: {sum(fitness_scores)/len(fitness_scores):.4f}")
            print(f"Diversity: {diversity:.4f}")
            
            # Show the selected sequences
            for i, (seq, fitness) in enumerate(diverse_results, 1):
                print(f"{i}. {seq[:50]}... (fitness: {fitness:.4f})")
        else:
            # Emergency fallback: get best individuals from current population
            print("Warning: No diverse sequences found in best_individuals collection")
            print("Using current population for emergency output...")
            
            if hasattr(self, 'population') and self.population:
                # Evaluate current population and get best ones
                current_fitness = self.evaluate_fitness(self.population)
                population_with_fitness = list(zip(self.population, current_fitness))
                population_with_fitness.sort(key=lambda x: x[1], reverse=True)
                
                # Taking top 5 from current population
                diverse_results = population_with_fitness[:5]
                
                if not self.early_terminated:
                    print("NOTE: Due to completion at generation limit, optimal diversity may not have been achieved.")
                    print("These represent the best individuals found within the generation constraint.")
                
                sequences = [seq for seq, _ in diverse_results]
                fitness_scores = [fitness for _, fitness in diverse_results]
                
                print(f"EMERGENCY OUTPUT: {len(diverse_results)} best available sequences")
                print(f"Best fitness: {max(fitness_scores):.4f}")
                print(f"Avg fitness: {sum(fitness_scores)/len(fitness_scores):.4f}")
                
                if len(sequences) > 1:
                    diversity = self.calculate_diversity(sequences)
                    print(f"Diversity: {diversity:.4f}")
                
                # Show the selected sequences
                for i, (seq, fitness) in enumerate(diverse_results, 1):
                    print(f"{i}. {seq[:50]}... (fitness: {fitness:.4f})")
            else:
                print("CRITICAL ERROR: No population available for output!")
                diverse_results = []
        
        return diverse_results
    
    def save_results(self, filename=None, results_folder=None):
        """
        Save results to output file following assignment format
        
        Args:
            filename: Custom filename (optional)
            results_folder: Folder to save results (optional, will auto-detect from experiment)
        """
        # Get diverse top sequences with STRONG diversity enforcement
        diverse_results = self.get_diverse_top_sequences(num_sequences=10, min_diversity_threshold=0.15, verbose=False)
        
        if not diverse_results:
            print("No high-quality diverse results to save")
            return
        
        # Determine save location - prioritize CSV format for assignment
        if results_folder:
            # Save both txt and CSV formats to results folder
            txt_file = os.path.join(results_folder, "output.txt")
            csv_file = os.path.join(results_folder, "output.csv")
        elif filename:
            # Use provided filename
            txt_file = filename
            csv_file = filename.replace('.txt', '.csv') ## better to have both
        else:
            # Default to current directory
            txt_file = "assignment1_results.txt"
            csv_file = "assignment1_results.csv"
        
        # Ensure directory exists
        for output_file in [txt_file, csv_file]:
            if os.path.dirname(output_file):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Filter for high-quality sequences only
        high_quality_results = [(seq, fitness) for seq, fitness in diverse_results if fitness > 0.5]
        
        # Emergency fallback if no high-quality sequences
        if not high_quality_results and diverse_results:
            print("Warning: No sequences with fitness > 0.5, using best available sequences")
            high_quality_results = diverse_results[:5]  # Take top 5 regardless of fitness
        elif not high_quality_results:
            print("Critical: No sequences available for output")
            return
        
        # Save TXT format (one sequence per line)
        with open(txt_file, 'w') as f:
            for seq, fitness in high_quality_results:
                # Follow RNA convention: 5' to 3' direction (no 5' sign needed)
                f.write(f"{seq}\n")
        
        # Save CSV format (for assignment submission)
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['sequence', 'fitness'])
            # Write sequences
            for seq, fitness in high_quality_results:
                writer.writerow([seq, f"{fitness:.4f}"])
        
        print(f"\nResults saved to:")
        print(f"  TXT: {txt_file}")
        print(f"  CSV: {csv_file}")
        print(f"Saved {len(high_quality_results)} high-quality diverse sequences (fitness > 0.5)")
        
        if high_quality_results:
            print(f"Top sequence fitness: {high_quality_results[0][1]:.4f}")
            print(f"Fitness range: {high_quality_results[-1][1]:.4f} - {high_quality_results[0][1]:.4f}")
            
            # Calculate diversity of selected sequences
            sequences = [seq for seq, _ in high_quality_results]
            if len(sequences) > 1:
                diversity = self.calculate_diversity(sequences)
                print(f"Selected sequences diversity: {diversity:.4f}")
                
                # Check if diversity meets grading standards
                if diversity < 0.2:
                    print(f"WARNING: Low diversity ({diversity:.4f}) may impact grading!")
                    print(f"   Consider adjusting diversity parameters or running longer")
                elif diversity >= 0.3:
                    print(f"Excellent diversity ({diversity:.4f}) for grading!")
                else:
                    print(f"Goood diversity ({diversity:.4f}) for grading")
        
        return txt_file, csv_file
        with open(output_file, 'w') as f:
            for seq, fitness in diverse_results:
                if fitness > 0.5:  # Only save high-quality sequences
                    # Follow RNA convention: 5' to 3' direction (5' sign not needed)
                    f.write(f"{seq}\n")
                    high_quality_count += 1
        
        print(f"\nResults saved to {output_file}")
        print(f"Saved {high_quality_count} high-quality diverse sequences (fitness > 0.5)")
        
        if diverse_results:
            print(f"Top sequence fitness: {diverse_results[0][1]:.4f}")
            print(f"Fitness range: {diverse_results[-1][1]:.4f} - {diverse_results[0][1]:.4f}")
            
            # Calculate diversity of selected sequences
            sequences = [seq for seq, _ in diverse_results]
            if len(sequences) > 1:
                diversity = self.calculate_diversity(sequences)
                print(f"Selected sequences diversity: {diversity:.4f}")
        
        return output_file


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