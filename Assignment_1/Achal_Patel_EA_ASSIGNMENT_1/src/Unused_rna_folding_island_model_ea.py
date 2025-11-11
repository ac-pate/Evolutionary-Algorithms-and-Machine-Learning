# Achal Patel - 40227663
# Island Model Multi-Population EA Implementation

import random
import numpy as np
import subprocess
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json

class RNAFoldingIslandModelEA:
    """
    Evolutionary Algorithm for Inverse RNA Folding Problem - Island Model Implementation
    
    This is the multi-population island model with migration between sub-populations.
    Features 3 specialized sub-populations with different evolutionary strategies.
    """
    
    def __init__(self, population_size, generations, sequence_constraint, structure_constraint, max_workers=8, elite_percentage=0.01, enable_cache_preloading=False):
        """
        Initialize the Evolutionary Algorithm
        
        Args:
            population_size (int): Number of individuals in population
            generations (int): Number of generations to evolve
            sequence_constraint (str): IUPAC notation string
            structure_constraint (str): Dot-bracket notation string
            max_workers (int): Number of parallel workers for fitness evaluation
            elite_percentage (float): Percentage of population to keep as elites (default: 0.01 = 1%)
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
        self.mutation_rate = 0.08 if self.is_unknown_problem else 0.02  # adaptive mutation rate
        print(f"Mutation_rate= {self.mutation_rate}")
        self.tournament_size = 3
        
        # Increase elitism for .2 problems to preserve good solutions
        if self.is_unknown_problem:
            self.elite_percentage = max(elite_percentage, 0.05)  # At least 5% for .2 problems
            print(f"Elite percentage increased to {self.elite_percentage:.3f} for unknown sequence problem")
        else:
            self.elite_percentage = elite_percentage  # Keep default for .1 problems
        
        self.elitism_count = max(1, int(population_size * self.elite_percentage))
        print(f"Elite count: {self.elitism_count} individuals")
        
        
        # === TERMINATION SETTINGS ===
        self.early_termination_fitness = 0.98 if self.is_unknown_problem else 0.95
        self.termination_patience = 25 if self.is_unknown_problem else 15
        self.restart_threshold = 5  # When to apply population restart (same as old "shake")
        self.high_fitness_streak = 0
        self.last_restart_generation = -10
        
        # === MULTI-POPULATION SETUP ===
        self.sub_populations = []
        self.migration_interval = 10  # Migrate every 10 generations for active island model
        self.warmup_period = 5  # Shorter warmup for faster interaction
        self.setup_multi_population()
        
        # Fitness cache for optimization
        self.fitness_cache = {}
        
        # Load previous cache if enabled
        if self.enable_cache_preloading:
            self.preload_cache_from_previous_runs()
        
        # === STAGNATION TRACKING ===
        self.fitness_stagnation_threshold = 10
        self.diversity_stagnation_threshold = 15
        self.fitness_stagnation_counter = 0
        self.diversity_stagnation_counter = 0
        self.last_best_fitness = 0.0
        self.last_diversity = 1.0
        
        # === RESTART PARAMETERS ===
        self.diversity_threshold = 0.2
        self.restart_rate = 0.3 if self.is_unknown_problem else 0.4
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
        print("Initializing island model populations...")
        
        # Initialize sub-populations separately
        for sub_pop in self.sub_populations:
            population = []
            seen_sequences = set()
            target_size = sub_pop['size']
            
            attempts = 0
            max_attempts = target_size * 10
            
            while len(population) < target_size and attempts < max_attempts:
                sequence = self.generate_random_sequence()
                if sequence not in seen_sequences and self.is_valid_sequence(sequence):
                    population.append(sequence)
                    seen_sequences.add(sequence)
                attempts += 1
                
            while len(population) < target_size:
                sequence = self.generate_random_sequence()
                population.append(sequence)
                
            sub_pop['population'] = population
            print(f"Initialized {sub_pop['name']} sub-population with {len(population)} individuals")
        
        # Also maintain combined population for compatibility
        self.population = []
        for sub_pop in self.sub_populations:
            self.population.extend(sub_pop['population'])
    
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
            copy_cmd = ["sudo", "docker", "cp", batch_file, f"ipknot_runner:/work/{base_name}"]
            subprocess.run(copy_cmd, check=True, capture_output=True)
            
            cmd = ["sudo", "docker", "exec", "ipknot_runner", "ipknot", f"/work/{base_name}"]
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
            print(f"Batch folding failed: {e}")
            if 'batch_file' in locals() and os.path.exists(batch_file):
                os.unlink(batch_file)
            return ['.' * len(seq) for seq in sequences]
    
    def _fold_single_sequence(self, sequence):
        """Original single sequence folding logic"""
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
        raw_fitness = [cached_results[i] for i in range(len(sequences))]
        
        # Apply fitness landscape adaptations
        return self.apply_fitness_landscape_adaptations(sequences, raw_fitness)
    
    def apply_fitness_landscape_adaptations(self, sequences, raw_fitness):
        """
        Apply problem-specific fitness adaptations:
        - .2 problems (unknown): Fitness sharing to maintain diversity
        - .1 problems (known): Diversity bonus to prevent premature convergence
        """
        if not sequences:
            return raw_fitness
        
        adapted_fitness = raw_fitness.copy()
        
        if self.is_unknown_problem:
            # For .2 problems: Apply fitness sharing
            adapted_fitness = self.apply_fitness_sharing(sequences, adapted_fitness)
        else:
            # For .1 problems: Add diversity bonus
            adapted_fitness = self.apply_diversity_bonus(sequences, adapted_fitness)
        
        return adapted_fitness
    
    def apply_fitness_sharing(self, sequences, fitness_scores):
        """
        Fitness sharing for unknown sequence problems (.2)
        Reduces fitness of similar individuals to maintain diversity
        """
        if len(sequences) <= 1:
            return fitness_scores
        
        shared_fitness = []
        
        for i, seq1 in enumerate(sequences):
            niche_count = 0.0
            
            for j, seq2 in enumerate(sequences):
                if i != j:
                    # Calculate sequence similarity
                    similarity = self.calculate_sequence_similarity(seq1, seq2)
                    
                    # Sharing function: linear decrease from 1 to 0
                    sigma_share = 0.3  # Sharing radius (30% similarity threshold)
                    if similarity < sigma_share:
                        sharing_value = 1.0 - (similarity / sigma_share)
                    else:
                        sharing_value = 0.0
                    
                    niche_count += sharing_value
            
            # Add self (niche_count starts from 1)
            niche_count += 1.0
            
            # Reduce fitness based on niche density
            shared_fitness.append(fitness_scores[i] / niche_count)
        
        return shared_fitness
    
    def apply_diversity_bonus(self, sequences, fitness_scores):
        """
        Diversity bonus for known sequence problems (.1)
        Rewards individuals that are different from the population average
        """
        if len(sequences) <= 1:
            return fitness_scores
        
        # Calculate average sequence characteristics
        avg_gc_content = sum(self.calculate_gc_content(seq) for seq in sequences) / len(sequences)
        
        bonused_fitness = []
        
        for i, seq in enumerate(sequences):
            base_fitness = fitness_scores[i]
            
            # Diversity bonus based on GC content deviation
            gc_content = self.calculate_gc_content(seq)
            gc_diversity = abs(gc_content - avg_gc_content)
            
            # Bonus for moderate deviation (not too extreme)
            diversity_bonus = min(0.05, gc_diversity * 0.1)  # Max 5% bonus
            
            # Only apply bonus if base fitness is reasonable (>0.3)
            if base_fitness > 0.3:
                final_fitness = base_fitness + diversity_bonus
            else:
                final_fitness = base_fitness
            
            bonused_fitness.append(min(1.0, final_fitness))  # Cap at 1.0
        
        return bonused_fitness
    
    def calculate_sequence_similarity(self, seq1, seq2):
        """Calculate similarity between two sequences (0 = identical, 1 = completely different)"""
        if len(seq1) != len(seq2):
            return 1.0
        
        differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
        return differences / len(seq1)
    
    def calculate_gc_content(self, sequence):
        """Calculate GC content of a sequence"""
        gc_count = sum(1 for base in sequence if base in ['G', 'C'])
        return gc_count / len(sequence) if sequence else 0.0
    
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
    
    def preload_cache_from_previous_runs(self):
        """Load fitness cache from previous experiments (if enabled)"""
        cache_file = "cache/fitness_cache.json"
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    self.fitness_cache.update(cached_data)
                    print(f"[CACHE] Loaded {len(cached_data)} cached fitness evaluations")
            else:
                print("[CACHE] No previous cache found, starting fresh")
        except Exception as e:
            print(f"[CACHE] Warning: Failed to load cache - {e}")
            
    def save_cache(self):
        """Save fitness cache for future runs (if cache preloading enabled)"""
        if not self.enable_cache_preloading:
            return
            
        try:
            os.makedirs("cache", exist_ok=True)
            cache_file = "cache/fitness_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(self.fitness_cache, f)
            print(f"[CACHE] Saved {len(self.fitness_cache)} fitness evaluations to cache")
        except Exception as e:
            print(f"[CACHE] Warning: Failed to save cache - {e}")
    
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
    
    def restart_sub_population(self, sub_pop, fitness_scores):
        """Apply population restart to a specific sub-population"""
        print(f"  Restarting {sub_pop['name']} sub-population")
        
        population = sub_pop['population']
        population_with_fitness = list(zip(population, fitness_scores))
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        elite_count = max(1, int(len(population) * 0.1))  # Top 10%
        mutate_count = int(len(population) * 0.4)         # 40% heavy mutation
        restart_count = len(population) - elite_count - mutate_count  # 50% restart
        
        new_population = []
        
        # Keep top elites unchanged
        for i in range(elite_count):
            new_population.append(population_with_fitness[i][0])
        
        # Heavy mutation on good individuals
        original_mutation_rate = sub_pop['mutation_rate']
        temp_mutation_rate = min(0.3, original_mutation_rate * 8)  # 8x mutation boost
        
        for i in range(elite_count, min(elite_count + mutate_count, len(population_with_fitness))):
            individual = population_with_fitness[i][0]
            # Apply multiple rounds of mutation with sub-pop specific rate
            old_mutation = self.mutation_rate
            self.mutation_rate = temp_mutation_rate
            for _ in range(2):  # Double mutation
                individual = self.mutate(individual)
            self.mutation_rate = old_mutation
            new_population.append(individual)
        
        # Fill remaining with random individuals
        while len(new_population) < len(population):
            new_population.append(self.generate_random_sequence())
        
        sub_pop['population'] = new_population
    
    def setup_multi_population(self):
        """
        Setup 3 sub-populations for island model with different strategies:
        1. Explorative: High mutation, small tournament
        2. Balanced: Medium mutation, medium tournament  
        3. Exploitative: Low mutation, large tournament
        """
        sub_pop_size = self.population_size // 3
        remainder = self.population_size % 3
        
        self.sub_populations = [
            {
                'name': 'Explorative',
                'size': sub_pop_size + (1 if remainder > 0 else 0),
                'mutation_rate': 0.25,  # Very high mutation for exploration
                'tournament_size': 2,
                'population': [],
                'best_fitness': 0.0,
                'last_best_fitness': 0.0,
                'last_diversity': 1.0,
                'fitness_stagnation_counter': 0,
                'diversity_stagnation_counter': 0
            },
            {
                'name': 'Balanced', 
                'size': sub_pop_size + (1 if remainder > 1 else 0),
                'mutation_rate': 0.125,  # Medium mutation for balance
                'tournament_size': 3,
                'population': [],
                'best_fitness': 0.0,
                'last_best_fitness': 0.0,
                'last_diversity': 1.0,
                'fitness_stagnation_counter': 0,
                'diversity_stagnation_counter': 0
            },
            {
                'name': 'Exploitative',
                'size': sub_pop_size,
                'mutation_rate': 1.0 / 24,  # Low mutation, precise optimization
                'tournament_size': 5,
                'population': [],
                'best_fitness': 0.0,
                'last_best_fitness': 0.0,
                'last_diversity': 1.0,
                'fitness_stagnation_counter': 0,
                'diversity_stagnation_counter': 0
            }
        ]
        
        print(f"\nIsland model setup:")
        for i, sub_pop in enumerate(self.sub_populations):
            print(f"  Island {i+1} ({sub_pop['name']}): {sub_pop['size']} individuals, "
                  f"mutation={sub_pop['mutation_rate']:.4f}, tournament={sub_pop['tournament_size']}")
    
    def migrate_individuals(self, generation):
        """
        Migrate best individuals between sub-populations
        Only starts after warmup period to allow populations to establish
        """
        if (generation < self.warmup_period or 
            generation % self.migration_interval != 0):
            return
        
        print(f"\n*** MIGRATION EVENT (Gen {generation}) ***")
        
        # Collect best individuals from each sub-population
        migrants = []
        for i, sub_pop in enumerate(self.sub_populations):
            if sub_pop['population']:
                fitness_scores = self.evaluate_fitness(sub_pop['population'])
                best_idx = max(range(len(fitness_scores)), key=lambda x: fitness_scores[x])
                migrants.append({
                    'individual': sub_pop['population'][best_idx],
                    'fitness': fitness_scores[best_idx],
                    'source': i
                })
                sub_pop['best_fitness'] = max(fitness_scores)
        
        # Migrate in circular fashion: 0->1, 1->2, 2->0
        for i in range(len(self.sub_populations)):
            target_pop = (i + 1) % len(self.sub_populations)
            if i < len(migrants) and self.sub_populations[target_pop]['population']:
                target_fitness = self.evaluate_fitness(self.sub_populations[target_pop]['population'])
                worst_idx = min(range(len(target_fitness)), key=lambda x: target_fitness[x])
                
                # Only migrate if migrant is better than worst in target
                if migrants[i]['fitness'] > target_fitness[worst_idx]:
                    self.sub_populations[target_pop]['population'][worst_idx] = migrants[i]['individual']
                    print(f"  Migrated from {self.sub_populations[i]['name']} to {self.sub_populations[target_pop]['name']} "
                          f"(fitness: {migrants[i]['fitness']:.4f})")
    
    def evolve_sub_population(self, sub_pop_idx, generation):
        """
        Evolve a single sub-population with its specific parameters
        """
        sub_pop = self.sub_populations[sub_pop_idx]
        population = sub_pop['population']
        
        if not population:
            return []
        
        # Store original parameters
        original_mutation = self.mutation_rate
        original_tournament = self.tournament_size
        
        # Set sub-population specific parameters
        self.mutation_rate = sub_pop['mutation_rate']
        self.tournament_size = sub_pop['tournament_size']
        
        # Evaluate fitness
        fitness_scores = self.evaluate_fitness(population)
        
        # Create new generation
        elite_count = max(1, int(len(population) * 0.10))  # 10% elitism per sub-pop
        new_population = []
        
        # Select elites
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
        for idx in elite_indices:
            new_population.append(population[idx])
        
        # Generate offspring
        while len(new_population) < len(population):
            if random.random() < self.crossover_rate:
                parent1 = self.tournament_selection_from_population(population, fitness_scores)
                parent2 = self.tournament_selection_from_population(population, fitness_scores)
                
                if len(parent1) > 100:
                    child1, child2 = self.multi_point_crossover(parent1, parent2)
                else:
                    child1, child2 = self.two_point_crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            else:
                parent = self.tournament_selection_from_population(population, fitness_scores)
                child = self.mutate(parent)
                new_population.append(child)
        
        # Restore original parameters
        self.mutation_rate = original_mutation
        self.tournament_size = original_tournament
        
        return new_population[:len(population)]
    
    def tournament_selection_from_population(self, population, fitness_scores):
        """Tournament selection from a specific population"""
        try:
            actual_tournament_size = min(self.tournament_size, len(population))
            tournament_indices = random.sample(range(len(population)), actual_tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            return population[winner_index]
        except Exception as e:
            print(f"Warning: Sub-population tournament selection failed: {e}")
            return random.choice(population)
    
    def handle_sub_population_stagnation(self, generation, sample_size):
        """Handle stagnation for each sub-population independently"""
        start_idx = 0
        population_changed = False
        
        for i, sub_pop in enumerate(self.sub_populations):
            sub_size = len(sub_pop['population'])
            if sub_size == 0:
                continue
                
            # Get fitness and diversity for this sub-population
            sub_fitness = self.evaluate_fitness(sub_pop['population'])
            sub_best_fitness = max(sub_fitness) if sub_fitness else 0.0
            sub_avg_fitness = sum(sub_fitness) / len(sub_fitness) if sub_fitness else 0.0
            
            try:
                sub_diversity = self.calculate_diversity(sub_pop['population'], min(sample_size, sub_size) if sample_size else None)
            except Exception:
                sub_diversity = 0.0
            
            # Track stagnation for this sub-population
            if sub_best_fitness <= sub_pop['last_best_fitness'] + 1e-6:
                sub_pop['fitness_stagnation_counter'] += 1
            else:
                sub_pop['fitness_stagnation_counter'] = 0
                sub_pop['last_best_fitness'] = sub_best_fitness
                
            if sub_diversity <= sub_pop['last_diversity'] + 1e-6:
                sub_pop['diversity_stagnation_counter'] += 1
            else:
                sub_pop['diversity_stagnation_counter'] = 0
                sub_pop['last_diversity'] = sub_diversity
            
            # Check if this sub-population needs restart
            should_restart_sub_pop = (
                (sub_best_fitness < self.early_termination_fitness) and
                (sub_pop['fitness_stagnation_counter'] >= self.fitness_stagnation_threshold or 
                 sub_pop['diversity_stagnation_counter'] >= self.diversity_stagnation_threshold)
            )
            
            if should_restart_sub_pop:
                print(f"\n*** SUB-POPULATION RESTART (Gen {generation}) ***")
                print(f"  Restarting {sub_pop['name']} sub-population (f_stag: {sub_pop['fitness_stagnation_counter']}, d_stag: {sub_pop['diversity_stagnation_counter']})")
                
                # Restart only this sub-population
                self.restart_sub_population(sub_pop, sub_fitness)
                sub_pop['fitness_stagnation_counter'] = 0
                sub_pop['diversity_stagnation_counter'] = 0
                sub_pop['last_best_fitness'] = 0.0
                sub_pop['last_diversity'] = 1.0
                population_changed = True
                
            # Update combined population with potentially restarted sub-population
            end_idx = start_idx + sub_size
            self.population[start_idx:end_idx] = sub_pop['population']
            start_idx = end_idx
        
        return population_changed
    
    def get_best_individuals(self):
        """Get best individuals as list of dictionaries"""
        return [{'sequence': seq, 'fitness': fitness} for seq, fitness in self.best_individuals]
    
    def run_evolution(self):
        """Main evolutionary algorithm loop - Island Model"""
        print(f"Starting Island Model evolution: {self.population_size} individuals, {self.generations} generations")
        print(f"Using {self.max_workers} parallel workers for fitness evaluation")
        print(f"Sequence constraint: {self.sequence_constraint}")
        print(f"Structure constraint: {self.structure_constraint}")
        
        # Show optimization features being used
        print(f"\n*** ISLAND MODEL OPTIMIZATIONS ACTIVE ***")
        print(f"🔹 3-Island architecture with specialized sub-populations")
        print(f"🔹 Migration every {self.migration_interval} generations")
        print(f"🔹 Sub-population specific stagnation handling")
        print(f"🔹 Fitness sharing for diversity maintenance")
        print(f"🔹 Adaptive termination: patience={self.termination_patience}, threshold={self.early_termination_fitness}")
        print()
        
        self.initialize_population()
        
        for generation in range(self.generations):
            self.current_generation = generation
            
            # === MULTI-POPULATION EVOLUTION ===
            # Evolve each sub-population
            for i, sub_pop in enumerate(self.sub_populations):
                new_sub_pop = self.evolve_sub_population(i, generation)
                sub_pop['population'] = new_sub_pop
            
            # Migration between sub-populations (after warmup)
            self.migrate_individuals(generation)
            
            # Update combined population for statistics
            self.population = []
            for sub_pop in self.sub_populations:
                self.population.extend(sub_pop['population'])
            
            # === FITNESS EVALUATION ===
            fitness_scores = self.evaluate_fitness(self.population)
            
            # === STATISTICS TRACKING ===
            max_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.fitness_history.append((generation, max_fitness, avg_fitness))
            
            # Calculate diversity
            try:
                sample_size = 100 if len(self.population) > 200 else None
                diversity = self.calculate_diversity(self.population, sample_size)
            except Exception as e:
                print(f"Warning: Diversity calculation failed: {e}, using fallback")
                diversity = 0.0
            
            # === SUB-POPULATION STAGNATION HANDLING ===
            self.handle_sub_population_stagnation(generation, sample_size)
            
            # === HIGH FITNESS RESTART MECHANISM ===
            if max_fitness >= self.early_termination_fitness:
                self.high_fitness_streak += 1
                
                # Apply restart when stuck in high fitness plateau
                if (self.high_fitness_streak >= self.restart_threshold and 
                    self.high_fitness_streak < self.termination_patience and
                    generation - self.last_restart_generation > 10):
                    
                    print(f"\nTRIGGERING RESTART (streak: {self.high_fitness_streak})")
                    
                    # Restart all sub-populations
                    for sub_pop in self.sub_populations:
                        sub_fitness = self.evaluate_fitness(sub_pop['population'])
                        self.restart_sub_population(sub_pop, sub_fitness)
                    # Update combined population
                    self.population = []
                    for sub_pop in self.sub_populations:
                        self.population.extend(sub_pop['population'])
                    
                    self.last_restart_generation = generation
                    
                    # Reset counters but keep the streak
                    self.fitness_stagnation_counter = 0
                    self.diversity_stagnation_counter = 0
                    
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
                
                # === FINAL TERMINATION CHECK ===
                if self.high_fitness_streak >= self.termination_patience:
                    # Additional check for diversity
                    if diversity < 0.15:
                        print(f"\nDelaying termination: diversity too low ({diversity:.3f})")
                        # Apply additional restart instead of terminating
                        if generation - self.last_restart_generation > 5:
                            for sub_pop in self.sub_populations:
                                sub_fitness = self.evaluate_fitness(sub_pop['population'])
                                self.restart_sub_population(sub_pop, sub_fitness)
                            self.population = []
                            for sub_pop in self.sub_populations:
                                self.population.extend(sub_pop['population'])
                            self.last_restart_generation = generation
                            self.high_fitness_streak = max(0, self.high_fitness_streak - 3)  # Reset partially
                    else:
                        self.early_terminated = True
                        termination_reason = f"fitness >={self.early_termination_fitness} for {self.high_fitness_streak} generations, diversity maintained ({diversity:.3f})"
                        self.termination_reason = f"Adaptive termination: {termination_reason}"
                        print(f"\n{self.termination_reason}")
                        break
            else:
                self.high_fitness_streak = 0
            
            # === CALLBACKS AND STORAGE ===
            for callback in self.callbacks:
                try:
                    callback(generation, max_fitness, avg_fitness, diversity)
                except Exception as e:
                    print(f"Warning: Callback failed: {e}")
            
            # Show sub-population details every 10 generations
            if (generation + 1) % 10 == 0:
                print("  Sub-population details:")
                start_idx = 0
                for i, sub_pop in enumerate(self.sub_populations):
                    sub_size = len(sub_pop['population'])
                    if sub_size > 0:
                        sub_fitness = fitness_scores[start_idx:start_idx + sub_size]
                        sub_best = max(sub_fitness) if sub_fitness else 0.0
                        sub_avg = sum(sub_fitness) / len(sub_fitness) if sub_fitness else 0.0
                        print(f"    {sub_pop['name']}: Best={sub_best:.4f}, Avg={sub_avg:.4f}, "
                              f"Size={sub_size}, Mut={sub_pop['mutation_rate']:.3f}")
                        start_idx += sub_size
            
            # Store high-quality individuals
            for i, (seq, fitness) in enumerate(zip(self.population, fitness_scores)):
                if fitness > 0.9:
                    self.best_individuals.append((seq, fitness))
        
        if self.early_terminated:
            print(f"\nEvolution terminated early!")
            print(f"Reason: {self.termination_reason}")
        else:
            print("\nEvolution completed!")
            
        # Save cache for future runs
        self.save_cache()
        
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
        print(f"Elite percentage: {self.elite_percentage:.1%} ({self.elitism_count} individuals)")
        
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
    
    def save_results(self, filename="assignment1_island_results.txt"):
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
    Main function for Island Model EA
    """
    print("RNA Folding Island Model Evolutionary Algorithm")
    print("===============================================")
    
    # Get user input as required
    try:
        population_size = int(input("Please enter the population size: "))
        num_generations = int(input("Please enter the number of generations: "))
    except ValueError:
        print("Invalid input. Using default values.")
        population_size = 300
        num_generations = 150
    
    # Load constraints from CSV (using the first constraint set for testing)
    sequence_constraint = "GARMUWYMNKKSSGMUCCKCYAGCNCMMNGAGKNCWAUKSKRUNCGNMYCNMNSCKCNCNCKUKKSWSAACSSSAMCN"
    structure_constraint = "((((....))))....(((((.......)))))............(((((.......[[[[[[))))).]]]]]]..."
    
    print("Running Island Model EA...")
    
    # Create and run EA
    ea = RNAFoldingIslandModelEA(population_size, num_generations, sequence_constraint, structure_constraint, max_workers=8)
    
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