# COEN 432/6321 Midterm 1 - Professor's Review Guide Practice Questions

## Based on Professor's Guidelines for Midterm 1

---

## 1. Genotype and Phenotype

### Question 1 (E): Define genotype and phenotype with an example.
**Answer:** 
- **Genotype**: The encoded representation used by the EA (chromosome/solution encoding)
- **Phenotype**: The actual solution that the genotype represents when decoded
- **Example**: For optimizing neural network weights:
  - Genotype: [0.5, -0.3, 0.8, 0.1] (encoded weights)
  - Phenotype: The actual neural network with these weights

### Question 2 (G): In RNA folding problem, what would be appropriate genotype and phenotype representations?
**Answer:**
- **Genotype**: String of nucleotides "AUCGAUCG" 
- **Phenotype**: The actual 3D folded structure of the RNA molecule
- **Fitness**: How closely the phenotype matches target secondary structure

---

## 2. Crossover and Mutation Applications

### Question 3 (E): Where are crossover and mutation applied in the EA cycle?
**Answer:**
- **Applied to**: Selected parents to create offspring
- **Timing**: After parent selection, before fitness evaluation of offspring
- **Purpose**: 
  - Crossover: Combine genetic material from parents
  - Mutation: Introduce random variations

### Question 4 (G): Why can't we use standard single-point crossover for permutation problems like TSP?
**Answer:**
- **Problem**: Creates invalid offspring with repeated/missing cities
- **Example**: 
  - Parent1: [1,2,3,4,5], Parent2: [3,1,5,2,4]
  - Single-point at position 2: [1,2|5,2,4] → Invalid (city 2 repeated, city 3 missing)
- **Solution**: Use specialized operators like Order Crossover (OX) or PMX

---

## 3. Initialization

### Question 5 (E): How is population initialization generally done?
**Answer:**
- **Method**: Random generation
- **Process**: Create diverse set of random valid solutions
- **Constraints**: Ensure all individuals satisfy problem constraints
- **Goal**: Maximum initial diversity to explore entire search space

### Question 6 (G): What are the challenges of random initialization for constrained problems?
**Answer:**
- **Constraint satisfaction**: Must generate only feasible solutions
- **Repair mechanisms**: Fix invalid solutions after generation
- **Biased sampling**: Some regions may be over/under-represented
- **Alternative**: Heuristic seeding with some good solutions

---

## 4. Parents and Offspring Generation

### Question 7 (E): What are parents in evolutionary algorithms?
**Answer:**
- **Definition**: Individuals chosen from current population for reproduction
- **Selection**: Based on fitness (better individuals more likely chosen)
- **Purpose**: Pass genetic material to next generation through crossover and mutation
- **Number**: Typically 2 for crossover, 1 for mutation-only

### Question 8 (G): How do offspring generation methods affect search behavior?
**Answer:**
- **Crossover-heavy**: Exploits existing genetic material, faster convergence
- **Mutation-heavy**: More exploration, better for rugged landscapes
- **Balance**: Usually 70-90% crossover, 1-5% mutation per gene
- **Adaptive**: Adjust ratios based on search progress

---

## 5. Fitness Function

### Question 9 (E): What is a fitness function and what does it return?
**Answer:**
- **Definition**: Function that evaluates quality of a solution
- **Returns**: Single number (single-objective) or multiple numbers (multi-objective)
- **Purpose**: Guide search toward better solutions
- **Design**: Must reflect true problem objectives

### Question 10 (G): How do you design fitness function for multi-objective problems?
**Answer:**
- **Approaches**:
  1. **Weighted sum**: f = w₁f₁ + w₂f₂ + ... + wₙfₙ
  2. **Pareto dominance**: Compare solutions across all objectives
  3. **Constraint handling**: Penalties for constraint violations
- **Example TSP**: Minimize distance AND minimize time
- **Challenge**: Weight selection affects solution bias

---

## 6. Diversity

### Question 11 (G): Why is diversity important in evolutionary algorithms?
**Answer:**
- **Exploration**: Maintains ability to discover new solution regions
- **Premature convergence**: Prevents getting stuck in local optima
- **Genetic material**: Preserves building blocks for future recombination
- **Measurement**: Hamming distance, genotype/phenotype variance
- **Maintenance**: Fitness sharing, crowding, speciation

### Question 12 (G): How do you measure diversity quantitatively?
**Answer:**
- **Hamming distance**: For binary/discrete: avg(Σᵢ≠ⱼ hamming(i,j))/(n(n-1)/2)
- **Euclidean distance**: For real-valued: avg(Σᵢ≠ⱼ ||xᵢ-xⱼ||)/(n(n-1)/2)
- **Entropy**: H = -Σₖ pₖ log(pₖ) where pₖ is proportion with allele k
- **Variance**: σ² = E[(X-μ)²] for each gene position

---

## 7. Selection Applications

### Question 13 (E): Where does selection end and where do we apply selection?
**Answer:**
- **Selection is applied**: 
  - **Parent selection**: Before reproduction (choosing parents)
  - **Survivor selection**: After reproduction (choosing next generation)
- **Selection ends**: Never - it's applied throughout the entire EA process
- **Cannot avoid**: Selection is fundamental to evolution - must always choose which individuals reproduce/survive

### Question 14 (G): What are the different types of selection and their applications?
**Answer:**
1. **Parent Selection**: Choose individuals for reproduction
   - Tournament, roulette wheel, rank-based
2. **Survivor Selection**: Choose next generation 
   - Generational, steady-state, elitist
3. **Environmental Selection**: Multi-objective scenarios
   - NSGA-II, SPEA-2

---

## 8. Selection Mechanisms Deep Dive

### Question 15 (G): How does window size and replacement affect tournament selection?
**Answer:**
- **Window Size (k)**:
  - k=1: Random selection (no pressure)
  - k=2: Moderate pressure  
  - k→n: Strong pressure (near-deterministic)
- **With Replacement**: Individual can appear multiple times in tournament
  - Higher variance, some individuals may escape selection
- **Without Replacement**: Each individual appears at most once
  - Lower variance, more uniform sampling

### Question 16 (E): Explain fitness proportional selection mechanism.
**Answer:**
- **Process**: 
  1. Calculate total fitness: F = Σfᵢ
  2. Selection probability: P(i) = fᵢ/F
  3. Use roulette wheel or stochastic universal sampling
- **Issues**: Requires positive fitness, sensitive to scaling
- **Advantage**: Natural selection pressure proportional to fitness

### Question 17 (G): How does tournament selection differ from fitness proportional selection?
**Answer:**
| Aspect | Tournament | Fitness Proportional |
|--------|------------|-------------------|
| **Pressure control** | Tournament size | Fitness distribution |
| **Scaling requirement** | None | Often needed |
| **Negative fitness** | Handles naturally | Requires adjustment |
| **Computational cost** | O(k) | O(n) for roulette |
| **Parameter sensitivity** | Robust | Sensitive |

---

## 9. Representation-Specific Operations

### Question 18 (G): What defines operations for 1 vs 2 individuals?
**Answer:**
- **1 Individual (Mutation)**: 
  - Unary operator affecting single chromosome
  - Maintains population size
  - Introduces variation
- **2 Individuals (Crossover)**:
  - Binary operator combining two parents  
  - May produce 1 or 2 offspring
  - Exploits existing genetic material

### Question 19 (E): For binary representation, what are the main crossover types?
**Answer:**
1. **Single-point**: Cut at one position, swap tails
2. **Two-point**: Cut at two positions, swap middle segment  
3. **Uniform**: Each bit independently chooses parent
4. **N-point**: Multiple cut points

### Question 20 (G): Can crossover alone search the entire binary space?
**Answer:**
- **No**: Crossover can only recombine existing alleles
- **Missing bits**: If no individual has '1' at position i, crossover cannot create it
- **Schema limitation**: Can only explore schemas present in initial population
- **Mutation necessity**: Required for complete space coverage

---

## 10. Traveling Salesman Problem

### Question 21 (E): Why is TSP a permutation problem?
**Answer:**
- **Constraint**: Each city visited exactly once
- **Representation**: Order matters (defines tour)
- **Size**: n! possible tours for n cities
- **Structure**: Cyclic permutation (1-2-3-4-1 same as 2-3-4-1-2)

### Question 22 (G): What crossover operators work for TSP and how?
**Answer:**
1. **Order Crossover (OX)**:
   - Copy segment from parent1, fill rest with parent2's order
2. **Partially Matched Crossover (PMX)**:
   - Establish mapping, resolve conflicts systematically
3. **Cycle Crossover (CX)**:
   - Preserve position-value relationships
4. **Edge Recombination**:
   - Preserve edge information from parents

### Question 23 (G): What mutation operators are suitable for TSP?
**Answer:**
1. **Swap mutation**: Exchange two random cities
2. **Insert mutation**: Remove city, reinsert elsewhere
3. **Inversion mutation**: Reverse subsequence
4. **Scramble mutation**: Randomly permute subsequence
5. **2-opt**: Remove two edges, reconnect differently

---

## 11. Integer and Floating Point Representations

### Question 24 (E): What are the characteristics of integer representation?
**Answer:**
- **Domain**: Discrete integer values
- **Crossover**: Arithmetic, uniform, multi-point
- **Mutation**: Gaussian noise, uniform random replacement
- **Constraints**: Range limits, step sizes
- **Applications**: Discrete optimization, combinatorial problems

### Question 25 (G): How do floating-point crossover operators work?
**Answer:**
1. **Arithmetic Crossover**: 
   - Child1 = α·Parent1 + (1-α)·Parent2
   - Child2 = (1-α)·Parent1 + α·Parent2
2. **BLX-α (Blend Crossover)**:
   - Range: [min(p1,p2)-α·d, max(p1,p2)+α·d]
   - α=0.5 typical, extends parent range
3. **SBX (Simulated Binary Crossover)**:
   - Mimics binary crossover behavior for real values

### Question 26 (G): What mutation strategies exist for floating-point?
**Answer:**
- **Fixed step size**: x' = x + N(0,σ)
- **Deterministic**: σ(t) = σ₀(1-t/T) 
- **Adaptive**: Adjust σ based on success rate (1/5 rule)
- **Self-adaptive**: σ evolves with solution
- **Correlated**: Full covariance matrix evolution

---

## 12. Advanced Selection and Replacement

### Question 27 (G): What is generational vs steady-state replacement?
**Answer:**
- **Generational**: Replace entire population each cycle
  - Clear generations, parallel evaluation
  - May lose good solutions temporarily
- **Steady-State**: Replace few individuals continuously  
  - Better preservation, overlapping generations
  - More complex bookkeeping

### Question 28 (E): What is generation gap?
**Answer:**
- **Definition**: Fraction of population replaced per generation
- **Range**: 0.0 to 1.0
- **Effect**: Controls replacement pressure
- **Low gap**: Conservative, preserves solutions
- **High gap**: Aggressive, faster turnover

### Question 29 (G): How do scaling methods affect fitness?
**Answer:**
1. **Windowing**: f' = f - min(f_pop)
   - Eliminates negative values
   - Amplifies relative differences
2. **Sigma Scaling**: f' = max(0, f - (μ - c·σ))
   - Maintains constant pressure
   - c=2.0 typical
3. **Rank-based**: Use rank instead of raw fitness
   - Position-based assignment
   - Immune to scaling issues

---

## 13. Multi-objective Concepts

### Question 30 (G): What is domination in multi-objective optimization?
**Answer:**
- **Definition**: Solution A dominates B if:
  - A ≥ B in all objectives AND
  - A > B in at least one objective
- **Non-dominated**: No other solution dominates it
- **Pareto optimal**: Non-dominated in entire space
- **Example**: (cost=10, time=5) dominates (cost=12, time=7)

### Question 31 (G): Explain design space vs objective space.
**Answer:**
- **Design Space**: Space of decision variables (where solutions exist)
  - Example: All possible TSP tours
  - Searched by EA operators
- **Objective Space**: Space of objective function values  
  - Example: (distance, time) pairs
  - Where we evaluate and compare solutions
- **Mapping**: Each design point maps to one objective point

### Question 32 (G): What is the relationship between convergence and diversity?
**Answer:**
- **Convergence**: Solutions approach true Pareto front
- **Diversity**: Solutions spread along Pareto front
- **Trade-off**: Focusing on one may hurt the other
- **Ideal**: Good convergence AND good diversity
- **Algorithms**: NSGA-II balances both with crowding distance

---

## 14. Parameter Control

### Question 33 (E): What is parameter tuning vs parameter control?
**Answer:**
- **Tuning**: Set parameters before run (static)
  - Based on preliminary experiments
  - Fixed throughout evolution
- **Control**: Adjust parameters during run (dynamic)
  - Responds to search progress
  - Three types: deterministic, adaptive, self-adaptive

### Question 34 (G): How do deterministic, adaptive, and self-adaptive control differ?
**Answer:**
1. **Deterministic**: Predetermined schedule
   - Example: σ(t) = σ₀(1-t/T)
   - Simple, predictable
2. **Adaptive**: Based on search feedback
   - Example: Increase mutation if no improvement
   - Responsive to search state  
3. **Self-adaptive**: Parameters part of chromosome
   - Example: (x₁,...,xₙ,σ₁,...,σₙ)
   - Evolves automatically

---

## 15. Problem-Specific Knowledge

### Question 35 (G): For RNA folding, what representation would you use and why?
**Answer:**
- **Representation**: String of nucleotides [A,U,C,G]
- **Constraints**: IUPAC codes limit valid nucleotides per position
- **Fitness**: Structural similarity to target (dot-bracket notation)
- **Operators**: Point mutation (change nucleotide), specialized crossover

### Question 36 (G): How would you evaluate diversity in TSP solutions?
**Answer:**
- **Edge diversity**: Count unique edges across population
- **Position diversity**: Hamming distance between permutations  
- **Tour shape**: Geometric/topological similarity measures
- **Objective diversity**: Variance in tour lengths

---

## 17. Algorithm Taxonomy and Models

### Question 37 (G): What does the "green chart" evolutionary algorithm taxonomy include?
**Answer:**
- **Genetic Algorithms**: Binary strings, genetic operators
- **Evolution Strategies**: Real-valued vectors, self-adaptation
- **Evolutionary Programming**: Finite state machines, mutation focus
- **Genetic Programming**: Tree structures, program evolution
- **Differential Evolution**: Real-valued, specific DE operators

### Question 38 (G): Describe the island model architecture.
**Answer:**
- **Structure**: Multiple subpopulations evolve independently
- **Migration**: Periodic exchange of best individuals
- **Parameters**: Migration rate, interval, topology
- **Benefits**: Parallelization, diversity maintenance
- **Variants**: Ring, grid, fully connected topologies

### Question 39 (G): What are cellular evolutionary algorithms?
**Answer:**
- **Structure**: Population on spatial grid (2D typical)
- **Neighborhoods**: Moore (8), von Neumann (4), custom
- **Selection**: Limited to local neighborhood
- **Benefits**: Slow information propagation, natural diversity
- **Applications**: Spatial problems, parallel implementation

---

## 18. Evaluation and Performance

### Question 40 (G): If population size is constant and you evaluate the same number of individuals, what's the relationship between generation time and evaluation time?
**Answer:**
- **Relationship**: Generation time ∝ Evaluation time × Population size
- **Constant factors**: With fixed population, they're proportional
- **Implication**: Time complexity per generation is constant
- **Fair comparison**: Count fitness evaluations, not generations
- **Practical**: Allows comparison across different algorithms

### Question 41 (G): How do you count fitness evaluations properly?
**Answer:**
- **Include**: All individual evaluations (parents + offspring)
- **Exclude**: Cached/repeated evaluations of same solution
- **Standard**: Primary performance metric in EA literature
- **Reason**: Hardware-independent, reflects actual work done
- **Reporting**: Usually plot best fitness vs evaluations

---

## 19. Implicit vs Explicit Diversity

### Question 42 (G): What are implicit vs explicit diversity approaches?
**Answer:**
- **Explicit**: Directly modify fitness/selection
  - Fitness sharing: f'ᵢ = fᵢ/Σⱼsh(dᵢⱼ)
  - Crowding: Replace most similar parent
  - Direct diversity measures in selection
- **Implicit**: Structural/algorithmic diversity maintenance
  - Island models, cellular EAs
  - Speciation, niching through structure
  - Parameter diversity (self-adaptation)

### Question 43 (G): How does fitness sharing formula work?
**Answer:**
- **Formula**: f'ᵢ = fᵢ / Σⱼ sh(dᵢⱼ)
- **Sharing function**: sh(d) = 1-(d/σshare)^α if d < σshare, else 0
- **Effect**: Reduces fitness of individuals in crowded regions
- **Parameters**: σshare (niche radius), α (sharing slope)
- **Result**: Maintains population spread across multiple peaks

---

## 20. Exam-Specific Practice

### Question 44 (E): The exam has 30 questions, equally weighted, multiple choice, lasting 1h15min. What's the time per question?
**Answer:** 
- **Time per question**: 75 minutes ÷ 30 questions = 2.5 minutes
- **Strategy**: Quick first pass, return to difficult questions
- **Time management**: Don't spend more than 3 minutes on any question initially

### Question 45 (G): According to the professor, what should you pay attention to regarding linguistics?
**Answer:**
- **Word choice**: Exact terminology matters in definitions
- **Qualifying words**: "always", "never", "sometimes", "usually"
- **Scope**: "All", "some", "most" change meaning significantly  
- **Context**: Same concept may have different meanings in different contexts
- **Precision**: Mathematical/algorithmic descriptions must be exact

---

## Quick Reference Summary

### Core EA Loop:
1. **Initialize** population randomly
2. **Evaluate** fitness of all individuals  
3. **Select** parents based on fitness
4. **Apply** crossover and mutation to create offspring
5. **Evaluate** offspring fitness
6. **Replace** some/all population with offspring
7. **Repeat** until termination criteria

### Key Formulas:
- **Tournament Selection**: Select best from k random individuals
- **Fitness Sharing**: f'ᵢ = fᵢ / Σⱼ sh(dᵢⱼ)
- **Hamming Distance**: Σᵢ(xᵢ ≠ yᵢ) for position i
- **Self-adaptive Mutation**: σ' = σ × exp(τ × N(0,1))

### Remember for TSP:
- **Search space**: (n-1)!/2 for n cities
- **Representation**: Permutation
- **Crossover**: OX, PMX, CX
- **Mutation**: Swap, insert, inversion

**Good luck with your midterm!** 🚀