# COEN 432/6321 Evolutionary Algorithms - Comprehensive Study Guide

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Population and Individual Representation](#population-and-individual-representation)
3. [Selection Mechanisms](#selection-mechanisms)
4. [Crossover Operators](#crossover-operators)
5. [Mutation Operators](#mutation-operators)
6. [Generational Models](#generational-models)
7. [Fitness and Scaling](#fitness-and-scaling)
8. [Multi-objective Optimization](#multi-objective-optimization)
9. [Parameter Control](#parameter-control)
10. [Advanced Topics](#advanced-topics)
11. [Practice Questions](#practice-questions)

---

## Core Concepts

### Question 1 (E): What is the difference between genotype and phenotype?
**Answer:** 
- **Genotype**: The encoded representation of a solution in the EA (e.g., binary string, real-valued vector)
- **Phenotype**: The actual solution that the genotype represents when decoded (e.g., the actual values, behaviors, or structures)
- **Example**: In TSP, genotype might be [3,1,4,2] (permutation), phenotype is the actual tour visiting cities in that order

### Question 2 (G): In evolutionary algorithms, why do we need both exploration and exploitation?
**Answer:**
- **Exploration**: Searching new regions of the solution space to discover potentially better solutions
- **Exploitation**: Focusing on refining known good regions to find local optima
- **Balance is crucial**: Too much exploration → slow convergence; Too much exploitation → premature convergence to local optima
- **Achieved through**: Mutation rates, selection pressure, diversity mechanisms

### Question 3 (E): What are the main components of an evolutionary algorithm?
**Answer:**
1. **Population**: Collection of candidate solutions
2. **Selection**: Choosing parents for reproduction
3. **Reproduction**: Creating offspring through crossover and mutation
4. **Replacement**: Deciding which individuals survive to next generation
5. **Fitness Evaluation**: Measuring solution quality

### Question 4 (G): How does population size affect EA performance?
**Answer:**
- **Small population**: 
  - Pros: Faster per-generation computation, less memory
  - Cons: Limited diversity, higher chance of premature convergence
- **Large population**:
  - Pros: Better diversity, more thorough search
  - Cons: Slower convergence, more computational resources
- **Rule of thumb**: Start with 30-100 individuals, adjust based on problem complexity

---

## Population and Individual Representation

### Question 5 (E): What are the common representation schemes in EAs?
**Answer:**
1. **Binary**: Strings of 0s and 1s (e.g., "1101001")
2. **Integer**: Arrays of integers (e.g., [3, 7, 1, 9])
3. **Real-valued**: Arrays of floating-point numbers (e.g., [0.5, 3.14, -1.2])
4. **Permutation**: Ordered sequences (e.g., [3, 1, 4, 2] for TSP)
5. **Tree**: Hierarchical structures (for genetic programming)

### Question 6 (G): For the Traveling Salesman Problem, which representation would be most appropriate and why?
**Answer:**
- **Best choice**: Permutation representation
- **Reasoning**: 
  - Each city must be visited exactly once
  - Order matters (defines the tour)
  - Natural mapping to problem constraints
- **Alternative approaches**: Binary matrix, path representation
- **Why not binary/real**: Would require complex constraint handling

### Question 7 (G): How do you initialize a population in evolutionary algorithms?
**Answer:**
- **Random initialization**: Most common, generates diverse starting population
- **Heuristic seeding**: Include some solutions from domain-specific heuristics
- **Considerations**:
  - Ensure feasibility (all constraints satisfied)
  - Maximize initial diversity
  - Avoid bias toward specific regions
- **Example**: For TSP, generate random permutations ensuring each city appears exactly once

---

## Selection Mechanisms

### Question 8 (E): What is tournament selection and how does it work?
**Answer:**
- **Process**: 
  1. Randomly select k individuals (tournament size)
  2. Compare their fitness values
  3. Select the best individual as parent
  4. Repeat for each parent needed
- **Advantage**: Simple, no need for fitness scaling
- **Tournament size effect**: Larger k → higher selection pressure

### Question 9 (G): Compare fitness proportional selection vs tournament selection.
**Answer:**
| Aspect | Fitness Proportional | Tournament |
|--------|---------------------|------------|
| **Selection pressure** | Depends on fitness variance | Controlled by tournament size |
| **Fitness scaling** | Often required | Not needed |
| **Complexity** | O(n) with roulette wheel | O(k) where k << n |
| **Premature convergence** | Risk with super-individuals | More controlled |
| **Implementation** | More complex | Simple |

### Question 10 (G): What is the difference between replacement and no-replacement in tournament selection?
**Answer:**
- **With replacement**: Same individual can be selected multiple times in one tournament
  - Higher variance in selection
  - Possible for weak individuals to avoid selection
- **Without replacement**: Each individual can only be selected once per tournament
  - Lower variance, more deterministic
  - Ensures all individuals have some chance
- **Window size effect**: Larger windows → more global view of population fitness

### Question 11 (E): What is rank-based selection?
**Answer:**
- **Process**:
  1. Sort population by fitness
  2. Assign ranks (1 = worst, n = best)
  3. Select based on ranks, not raw fitness
- **Advantages**: 
  - Immune to fitness scaling issues
  - Consistent selection pressure
- **Linear ranking**: P(rank) = (2-s)/n + 2*rank*(s-1)/(n*(n-1))

---

## Crossover Operators

### Question 12 (E): What is single-point crossover?
**Answer:**
- **Process**:
  1. Choose random crossover point
  2. Swap tails of parent chromosomes
- **Example**: 
  - Parent 1: 11|01001, Parent 2: 00|11110
  - Offspring: 11|11110, 00|01001
- **Properties**: Preserves some building blocks, simple to implement

### Question 13 (G): For binary representation, is single-point crossover sufficient to search the entire space without mutation?
**Answer:**
- **No, it is not sufficient**
- **Reasoning**:
  - Can only recombine existing alleles
  - Cannot introduce new bit values not present in initial population
  - Example: If no individual has '1' at position 5, crossover alone cannot create it
- **Implication**: Mutation is essential for complete space exploration

### Question 14 (G): What are the different crossover operators for floating-point representation?
**Answer:**
1. **Arithmetic crossover**: c₁ = α·p₁ + (1-α)·p₂, c₂ = (1-α)·p₁ + α·p₂
2. **BLX-α**: Uniform selection from [min(p₁,p₂)-α·d, max(p₁,p₂)+α·d] where d=|p₁-p₂|
3. **SBX (Simulated Binary Crossover)**: Mimics binary crossover behavior
4. **Uniform arithmetic**: Apply arithmetic crossover to randomly selected components

### Question 15 (G): Describe crossover operators for permutation representations.
**Answer:**
1. **Order Crossover (OX)**:
   - Copy substring from parent 1
   - Fill remaining positions with parent 2's order
2. **Partially Matched Crossover (PMX)**:
   - Establish mapping between crossover segments
   - Apply mapping to resolve conflicts
3. **Cycle Crossover (CX)**:
   - Preserve position-element relationships
   - Create cycles to maintain feasibility

### Question 16 (E): What happens if you apply n-point crossover with n = length of chromosome?
**Answer:**
- **Result**: Uniform crossover
- **Behavior**: Each position independently chooses parent with 50% probability
- **Effect**: Maximum disruption of building blocks
- **Use case**: When fine-grained mixing is desired

---

## Mutation Operators

### Question 17 (E): What are the different types of mutation step sizes?
**Answer:**
1. **Fixed**: Constant step size throughout evolution
2. **Deterministic**: Decreases according to predetermined schedule (e.g., 1/t)
3. **Adaptive**: Adjusts based on search progress or success rate
4. **Self-adaptive**: Step size evolves with the solution (part of genotype)

### Question 18 (G): Compare small vs large mutation step sizes.
**Answer:**
| Aspect | Small Steps | Large Steps |
|--------|-------------|-------------|
| **Local search** | Excellent | Poor |
| **Exploration** | Limited | Extensive |
| **Convergence** | Smooth, gradual | Erratic, may diverge |
| **Early evolution** | Slow progress | Good for escaping local optima |
| **Late evolution** | Fine-tuning | Disruptive |

### Question 19 (G): How does self-adaptive mutation work?
**Answer:**
- **Concept**: Step sizes are part of the chromosome and evolve
- **Implementation**: x' = x + N(0,σ), σ' = σ·exp(τ·N(0,1))
- **Parameters**: τ ≈ 1/√n (global), τ' ≈ 1/√(2√n) (local)
- **Advantages**: Automatically adapts to problem landscape
- **Challenge**: Increases chromosome length

### Question 20 (E): What is the effect of mutation rate on EA performance?
**Answer:**
- **Too low**: Insufficient exploration, may get stuck
- **Too high**: Random search, destroys good solutions
- **Typical values**: 1/L for binary (L = length), 1/n for real-valued
- **Balance**: Should complement crossover operator strength

---

## Generational Models

### Question 21 (E): What is the difference between generational and steady-state models?
**Answer:**
- **Generational**: Replace entire population each generation
  - Clear generation boundaries
  - Parallel evaluation possible
  - May lose good solutions
- **Steady-state**: Replace one or few individuals at a time
  - Continuous evolution
  - Better solution preservation
  - More complex implementation

### Question 22 (G): What is the generation gap and how does it affect evolution?
**Answer:**
- **Definition**: Proportion of population replaced each generation
- **Full generation gap (1.0)**: Replace all individuals
- **Partial generation gap (<1.0)**: Replace subset
- **Effects**:
  - Lower gap → Better solution preservation
  - Higher gap → Faster population turnover
  - Intermediate values often optimal

### Question 23 (E): What is elitism and why is it used?
**Answer:**
- **Definition**: Preserve best individual(s) from each generation
- **Benefits**:
  - Prevents loss of best solution found so far
  - Guarantees monotonic fitness improvement
  - Provides steady progress measure
- **Drawback**: May slow population diversity
- **Implementation**: Copy elite to next generation unchanged

### Question 24 (G): Explain (μ + λ) vs (μ, λ) selection strategies.
**Answer:**
- **(μ + λ)**: 
  - Select μ from μ parents + λ offspring
  - Elitist (parents can survive)
  - More conservative
- **(μ, λ)**:
  - Select μ only from λ offspring (λ > μ)
  - Non-elitist (parents die)
  - More explorative
- **Usage**: (μ,λ) often better for dynamic environments

---

## Fitness and Scaling

### Question 25 (G): Why might fitness scaling be necessary?
**Answer:**
- **Raw fitness problems**:
  - Super-individuals dominate selection
  - Loss of selection pressure over time
  - Negative fitness values
- **Scaling solutions**:
  - **Linear scaling**: f' = a·f + b
  - **Sigma scaling**: f' = max(0, f - (μ - c·σ))
  - **Power scaling**: f' = f^k
  - **Windowing**: f' = f - min(f_population)

### Question 26 (G): How does windowing affect selection pressure?
**Answer:**
- **Process**: Subtract worst fitness from all individuals
- **Effect**: Eliminates negative values, increases relative differences
- **Problem**: May amplify noise in fitness evaluation
- **Best for**: Situations where absolute fitness matters less than relative ordering

### Question 27 (E): What is sigma scaling and when is it used?
**Answer:**
- **Formula**: f' = max(0, f - (μ - c·σ))
- **Purpose**: Maintain constant selection pressure
- **Parameters**: c typically 2.0-3.0
- **Effect**: Individuals more than c standard deviations below mean get zero fitness
- **Use case**: When population fitness converges

---

## Multi-objective Optimization

### Question 28 (G): What is the difference between dominated and non-dominated solutions?
**Answer:**
- **Dominated**: Solution A dominates B if A is better or equal in all objectives and strictly better in at least one
- **Non-dominated**: No other solution dominates it
- **Pareto-optimal**: Non-dominated in the entire search space
- **Example**: In TSP with time and cost: (10min, $50) dominates (15min, $60)

### Question 29 (G): Explain the difference between design space and objective space.
**Answer:**
- **Design space**: Space of possible solutions/variables
  - Example: All possible TSP tours
  - Where the EA searches
- **Objective space**: Space of objective function values
  - Example: (travel_time, cost) pairs
  - Where we evaluate quality
- **Mapping**: Each design point maps to one objective point

### Question 30 (G): What is convergence vs diversity in multi-objective optimization?
**Answer:**
- **Convergence**: Solutions approach the true Pareto front
- **Diversity**: Solutions are spread along the Pareto front
- **Trade-off**: Focus on convergence may reduce diversity
- **Desired**: Both good convergence AND good diversity
- **Metrics**: IGD (convergence), spacing (diversity)

### Question 31 (G): Why use rank instead of raw fitness in multi-objective optimization?
**Answer:**
- **Rank-based assignment**: All non-dominated solutions get same rank
- **Advantages**:
  - No scaling needed
  - Equal selection pressure within fronts
  - Focuses on dominance relationships
- **Secondary criteria**: Crowding distance, diversity measures
- **Implementation**: Fast non-dominated sorting in NSGA-II

---

## Parameter Control

### Question 32 (G): What is the difference between parameter tuning and parameter control?
**Answer:**
- **Parameter tuning**: Set parameters before the run based on preliminary experiments
  - Static throughout run
  - Requires extensive experimentation
  - Example: Set mutation rate = 0.1 for entire run
- **Parameter control**: Adjust parameters during the run
  - Dynamic adaptation
  - Can respond to search progress
  - Types: deterministic, adaptive, self-adaptive

### Question 33 (G): What are the three types of parameter control?
**Answer:**
1. **Deterministic**: Change according to predetermined rule
   - Example: μ(t) = μ₀ · (1-t/T)
   - Simple, predictable
2. **Adaptive**: Change based on search statistics
   - Example: Increase mutation if no improvement
   - Responsive to search state
3. **Self-adaptive**: Parameters evolve with solutions
   - Example: Strategy parameters in Evolution Strategies
   - Automatic optimization

### Question 34 (E): In the context of constant population size and fitness evaluations, what is the relationship between generation time and evaluation time?
**Answer:**
- **With constant population**: Each generation evaluates same number of individuals
- **Relationship**: Generation time ∝ Evaluation time
- **Implication**: Time complexity per generation is constant
- **Measurement**: Often count fitness evaluations instead of generations
- **Practical**: Allows fair comparison between different population sizes

---

## Advanced Topics

### Question 35 (G): What is fitness sharing and how does it work?
**Answer:**
- **Purpose**: Maintain population diversity by reducing fitness of similar individuals
- **Formula**: f'ᵢ = fᵢ / Σⱼ sh(dᵢⱼ)
- **Sharing function**: sh(d) = 1 - (d/σshare)^α if d < σshare, else 0
- **Effect**: Creates niches, prevents convergence to single optimum
- **Parameters**: σshare (niche radius), α (sharing slope)

### Question 36 (G): What is crowding and how does it differ from fitness sharing?
**Answer:**
- **Crowding**: Competition between parents and children
- **Process**: Replace most similar parent with offspring
- **Types**: Deterministic crowding, probabilistic crowding
- **Difference from sharing**: 
  - Implicit rather than explicit diversity maintenance
  - Local replacement vs global fitness modification
  - Often simpler to implement

### Question 37 (G): What is speciation in evolutionary algorithms?
**Answer:**
- **Concept**: Divide population into species/subpopulations
- **Methods**: 
  - Distance-based clustering
  - Spatial separation
  - Behavioral measures
- **Benefits**: 
  - Parallel exploration of multiple niches
  - Prevents species interference
  - Maintains diversity naturally

### Question 38 (G): Describe the island model for parallel EAs.
**Answer:**
- **Structure**: Multiple subpopulations (islands) evolve independently
- **Migration**: Periodic exchange of individuals between islands
- **Parameters**: 
  - Migration rate (how many)
  - Migration interval (how often)
  - Topology (which islands connected)
- **Benefits**: Natural parallelization, diversity maintenance

### Question 39 (G): What are cellular evolutionary algorithms?
**Answer:**
- **Structure**: Population arranged in spatial grid
- **Selection**: Limited to local neighborhood
- **Benefits**: 
  - Slow selection pressure propagation
  - Natural diversity maintenance
  - Emergent spatiotemporal patterns
- **Neighborhoods**: Moore (8 neighbors), von Neumann (4 neighbors)

### Question 40 (G): What does the "green chart" taxonomy of evolutionary algorithms include?
**Answer:**
The taxonomy typically covers:
- **Genetic Algorithms**: Binary representation, genetic operators
- **Evolution Strategies**: Real-valued, self-adaptation
- **Evolutionary Programming**: Mutation-only, behavioral evolution
- **Genetic Programming**: Tree-based, program evolution
- **Differential Evolution**: Real-valued, specific operators
- **Estimation of Distribution Algorithms**: Model-based approach

---

## Practice Questions

### Question 41 (E): What is the primary purpose of crossover in evolutionary algorithms?
A) Introduce random changes  
B) Combine features from two parents  
C) Evaluate fitness  
D) Select best individuals  

**Answer: B) Combine features from two parents**

### Question 42 (G): In tournament selection with tournament size k=1, what type of selection does this become?
A) Fitness proportional selection  
B) Random selection  
C) Rank-based selection  
D) Elitist selection  

**Answer: B) Random selection**

### Question 43 (G): For a TSP with 10 cities, what is the size of the search space?
A) 10!  
B) 9!/2  
C) 2^10  
D) 10^10  

**Answer: B) 9!/2** (due to symmetry and fixed starting city)

### Question 44 (E): What happens to selection pressure as tournament size increases?
A) Decreases  
B) Stays constant  
C) Increases  
D) Becomes random  

**Answer: C) Increases**

### Question 45 (G): In self-adaptive mutation, what typically happens to step sizes during evolution?
A) They increase monotonically  
B) They decrease monotonically  
C) They adapt to local landscape  
D) They remain constant  

**Answer: C) They adapt to local landscape**

### Question 46 (G): Which statement about (μ,λ) vs (μ+λ) selection is correct?
A) (μ,λ) is always better  
B) (μ+λ) is elitist, (μ,λ) is not  
C) Both are equivalent  
D) (μ,λ) requires μ > λ  

**Answer: B) (μ+λ) is elitist, (μ,λ) is not**

### Question 47 (E): What is the main advantage of rank-based selection over fitness proportional selection?
A) Faster computation  
B) Better for minimization problems  
C) Less sensitive to fitness scaling  
D) Requires less memory  

**Answer: C) Less sensitive to fitness scaling**

### Question 48 (G): In multi-objective optimization, what does IGD measure?
A) Population diversity  
B) Convergence to Pareto front  
C) Number of objectives  
D) Selection pressure  

**Answer: B) Convergence to Pareto front**

### Question 49 (G): For real-valued representation, which crossover operator preserves the range of parent values?
A) Arithmetic crossover  
B) BLX-α with α=0  
C) Uniform crossover  
D) All of the above  

**Answer: B) BLX-α with α=0**

### Question 50 (G): What is the primary benefit of the island model?
A) Faster convergence  
B) Better parallelization  
C) Lower memory usage  
D) Simpler implementation  

**Answer: B) Better parallelization**

---

## Exam Strategy Tips

1. **Time Management**: 30 questions in 75 minutes = 2.5 minutes per question
2. **Read Carefully**: Pay attention to linguistic details mentioned by professor
3. **Process of Elimination**: Use for difficult multiple choice questions
4. **Key Concepts**: Focus on concepts from professor's review list
5. **Definitions**: Know precise definitions of technical terms
6. **Comparisons**: Understand trade-offs between different approaches
7. **Applications**: Be able to apply concepts to specific problems like TSP

**Good Luck!** 🍀