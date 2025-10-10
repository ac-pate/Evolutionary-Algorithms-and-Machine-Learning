# Evolutionary Algorithms Comprehensive Solutions

Q01. **General scheme of an EA:**
- Initialize population randomly
- REPEAT until termination condition:
  1. Evaluate fitness of all individuals
  2. Select parents based on fitness
  3. Apply variation operators (crossover/mutation) to create offspring
  4. Select survivors for next generation
- Return best solution found

Q02. **Variation operators and their role:**
- Variation operators create new solutions from existing ones
- Crossover: combines good features from parents
- Mutation: introduces new features not present in parents
- They balance exploration (finding new areas) and exploitation (improving known good areas)

Q02. **Exploration vs Exploitation relationship:**
- Exploration: searching new areas of solution space
- Exploitation: focusing on promising areas already found
- Need balance: too much exploration wastes resources, too much exploitation risks local optima
- Typically exploration decreases and exploitation increases during run

Q03. **Diversity and Selection Pressure:**
- Diversity: variety of different solutions in population
- Selection Pressure: strength of preference for better solutions
- Inverse relationship: higher selection pressure reduces diversity
- Need to balance: too low pressure = slow progress, too high = premature convergence

Q04. **Genotype vs Phenotype:**
- Genotype: internal representation of solution (e.g., binary string)
- Phenotype: external manifestation/actual solution (e.g., circuit design)
- Same phenotype can have different genotypes
- Mapping from genotype to phenotype can be complex

Q05. **Selection operation timing:**
- Parent selection: choosing individuals to create offspring
- Survival selection: choosing individuals for next generation
- Can occur twice per generation cycle

Q06. **Typical fitness progression:**
- Initially: rapid improvement
- Middle: slower improvement
- Late: plateau/diminishing returns
- Best fitness monotonically improves
- Mean fitness may fluctuate but generally improves

Q07. **Typical representations:**
- Binary: string of 0s and 1s (e.g., "10110")
- Integer: array of integers (e.g., [1,4,2,7])
- Real-valued: array of floating points (e.g., [3.14, 2.71, 1.41])
- Permutation: ordered sequence (e.g., [3,1,4,2] for TSP)

Q08. **Crossover types and biases:**
- 1-point: cuts at one point, biased toward keeping together genes that are close
- n-point: cuts at n points, reduces positional bias
- Uniform: each gene chosen randomly from either parent, no positional bias
- All suffer from distribution bias (tendency toward certain structures)

Q09. **Mutation types:**
- Uniform: fixed probability for each gene, simple but blind to progress
- Non-uniform: probability decreases over time, helps fine-tuning
- Adaptive: probability changes based on success, most flexible but complex
Advantages/Disadvantages:
- Uniform: Simple, consistent, but may disrupt too much late in search
- Non-uniform: Better late-stage optimization, but requires parameter scheduling
- Adaptive: Self-adjusting, but complex to implement and tune

Q10. **Real-valued crossover operators:**
- Arithmetic: weighted average of parents
- BLX-α: random value from interval around parents
- SBX (Simulated Binary): creates offspring near parents with controlled spread
- Linear: linear combinations of parents

Q11. **Problem types for different representations:**
- Binary: Boolean problems, digital circuit design
- Integer: scheduling, resource allocation
- Real-valued: numerical optimization, engineering design
- Permutation: routing, ordering problems (TSP)

Q12. **Permutation mutation operators:**
- Scramble: randomly reorders subset of elements
- Swap: exchanges two elements
- Insert: moves one element to new position
- Inversion: reverses order of subset
Effects:
- Order: Scramble/Inversion heavily affect, Swap/Insert less
- Adjacency: All affect but Insert preserves most relations

Q13. **PMX and Cycle crossover:**
- PMX (Partially Mapped Crossover):
  1. Selects segment and swaps
  2. Creates mapping between elements
  3. Uses mapping to resolve conflicts
- Cycle Crossover:
  1. Identifies cycles in parent permutations
  2. Alternates taking complete cycles from parents

Q14. **Edge crossover:**
- Creates edge table listing neighbors of each city
- Builds offspring maintaining edges from parents
- Prioritizes common edges between parents
- When tie occurs, chooses shortest available edge
- Good for TSP as it preserves adjacency information

Q15. **Program representation:**
- Tree structure used for programs/expressions
- Internal nodes: operators/functions
- Leaf nodes: terminals/variables
- Example: (+ (* x 2) y) represents x*2 + y

Q16. **Generational vs Steady-state:**
- Generational: entire population replaced each generation
- Steady-state: one/few individuals replaced at a time
- Generational gap: proportion of population replaced
- Gap = 1.0 for generational, < 1.0 for steady-state

Q17. **Fitness-based selection:**
1. Calculate total fitness of population
2. For each selection:
   - Generate random number r between 0 and total fitness
   - Sum fitness values until sum > r
   - Select individual where sum exceeds r

Q18. **Fitness scaling methods:**
- Linear scaling: f' = a*f + b
- Power law scaling: f' = f^k
- Ranking: f' based on rank, not actual fitness
- Exponential ranking
- Windowing: subtract minimum fitness
- Sigma truncation: based on standard deviations

Q19. **Tournament selection:**
With replacement:
1. Randomly select k individuals
2. Choose best as parent
3. Return individuals to population
Without replacement:
1. Randomly select k individuals
2. Choose best as parent
3. Remove selected individuals from pool

Q20. **Survivor selection:**
Purpose:
- Maintain fixed population size
- Control selection pressure
- Preserve best solutions
Can be based on:
- Fitness
- Age
- Diversity
- Hybrid criteria

Q21. **Fitness-based replacement methods:**
1. Replace worst
2. Probabilistic replacement proportional to fitness
3. Tournament replacement

Q22. **(μ,λ) vs (μ+λ) selection:**
- (μ,λ): Parents cannot survive, only λ offspring compete
- (μ+λ): Parents compete with offspring for survival
- (μ,λ) allows escape from local optima but may lose good solutions
- (μ+λ) ensures monotonic improvement but may get stuck

Q23. **Explicit vs Implicit diversity:**
- Explicit: directly measures and maintains diversity
- Implicit: indirectly promotes diversity through selection/replacement
- Explicit is more controlled but computationally expensive
- Implicit is simpler but less predictable

Q24. **Genotype vs Phenotype diversity:**
Genotype diversity:
- Measured in representation space
- Hamming distance for binary
- Euclidean distance for real-valued
Phenotype diversity:
- Measured in solution space
- Problem-specific metrics
- Often more meaningful but harder to calculate

Q25. **Sharing and Crowding:**
Sharing:
- Reduces fitness based on similarity to others
- Requires similarity metric
- Applied during selection
Crowding:
- Replaces similar individuals
- Maintains niches
- Applied during replacement

Q26. **Island Model and Cellular EAs:**
Island Model parameters:
- Number of islands
- Migration interval
- Migration size
- Migration topology
- Selection policy for migrants
Cellular EAs:
- Population on grid
- Interaction with neighbors only
- Implicit neighborhood structure
- Local selection and replacement

Q27. **Parameter tuning vs control:**
- Tuning: Set parameters before run and keep fixed
- Control: Adjust parameters during run
- Key difference: Static vs Dynamic approach

Q28. **Parameter control types:**
1. Deterministic: Changed by predetermined schedule
2. Adaptive: Changed based on EA progress
3. Self-adaptive: Encoded in genome and evolved

Q29. **EA parameter optimization qualities:**
- Solution quality
- Convergence speed
- Success rate
- Robustness
- Computational efficiency

Q30. **Performance metrics:**
- Best fitness: Quality of best solution found
- Average evaluations: Efficiency measure
- Success rate: Reliability measure
- Robustness: Consistency across different problems/runs

Q31. **Relevant vs Irrelevant parameters:**
- Sensitivity analysis
- Statistical significance testing
- Parameter interaction analysis
- Impact on performance metrics

Q32. **Symbolic parameter optimization challenge:**
- Discrete vs continuous search space
- No natural ordering of options
- Cannot interpolate between values
- Requires different optimization approaches

Q33. **EA parameters to optimize:**
1. Population size
2. Mutation rate
3. Crossover probability
4. Selection pressure
5. Number of generations

Q34. **Evolutionary Strategies features:**
- Self-adaptive mutation parameters
- Real-valued representation
- (μ,λ) or (μ+λ) selection
Best for:
- Continuous optimization
- Engineering design
- Parameter optimization

Q35. **Evolutionary Programming features:**
- Focuses on behavioral evolution
- Usually no crossover
- Emphasis on mutation
- Problem-specific representation

Q36. **EP applications:**
- Finite state machines
- Neural networks
- Prediction problems
- Pattern recognition

Q37. **GP initialization (Ramped half-and-half):**
- Creates trees of various depths up to max depth
- Half full trees (all branches same depth)
- Half grown trees (variable depth branches)
- Ensures diversity in initial population

Q38. **GP variation operators:**
Mutation:
- Subtree replacement
- Point mutation
- Node type change
Crossover:
- Subtree exchange
- Context-preserving
- Homologous

Q39. **GP mutation timing:**
- Typically applied after crossover
- Lower probability than crossover
- Can be used as repair mechanism
- Some systems apply in parallel

Q40. **100-X parent selection:**
- Take best X% of population as parents
- Similar to truncation selection
- High selection pressure
- Risk of premature convergence

Q41. **Bloat in GP:**
Bloat: Uncontrolled growth of program size
Reduction methods:
- Parsimony pressure
- Size limits
- Depth limits
- Multi-objective optimization

Q42. **Classifier evolution:**
- Classifier: Rule-based system for categorization
- Evolution can optimize:
  - Rule conditions
  - Rule actions
  - Rule weights
  - Rule set composition

Q43. **Michigan vs Pittsburgh LCS:**
Michigan:
- Individual = single rule
- Population = rule set
- Online learning
Pittsburgh:
- Individual = complete rule set
- Population = multiple rule sets
- Offline learning

Q44. **MCS (Michigan Classifier System) algorithm:**
1. Initialize rule population
2. For each input:
   - Match rules
   - Select action
   - Get reward
   - Update rule strengths
3. Periodically apply GA to rule population

Q45. **Human-Computer collaboration:**
Yes, through interactive evolution:
- Computer generates variations
- Human evaluates fitness
- Example: Art/music generation
- Computer provides diversity
- Human provides aesthetic judgment
