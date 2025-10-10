# Evolutionary Algorithms - 4hr Midterm Mastery Practice Pack (COEN 432 / COEN 6321)

> purpose: this markdown pack is a dense practice bank and quick reference designed to push you from zero to midterm-ready in about 4 hours. it contains a focused study plan, a cheat-sheet, 60 multiple-choice practice questions (each tagged as (E) - easy or (G) - graduate level) with short solutions right under every question, and a timed 30-question mock exam you can use for a real midterm simulation.

# quick instructions

* spend the first 90 minutes on the "how-to-study in 4 hours" plan in the summary below, and use the linked resources in the chat to read lecture slides and 1-2 textbook chapters.
* do the MCQ bank for coverage. do the mock 30-question exam under timed conditions (60 minutes) once you finished practice.
* use the cheat-sheet pages when reviewing wrong answers.

# cheat sheet - pages to memorize (one-page condensate)

* evolution algorithm core loop - initialize population, evaluate fitness, select parents, apply variation (crossover/mutation), create offspring, select survivors, repeat.
* genotype vs phenotype - genotype: encoded representation; phenotype: decoded candidate solution evaluated by fitness.
* selection highlights - roulette is fitness-proportional and sensitive to scaling; tournament is robust to scaling and simple; rank selection prevents super-individual domination.
* variation highlights - crossover recombines parent material; mutation introduces novelty and helps exploration.
* key algorithms - ga (binary/perm/real), es (real-valued, strategy parameters), de (differential candidate-based real recomb), gp (program trees), gp uses subtree crossover.
* parameter heuristics - population size increases exploration; higher mutation increases exploration but can destroy building blocks; use elitism (1-5%) to keep best solutions.
* multiobjective - pareto dominance, front ranking, crowding distance (nsga-ii idea).
* common encodings - binary, real-valued vector, permutation for tsp, tree for gp.
* quick rules - apply repair for hard permutation constraints, use penalty for simpler constraints; consider self-adaptation for step-size parameters.

# study plan - 4 hours to midterm (concise and brutal)

1. 0:00-0:10 - rapid orientation

* skim the course outline and the instructor's slides for topics emphasized. mark high-frequency topics (selection, crossover, mutation, representations, ES/DE basics, fitness, diversity, multiobjective concepts).

2. 0:10-1:10 - core concepts from lightweight sources

* read the short chapter "what is an ea" and a 20-page tutorial (eiben & smith chapter + whitley tutorial are ideal). focus on: components of an ea, representation, selection, variation, generational vs steady-state, simple algorithms (1+1 es, basic ga), and key vocabulary.

3. 1:10-2:10 - practice questions (mcq bank)

* do the first 30 practice questions in this pack (mix of easy and graduate). for every question you missed, immediately read the short explanation in the answer and add the concept to the cheat-sheet.

4. 2:10-3:10 - focused reading on weak areas

* use the provided links for short papers/slides on: differential evolution (operators and parameters), evolutionary strategies and the 1/5 success rule, and basics of nsga-ii for multiobjective.

5. 3:10-4:00 - timed mock

* take the 30-question timed mock (60 minutes if you want to simulate fully; since you have one hour midterm, do 60 min now). then review the answer explanations for incorrect items.

# practice question bank - 60 mcqs (answers and short solutions under each question)

## notation

(E) - easy  (G) - graduate level

---

# 1

**Q1 (E)**: In the minimal definition, for an algorithm to be called an evolutionary algorithm it must have which two components?

A. gradient estimation and crossover
B. population of candidate solutions and variation operators
C. ancestor tracking and mutation only
D. deterministic hill-climbing and elitism

**Answer: B**

**Explanation:** an ea requires a population and variation (mutation or recombination) that produce new candidate solutions which are then selected. without a population or variation it is not an evolutionary method.

# 2

**Q2 (E)**: which statement correctly describes genotype vs phenotype?

A. genotype is the decoded solution; phenotype is the encoded string
B. genotype is the encoded representation; phenotype is the expressed solution evaluated by the fitness function
C. genotype is always binary; phenotype is always real-valued
D. genotype is the fitness value; phenotype is the selection probability

**Answer: B**

**Explanation:** genotype refers to the encoding (bitstring, vector, tree); phenotype is the decoded candidate used to evaluate fitness.

# 3

**Q3 (E)**: which selection method is robust to scaling of the raw fitness values?

A. fitness-proportional (roulette) selection
B. tournament selection
C. fitness-sharing
D. elitist selection

**Answer: B**

**Explanation:** tournament selection uses comparisons between individuals, so it is less sensitive to absolute fitness scale than roulette selection.

# 4

**Q4 (E)**: which of the following is a primary disadvantage of roulette-wheel (fitness-proportional) selection?

A. it cannot produce selective pressure
B. it always picks the worst individual
C. it is sensitive to fitness scaling and can be dominated by a super-individual
D. it needs a tree representation

**Answer: C**

**Explanation:** roulette selection can give extremely high selection probability to very fit individuals, causing premature convergence unless scaling is applied.

# 5

**Q5 (E)**: one-point crossover applied to binary strings does what?

A. swaps a prefix of bits between parents at a randomly chosen point
B. swaps exactly one bit between parents
C. mutates a single bit in a child
D. applies gaussian noise to bits

**Answer: A**

**Explanation:** one-point crossover picks a cut point and exchanges tails between parents, creating two offspring that combine parental segments.

# 6

**Q6 (E)**: in permutation representations (e.g., TSP), which operator is suited to keep permutations valid?

A. simple bit flip
B. order crossover (OX) or pmx
C. gaussian mutation
D. uniform crossover on indices

**Answer: B**

**Explanation:** order crossover and partially mapped crossover (pmx) preserve permutation validity. naive bit flips break the permutation.

# 7

**Q7 (E)**: differential evolution (de) primarily creates offspring by which operation?

A. averaging parents
B. adding a scaled difference of two vectors to a third vector
C. swapping subtrees of program trees
D. applying bitwise mutation

**Answer: B**

**Explanation:** de uses donors formed by v = a + F*(b - c) which injects directional information from differences between solution vectors.

# 8

**Q8 (E)**: which multiobjective concept defines solutions that are not strictly worse than any other solution across all objectives?

A. hubbard set
B. pareto dominance and pareto optimal front
C. elitist survivor set
D. convex hull point

**Answer: B**

**Explanation:** pareto-optimal solutions are those not dominated by any other; the set of such solutions is the pareto front.

# 9

**Q9 (E)**: what is elitism in evolutionary algorithms?

A. increasing mutation rate for top individuals
B. copying the best individual(s) unchanged into the next generation
C. adding noise to the fitness values
D. discarding the top 5% of population

**Answer: B**

**Explanation:** elitism preserves best solutions by ensuring they survive to the next generation unchanged.

# 10

**Q10 (E)**: which of the following is true about the 1/5 success rule in evolution strategies?

A. it recommends increasing step-size when the fraction of successful mutations is greater than 1/5
B. it recommends decreasing mutation when success rate exceeds 1/5
C. it is a crossover operator for permutations
D. it is a multiobjective ranking method

**Answer: A**

**Explanation:** the 1/5 success rule adjusts mutation step size to keep success rate near 1/5; if success > 1/5 increase step size, else decrease.

# 11

**Q11 (E)**: which representation and operator pair is most natural for genetic programming?

A. binary string and one-point crossover
B. tree representation and subtree crossover
C. permutation and pmx
D. real-valued vector and gaussian mutation

**Answer: B**

**Explanation:** genetic programming uses tree-structured programs and subtree crossover which swaps subtrees between parents.

# 12

**Q12 (E)**: what does "premature convergence" refer to in population-based search?

A. population size becoming infinite
B. population losing diversity and converging to a local optimum
C. mutation rate becoming zero
D. running out of memory before finishing

**Answer: B**

**Explanation:** premature convergence happens when diversity collapses and the algorithm gets trapped in suboptimal areas.

# 13

**Q13 (G)**: the schema theorem is often interpreted as explaining why crossover is beneficial. which statement aligns with the schema theorem intuition?

A. short low-order schemata with above-average fitness increase exponentially in frequency under selection and crossover
B. crossover eliminates all schemata
C. schema theorem proves convergence to global optimum
D. schema theorem applies only to real-valued vectors

**Answer: A**

**Explanation:** the schema theorem suggests that building blocks (short, low-order, fit schemata) are propagated and combined by crossover, although the theorem is an inequality and has caveats.

# 14

**Q14 (G)**: which concept best captures 'deceptive' fitness functions used to study GA behavior?

A. functions that reward random search
B. functions where low-order building blocks lead search away from the global optimum
C. unimodal smooth landscapes
D. functions with trivial optima

**Answer: B**

**Explanation:** deceptive functions mislead recombination by making locally attractive schemata incompatible with global optimum.

# 15

**Q15 (G)**: which algorithm family is most commonly associated with self-adapting step-size parameters embedded in the genotype?

A. genetic programming
B. evolution strategies
C. particle swarm optimization
D. simulated annealing

**Answer: B**

**Explanation:** evolution strategies often use self-adaptation where mutation step-sizes are part of the individual’s genome and undergo variation.

# 16

**Q16 (E)**: which replacement strategy preserves more parental material: (mu, lambda) or (mu + lambda)?

A. (mu, lambda) preserves parents by default
B. (mu + lambda) is elitist and can preserve parents if they are among the top mu
C. both are identical in practice
D. neither uses parents at all

**Answer: B**

**Explanation:** (mu + lambda) selects from parents and offspring combined and thus can preserve good parents (elitist); (mu, lambda) selects only from offspring.

# 17

**Q17 (E)**: in a tournament selection with size 3, tie-breaking is typically resolved how?

A. randomly among the tied competitors
B. lexicographically by genotype
C. by always choosing the older individual
D. by flipping bits of the genotype

**Answer: A**

**Explanation:** ties in tournaments are usually broken randomly among the tied individuals.

# 18

**Q18 (G)**: what is the primary advantage of rank-based selection over fitness-proportional selection?

A. rank selection is faster computationally
B. rank selection removes sensitivity to the absolute fitness scale and limits selection pressure of outliers
C. rank selection always converges faster
D. rank selection does not need fitness evaluations

**Answer: B**

**Explanation:** rank-based methods sort individuals and assign selection probabilities based on rank, not raw fitness, reducing domination by a single super-individual.

# 19

**Q19 (E)**: which of the following is a niching method used to maintain diversity?

A. elitism
B. fitness sharing
C. roulette wheel
D. one-point crossover

**Answer: B**

**Explanation:** fitness sharing reduces fitness of similar individuals so multiple niches can be maintained across the population.

# 20

**Q20 (G)**: nsga-ii introduced which two key ideas to improve multiobjective evolutionary optimization?

A. boltzmann selection and simulated annealing
B. fast nondominated sorting and crowding-distance based diversity preservation
C. differential operators and tournament sizes
D. genetic programming and subtree mutation

**Answer: B**

**Explanation:** nsga-ii's contributions include efficient nondominated sorting and a crowding distance metric to maintain spread along the pareto front.

# 21

**Q21 (E)**: which operator is most appropriate for real-valued vectors when you want small, normally distributed perturbations?

A. gaussian mutation (add normal noise)
B. bit-flip
C. pmx
D. order crossover

**Answer: A**

**Explanation:** gaussian additive mutation fits real-valued representations and produces small continuous changes.

# 22

**Q22 (E)**: what is a primary use of crossover in genetic algorithms?

A. to evaluate fitness functions
B. to recombine partial solutions from parents to create offspring that inherit building blocks
C. to remove constraints from solutions
D. to guarantee global optimum

**Answer: B**

**Explanation:** crossover recombines parental material, aiming to create offspring that preserve partial solutions (building blocks).

# 23

**Q23 (G)**: which of these best describes the no free lunch theorem relevant to search algorithms?

A. averaged over all possible problems, every search algorithm has the same performance
B. gradient descent always outperforms random search
C. genetic algorithms always beat simulated annealing
D. no algorithm can solve NP-complete problems

**Answer: A**

**Explanation:** the no free lunch theorem states that over the space of all possible objective functions, no algorithm is universally better than any other when averaged across problems.

# 24

**Q24 (E)**: which of the following is an example of a deceptive function used in GA analysis?

A. one-max
B. trap function
C. sphere function
D. linear regression loss

**Answer: B**

**Explanation:** trap functions are specifically designed to have local optima that attract search away from the global optimum, making them deceptive.

# 25

**Q25 (G)**: in differential evolution, the mutation factor F typically lies in which range?

A. negative values only
B. around 0.4 to 1.0 typically
C. only integer values 1 or 2
D. exactly 0.01

**Answer: B**

**Explanation:** practical DE uses F in roughly 0.4-1.0; values near 0.5 are common starting points.

# 26

**Q26 (E)**: which population initialization strategy is generally good practice?

A. initialize all individuals to the same random vector
B. sample uniformly across the feasible domain to maximize initial diversity
C. start with only zeros
D. copy the same best-known solution into all slots

**Answer: B**

**Explanation:** uniform sampling gives diverse starting points and helps exploration in early generations.

# 27

**Q27 (G)**: what is the effect of increasing tournament size on selection pressure?

A. increases selection pressure because the best of a larger group tends to be stronger
B. decreases selection pressure
C. has no effect
D. converts to rank selection

**Answer: A**

**Explanation:** larger tournaments increase the probability that a highly fit individual will be selected, intensifying selection pressure.

# 28

**Q28 (E)**: which of the following encodings is best suited for problems like knapsack with binary decisions?

A. permutation encoding
B. binary vector encoding
C. tree encoding
D. real-valued gaussian encoding

**Answer: B**

**Explanation:** binary vector encoding naturally represents presence/absence decisions like knapsack items.

# 29

**Q29 (G)**: which method is a common constraint-handling technique when constraints produce infeasible offspring for permutation problems?

A. ignore constraints always
B. repair the offspring to a valid permutation
C. convert the permutation to binary
D. stop the algorithm

**Answer: B**

**Explanation:** repair functions adjust infeasible offspring so they become valid solutions while preserving as much structure as possible.

# 30

**Q30 (E)**: which is a common termination criterion for EAs?

A. fixed number of generations
B. stagnation threshold (no improvement for K generations)
C. reaching a target fitness
D. any of the above

**Answer: D**

**Explanation:** any of these are valid termination criteria depending on examiner or user preference.

# 31

**Q31 (G)**: covariance matrix adaptation evolution strategy (cma-es) primarily adapts which element of mutation?

A. mutation step size and correlated directions via covariance matrix
B. tournament size
C. pmx operator probability
D. number of offspring only

**Answer: A**

**Explanation:** cma-es adapts a covariance matrix to shape mutation distribution to match the problem landscape.

# 32

**Q32 (E)**: in a (1+1) evolutionary strategy, how many parents and offspring are produced each generation?

A. one parent, one offspring; the best survives
B. one parent, zero offspring
C. two parents, ten offspring
D. ten parents, ten offspring

**Answer: A**

**Explanation:** (1+1) es has one parent that produces one offspring by mutation; selection between them determines the next parent.

# 33

**Q33 (G)**: which property differentiates steady-state GA from generational GA?

A. steady-state replaces only a few individuals each generation while generational replaces the whole population
B. steady-state cannot use crossover
C. generational has no mutation
D. they are mathematically identical always

**Answer: A**

**Explanation:** steady-state evolves by inserting a few offspring into population and removing some individuals, creating overlapping generations.

# 34

**Q34 (E)**: which metric is commonly used in multiobjective EAs to measure diversity along the pareto front?

A. hamming distance
B. crowding distance
C. euclidean fitness
D. tournament index

**Answer: B**

**Explanation:** crowding distance estimates local density along the pareto front so algorithms can prefer less crowded solutions.

# 35

**Q35 (G)**: consider an nk landscape: increasing k (epistasis) does what to the landscape?

A. makes the landscape smoother and easier
B. makes the landscape more rugged and increases local optima
C. reduces the number of local optima to zero
D. converts the landscape to convex

**Answer: B**

**Explanation:** higher k increases interdependency between bits, producing more rugged landscapes and making search harder.

# 36

**Q36 (E)**: which is a simple diversity-preserving operator or mechanism?

A. elitism with clone promotion
B. random immigrant insertion
C. always picking the same parent
D. removing mutation

**Answer: B**

**Explanation:** random immigrants periodically add new random individuals to increase diversity.

# 37

**Q37 (G)**: what is a common reason to use surrogate models in evolutionary optimization?

A. fitness is cheap to compute
B. fitness evaluation is expensive (e.g., simulation) so surrogate approximates it to save computation
C. to avoid mutation entirely
D. to make the algorithm deterministic

**Answer: B**

**Explanation:** surrogate or meta-models approximate expensive fitness evaluations, enabling more candidate evaluations at lower cost.

# 38

**Q38 (E)**: which of these is an advantage of island models in population-based EAs?

A. reduce exploration via isolation
B. allow parallelism and preserve diversity by occasional migration
C. eliminate the need for mutation
D. guarantee global optimum

**Answer: B**

**Explanation:** island models run subpopulations in parallel with migration events that exchange individuals, helping maintain diversity.

# 39

**Q39 (E)**: what does "elitist" mean in multiobjective EAs like nsga-ii?

A. best individuals are discarded
B. best individuals are retained when constructing next generation, e.g., using combined parent-offspring selection
C. only one objective is considered elite
D. elitist means no diversity measure

**Answer: B**

**Explanation:** elitist multiobjective EAs keep nondominated solutions from previous generations, preventing loss of good pareto points.

# 40

**Q40 (G)**: which of the following statements about schema theorem limitations is true?

A. it is a precise equality describing exact propagation
B. it is an inequality and ignores linkage and epistasis complexities
C. it prohibits mutation
D. it applies only to nsga-ii

**Answer: B**

**Explanation:** schema theorem gives a lower bound and has limitations; it does not fully account for disruptive effects of crossover, linkage and epistasis.

# 41

**Q41 (E)**: which selection method can be implemented without sorting the whole population?

A. rank selection
B. tournament selection
C. nondominated sorting
D. crowding distance sorting

**Answer: B**

**Explanation:** tournament selection picks k random individuals and selects the best, requiring only k comparisons per selection.

# 42

**Q42 (G)**: what is the primary idea behind steady-state replacement's potential advantage?

A. faster evaluation of new solutions by keeping population mostly unchanged and only updating a few entries at a time, leading to smoother dynamics
B. it always yields global optimum in fewer generations
C. it eliminates selection pressure entirely
D. it uses no variation operators

**Answer: A**

**Explanation:** steady-state replacement introduces slight steady change which can be beneficial for tracking moving optima and smoothing search dynamics.

# 43

**Q43 (E)**: what does 'fitness sharing' do to an individual's fitness?

A. increases it proportionally to similarity
B. reduces effective fitness in crowded regions to encourage niches
C. sets it to zero always
D. duplicates it across population

**Answer: B**

**Explanation:** fitness sharing reduces the fitness of individuals in crowded niches to discourage overcrowding.

# 44

**Q44 (G)**: which property did the early evolutionary programming researchers emphasize compared to genetic algorithms?

A. focus on evolving finite-state machines or program behaviors rather than binary string recombination
B. use of crossover exclusively
C. working only on real-valued vectors
D. avoidance of stochastic elements

**Answer: A**

**Explanation:** evolutionary programming historically focused on evolving behaviors (e.g., finite state automata) and emphasized mutation-based search rather than recombination.

# 45

**Q45 (E)**: which of the following is the simplest mutation operator for bitstring encodings?

A. gaussian additive mutation
B. bit-flip with probability pm
C. subtree mutation
D. edge recombination

**Answer: B**

**Explanation:** bit-flip toggles each bit with small probability and is commonly used for binary encodings.

# 46

**Q46 (G)**: when encoding traveling salesman problem (tsp) as permutation, which crossover operator typically preserves edges best?

A. one-point crossover
B. edge recombination crossover
C. gaussian crossover
D. subtree crossover

**Answer: B**

**Explanation:** edge recombination tries to preserve adjacency relationships (edges) from parents, which often helps tsp performance.

# 47

**Q47 (E)**: what is the role of mutation in population-based EAs?

A. create new genetic material and prevent premature convergence
B. compute fitness faster
C. always reverse crossovers
D. guarantee global optimality

**Answer: A**

**Explanation:** mutation injects new variation and enables exploration of previously unseen genotypes.

# 48

**Q48 (G)**: in runtime analysis for simple EAs optimizing a finite domain, drift analysis is used to...

A. transform the algorithm into a differential equation
B. bound expected time to hit the optimum by analyzing expected progress per step
C. measure floating point error
D. prove multiobjective pareto optimality

**Answer: B**

**Explanation:** drift analysis studies expected change of a potential function to bound hitting times to target states in stochastic processes.

# 49

**Q49 (E)**: why might one use steady-state insertion of random immigrants during evolution?

A. to guarantee mutation goes to zero
B. to maintain or reintroduce diversity
C. to remove fitness evaluations
D. to reduce run time to zero

**Answer: B**

**Explanation:** random immigrants help keep population diverse and avoid stagnation.

# 50

**Q50 (E)**: in genetic programming, bloat refers to what phenomenon?

A. explosive growth in program tree sizes without corresponding fitness improvements
B. rapid reduction of tree sizes
C. overheating of processor
D. inability to mutate nodes

**Answer: A**

**Explanation:** bloat is growth of program representation size (often redundant code) that does not improve performance.

# 51

**Q51 (G)**: epistasis in genetic algorithms is best described as...

A. a measure of how independently gene positions contribute to fitness
B. the probability of mutation per generation
C. a crossover operator for trees
D. the method of ranking in nsga-ii

**Answer: A**

**Explanation:** epistasis refers to interactions between genes; high epistasis implies strong dependencies making recombination less straightforward.

# 52

**Q52 (E)**: which of the following commonly speeds up evolutionary algorithm experiments using parallel hardware?

A. evaluate fitnesses of different individuals in parallel
B. serially evaluate individuals only
C. turn off mutation and crossovers
D. run algorithm for fewer generations

**Answer: A**

**Explanation:** fitness evaluation is often the most expensive step and is easily parallelizable across individuals.

# 53

**Q53 (G)**: in a population of size N, genetic drift describes...

A. the deterministic improvement of best fitness
B. random fluctuations in allele frequencies due to finite sampling which can lead to loss of diversity
C. the process of crossover exclusively
D. the method to adapt mutation rate

**Answer: B**

**Explanation:** genetic drift is stochastic sampling variation in finite populations, possibly eliminating alleles even if neutral.

# 54

**Q54 (E)**: what is a common initial value for crossover probability in classic genetic algorithms?

A. 0.0
B. 0.6 to 0.9 commonly used
C. exactly 2
D. 100

**Answer: B**

**Explanation:** crossover probability is often set high (0.6-0.9) so recombination is frequent, while mutation is kept low.

# 55

**Q55 (G)**: when would you prefer differential evolution over a standard real-coded GA?

A. when the problem is discrete and small
B. when the problem benefits from vector-difference based exploration and has continuous parameters
C. when you need tree-structured programs
D. when fitness is not defined

**Answer: B**

**Explanation:** de often performs well for continuous parameter optimization because its operators exploit vector differences for search direction.

# 56

**Q56 (E)**: which factor increases exploration in an ea?

A. larger mutation rate
B. zero population size
C. no variation operators
D. immediate stop

**Answer: A**

**Explanation:** higher mutation increases the chance of exploring new regions of the search space.

# 57

**Q57 (G)**: in multiobjective optimization with combinatorial problems, which technique handles constraints and multiple objectives simultaneously most naturally?

A. single-objective weighted sum always works and finds all pareto points
B. specialized multiobjective EA with repair and nondominated sorting often works well in practice
C. convert to GP always
D. use tournament size zero

**Answer: B**

**Explanation:** multiobjective EAs with mechanisms for constraints (repair/penalty) plus nondominated selection can approximate pareto sets.

# 58

**Q58 (E)**: which method can reduce the effect of fitness scaling making roulette selection less extreme?

A. sigma scaling or linear scaling
B. reducing population size to 1
C. eliminating mutation entirely
D. using pmx only

**Answer: A**

**Explanation:** sigma scaling and other scaling methods adjust raw fitness to reduce dominance of outliers.

# 59

**Q59 (G)**: explain why crossover might be harmful on representations with high epistasis unless linkage is respected?

A. crossover is always beneficial
B. crossover can break useful joint gene interactions (linkage) that are necessary for high fitness, thus disrupting good building blocks
C. crossover reduces algorithm speed
D. crossover prevents mutation

**Answer: B**

**Explanation:** when gene interactions are important, naive crossover disrupts beneficial combinations. linkage-aware operators or representation changes help.

# 60

**Q60 (E)**: which termination rule uses "no improvement for K generations"?

A. fitness threshold
B. stagnation detection
C. insanity test
D. tournament finish

**Answer: B**

**Explanation:** stagnation detection stops the run if no improvement occurs for a preset number of generations.

---

# timed mock - 30-question practice (simulate your midterm)

* instructions: set a timer for 60 minutes. answer the 30 questions below without looking at solutions. after finishing, check the answer key and read explanations for any mistakes.

### mock questions (30)

1. q1 from bank (use same as question 1)
2. q3 from bank
3. q5 from bank
4. q7 from bank
5. q9 from bank
6. q10 from bank
7. q12 from bank
8. q14 from bank
9. q17 from bank
10. q18 from bank
11. q20 from bank
12. q21 from bank
13. q23 from bank
14. q25 from bank
15. q26 from bank
16. q27 from bank
17. q29 from bank
18. q31 from bank
19. q33 from bank
20. q35 from bank
21. q36 from bank
22. q38 from bank
23. q40 from bank
24. q42 from bank
25. q43 from bank
26. q45 from bank
27. q46 from bank
28. q49 from bank
29. q52 from bank
30. q54 from bank

### mock answer key

1. B
2. B
3. A
4. B
5. B
6. A
7. A
8. B
9. A
10. B
11. B
12. A
13. B
14. B
15. B
16. A
17. B
18. A
19. A
20. B
21. B
22. B
23. B
24. A
25. B
26. B
27. B
28. B
29. B
30. B

# short pseudocode snippets (for mental model)

```python
# simple generational genetic algorithm - pseudo python
# population: list of genotypes
# fitness: function mapping genotype to real number
# pm: mutation probability
# pc: crossover probability

def initialize_population(n):
    # initialize n individuals uniformly in domain
    return [random_individual() for _ in range(n)]

# selection: tournament selection of size t

def tournament_select(pop, t):
    # pick t random individuals and return the best
    contestants = random.sample(pop, t)
    return max(contestants, key=lambda ind: fitness(ind))

# main loop
pop = initialize_population(N)
for generation in range(max_gens):
    offspring = []
    while len(offspring) < N:
        parent1 = tournament_select(pop, 3)
        parent2 = tournament_select(pop, 3)
        if random() < pc:
            child1, child2 = one_point_crossover(parent1, parent2)
        else:
            child1, child2 = copy(parent1), copy(parent2)
        # mutation
        mutate(child1, pm)
        mutate(child2, pm)
        offspring.extend([child1, child2])
    # elitism - keep best from parents
    best_parent = max(pop, key=lambda ind: fitness(ind))
    pop = select_survivors(pop + offspring, N, elitist=best_parent)
```

# final notes

* use the mock exam as a final rehearsal exactly one time before the test. try to simulate conditions: sitting in your study space, no phones, strict timer.
* correct the mock by reading the short answers here and marking the 5 concepts you missed. spend the remaining review time on those concepts.

# good luck
