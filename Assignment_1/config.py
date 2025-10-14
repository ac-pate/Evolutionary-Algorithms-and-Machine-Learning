# config.py - Central control panel for RNA Folding EA
# Achal Patel - 40227663


# EVOLUTIONARY ALGORITHM PARAMETERS

# Population and Generation Settings
POPULATION_SIZE = 300
GENERATIONS = 250
MAX_WORKERS = 8  # Number of parallel workers for fitness evaluation

# Device-specific worker configurations (override MAX_WORKERS based on device)
DEVICE_CONFIGURATIONS = {
    'odin': {
        'max_workers': 28,  # AMD Ryzen 9 9950X - 32 threads (28 for EA)
        'population_size': 500,
        'generations': 200
    },
    'nyquist': {
        'max_workers': 10,  # AMD Ryzen 5 3600 - 12 threads (10 for EA)
        'population_size': 250,
        'generations': 150
    },
    'laptop': {
        'max_workers': 8,   # Intel i7-1260P - 12 threads (8 for EA)
        'population_size': 150,
        'generations': 120
    },
    'minipc': {
        'max_workers': 6,   # Dell OptiPlex 7050 - ~8 threads (6 for EA)
        'population_size': 100,
        'generations': 130
    }
}

# Selection and Reproduction Parameters
ELITE_PERCENTAGE = 0.01  # Percentage of elite individuals to preserve
MUTATION_RATE = 0.12  # Base mutation rate
CROSSOVER_RATE = 0.8  # Crossover probability

# TERMINATION CRITERIA
# Early termination settings
EARLY_TERMINATION_FITNESS = 0.95  # Terminate if fitness reaches this threshold
HIGH_FITNESS_STREAK_THRESHOLD = 15  # Number of consecutive high-fitness generations
MIN_DIVERSITY_FOR_TERMINATION = 0.20  # Minimum diversity required for early termination
MIN_DIVERSE_FITNESS = 0.85  # Minimum fitness for diverse sequences


# STAGNATION HANDLING

# Fitness stagnation parameters
FITNESS_STAGNATION_THRESHOLD = 10  # Generations before applying mutation boost
MUTATION_RATE_BOOST_FACTOR = 3.0  # Multiply mutation rate by this factor
MUTATION_BOOST_GENERATIONS = 5  # How long mutation boost lasts
FITNESS_THRESHOLD_FOR_BOOST = 0.75  # Only boost when fitness > this threshold

# Diversity stagnation parameters  
DIVERSITY_STAGNATION_THRESHOLD = 15  # Generations before population restart
DIVERSITY_THRESHOLD = 0.3  # Minimum acceptable population diversity
RESTART_RATE = 0.3  # Proportion of population to restart


# IPknot CONFIGURATION
# Path to IPknot executable (modify as needed for your system)
IPKNOT_PATH = "/usr/local/bin/ipknot"  # Adjust this path as needed
IPKNOT_DOCKER_CONTAINER = "ipknot_runner"  # Docker container name if using Docker
USE_SUDO_FOR_DOCKER = False  # Set to True if you need sudo for Docker commands


# OUTPUT AND RESULTS

# Results configuration
RESULTS_FOLDER = "results"  # Base folder for saving results
OUTPUT_FORMAT = ["txt", "csv"]  # Output file formats
NUM_DIVERSE_SEQUENCES = 5  # Number of diverse sequences to output
MIN_DIVERSITY_THRESHOLD = 0.15  # Minimum diversity between selected sequences

# Cache settings
ENABLE_CACHE_PRELOADING = True  # Load previous fitness cache for optimization
CACHE_FILE = "fitness_cache.json"  # Cache file name

# INPUT FILE PROCESSING
# CSV input file settings
INPUT_CSV_FILE = "src/EA_Assignment_1_constraints.csv"  # Default input file name
OUTPUT_CSV_FILE = "output/EA_Assignment_1_output.csv"  # Default output file name

# CSV column names (adjust if your file has different column names)
CSV_ID_COLUMN = "id"
CSV_STRUCTURE_COLUMN = "2d_structure" 
CSV_IUPAC_COLUMN = "IUPAC"


# LOGGING AND DEBUG

# Debug and logging settings
VERBOSE_OUTPUT = True  # Print detailed progress information
LOG_LEVEL = "INFO"  # Logging level
SAVE_INTERMEDIATE_RESULTS = False  # Save results after each generation