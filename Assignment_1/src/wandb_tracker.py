# Achal Patel - 40227663
# Weights & Biases (wandb) tracking integration for RNA Folding EA
# Handles experiment logging, metrics tracking, and visualization

import datetime
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandbTracker:
    """
    Weights & Biases tracker for RNA Folding EA experiments
    Handles initialization, logging, and cleanup for wandb tracking
    """
    
    def __init__(self, experiment_name, device, config, problem_id, run_number=1, enable_wandb=True):
        """
        Initialize wandb tracker for experiment
        
        Args:
            experiment_name (str): Name of the experiment
            device (str): Device name (odin, nyquist, laptop, minipc)
            config (dict): Experiment configuration
            problem_id (str): Problem identifier (e.g., '1.1', '2.2')
            run_number (int): Run number for this experiment
            enable_wandb (bool): Whether to enable wandb tracking
        """
        self.wandb_run = None
        self.problem_id = problem_id
        self.experiment_name = experiment_name
        self.device = device
        self.enabled = enable_wandb and WANDB_AVAILABLE
        
        if not WANDB_AVAILABLE:
            if enable_wandb:
                print("Warning: wandb is not installed. Running without wandb tracking.")
                print("To enable wandb tracking, install with: pip3 install wandb")
            return
        
        if not enable_wandb:
            print("Wandb tracking disabled by user")
            return
        
        try:
            # Create a unique run name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{experiment_name}_problem_{problem_id}_run{run_number:03d}_{timestamp}"
            
            # Initialize wandb
            self.wandb_run = wandb.init(
                project="rna-folding-ea",
                name=run_name,
                config={
                    "experiment_name": experiment_name,
                    "device": device,
                    "problem_id": problem_id,
                    "run_number": run_number,
                    "population_size": config.get('POPULATION_SIZE'),
                    "generations": config.get('GENERATIONS'),
                    "crossover_rate": config.get('CROSSOVER_RATE', 0.8),
                    "mutation_rate": config.get('MUTATION_RATE', 0.01),
                    "tournament_size": config.get('TOURNAMENT_SIZE', 3),
                    "elite_percentage": config.get('ELITE_PERCENTAGE', 0.1),
                    "early_termination_fitness": config.get('EARLY_TERMINATION_FITNESS', 0.95),
                    "high_fitness_streak_threshold": config.get('HIGH_FITNESS_STREAK_THRESHOLD', 25),
                    "timestamp": timestamp,
                    "sequence_length": len(config.get('SEQUENCE_CONSTRAINT', '')),
                    "structure_length": len(config.get('STRUCTURE_CONSTRAINT', ''))
                },
                tags=[
                    f"device:{device}",
                    f"problem:{problem_id}",
                    f"experiment:{experiment_name}",
                    f"run:{run_number}"
                ],
                group=f"{experiment_name}_problem_{problem_id}",
                job_type="evolutionary_algorithm"
            )
            
            print(f"Initialized wandb tracking: {run_name}")
            
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            self.wandb_run = None
            self.enabled = False
    
    def log_generation(self, generation, max_fitness, avg_fitness, diversity):
        """
        Log generation data to wandb - SIMPLIFIED for clean charts
        
        Args:
            generation (int): Current generation number
            max_fitness (float): Best fitness in generation
            avg_fitness (float): Average fitness in generation
            diversity (float): Population diversity
        """
        if not self.enabled or not self.wandb_run:
            return
        
        try:
            # Only log essential metrics for clean line plots
            self.wandb_run.log({
                "best_fitness": max_fitness,
                "diversity": diversity,
                "generation": generation
            }, step=generation)
            
        except Exception as e:
            print(f"Warning: Failed to log generation {generation} to wandb: {e}")
    
    def log_final_results(self, best_fitness, generation_found, runtime, valid_sequences, fitness_history, early_terminated=False, termination_reason=None):
        """
        Log minimal final experiment results to wandb - most data goes to CSV
        
        Args:
            best_fitness (float): Best fitness achieved
            generation_found (int): Generation where best fitness was found
            runtime (float): Total runtime in seconds
            valid_sequences (list): List of valid sequences with fitness > threshold
            fitness_history (list): Complete fitness history
            early_terminated (bool): Whether algorithm terminated early
            termination_reason (str): Reason for early termination
        """
        if not self.enabled or not self.wandb_run:
            return
        
        try:
            # Only log essential final metrics
            summary_metrics = {
                "final_best_fitness": best_fitness,
                "final_runtime_minutes": runtime / 60,
                "final_valid_solutions": len(valid_sequences),
                "success": 1.0 if best_fitness > 0.9 else 0.0,
                "early_terminated": early_terminated,
                "termination_reason": termination_reason or "max_generations"
            }
            
            self.wandb_run.log(summary_metrics)
            print(f"Logged essential results to wandb for problem {self.problem_id}")
            
        except Exception as e:
            print(f"Warning: Failed to log final results to wandb: {e}")
    
    def finish(self):
        """
        Finish and cleanup wandb run
        """
        if not self.enabled or not self.wandb_run:
            return
        
        try:
            self.wandb_run.finish()
            print(f"Finished wandb tracking for problem {self.problem_id}")
        except Exception as e:
            print(f"Warning: Failed to finish wandb run: {e}")
    
    def create_callback(self):
        """
        Create a callback function for EA evolution logging
        
        Returns:
            function: Callback function compatible with RNAFoldingEA.add_callback()
        """
        if not self.enabled:
            return lambda *args, **kwargs: None  # No-op function
        
        return self.log_generation


def add_wandb_tracking(ea_instance, experiment_name, device, config, problem_id, run_number=1, enable_wandb=True):
    """
    Add wandb tracking to an EA instance (similar to add_progress_monitoring)
    
    Args:
        ea_instance: Instance of RNAFoldingEA
        experiment_name (str): Name of the experiment
        device (str): Device name
        config (dict): Experiment configuration
        problem_id (str): Problem identifier
        run_number (int): Run number
        enable_wandb (bool): Whether to enable wandb tracking
    
    Returns:
        tuple: (Modified EA instance, WandbTracker instance)
    """
    # Create wandb tracker
    wandb_tracker = WandbTracker(
        experiment_name=experiment_name,
        device=device,
        config=config,
        problem_id=problem_id,
        run_number=run_number,
        enable_wandb=enable_wandb
    )
    
    # Add the callback to EA instance
    if wandb_tracker.enabled:
        callback = wandb_tracker.create_callback()
        ea_instance.add_callback(callback)
        print("Added wandb logging callback to EA")
    
    # Store reference to tracker in EA instance for later use
    ea_instance.wandb_tracker = wandb_tracker
    
    return ea_instance, wandb_tracker


def log_experiment_summary(experiment_name, device, problems_results, total_runtime, enable_wandb=True):
    """
    Log minimal experiment summary to wandb - detailed data goes to CSV
    
    Args:
        experiment_name (str): Name of the experiment
        device (str): Device name
        problems_results (dict): Results from all problems
        total_runtime (float): Total experiment runtime
        enable_wandb (bool): Whether to enable wandb tracking
    """
    if not WANDB_AVAILABLE or not enable_wandb:
        return
    
    try:
        # Create minimal experiment summary run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_run_name = f"{experiment_name}_SUMMARY_{timestamp}"
        
        with wandb.init(
            project="rna-folding-ea",
            name=summary_run_name,
            job_type="experiment_summary",
            tags=[f"device:{device}", f"experiment:{experiment_name}", "summary"]
        ) as run:
            
            # Only essential aggregate metrics
            all_fitnesses = [result['best_fitness'] for result in problems_results.values()]
            
            summary_metrics = {
                "total_runtime_minutes": total_runtime / 60,
                "problems_solved": len(problems_results),
                "avg_best_fitness": sum(all_fitnesses) / len(all_fitnesses) if all_fitnesses else 0,
                "max_fitness_achieved": max(all_fitnesses) if all_fitnesses else 0,
                "success_rate": sum(1 for f in all_fitnesses if f > 0.9) / len(all_fitnesses) if all_fitnesses else 0,
                "device": device,
                "experiment_name": experiment_name
            }
            
            run.log(summary_metrics)
            print(f"Logged minimal experiment summary to wandb: {summary_run_name}")
    
    except Exception as e:
        print(f"Warning: Failed to log experiment summary: {e}")


def create_wandb_demo():
    """
    Create a simple demonstration of wandb tracking (for testing)
    """
    if not WANDB_AVAILABLE:
        print("wandb not available for demo")
        return
    
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    
    try:
        from rna_folding_ea import RNAFoldingEA
    except ImportError:
        print("RNAFoldingEA not available for demo")
        return
    
    print("Creating wandb tracking demo...")
    
    # Create a small EA for demo
    ea = RNAFoldingEA(
        population_size=20,
        generations=5,
        sequence_constraint="NNNNNNNNNN",
        structure_constraint="((((...))))",
        max_workers=2
    )
    
    # Add wandb tracking
    config = {
        'POPULATION_SIZE': 20,
        'GENERATIONS': 5,
        'SEQUENCE_CONSTRAINT': "NNNNNNNNNN",
        'STRUCTURE_CONSTRAINT': "((((...))))"
    }
    
    ea, tracker = add_wandb_tracking(
        ea_instance=ea,
        experiment_name="demo_experiment",
        device="demo",
        config=config,
        problem_id="demo",
        run_number=1,
        enable_wandb=True
    )
    
    print("Running demo with wandb tracking...")
    ea.run_evolution()
    
    # Log final results
    if tracker.enabled:
        results = ea.get_best_individuals()
        tracker.log_final_results(
            best_fitness=max(r['fitness'] for r in results) if results else 0,
            generation_found=0,
            runtime=10.0,
            valid_sequences=results,
            fitness_history=ea.fitness_history
        )
        tracker.finish()
    
    print("Demo completed!")


if __name__ == "__main__":
    # Run a simple demo
    create_wandb_demo()