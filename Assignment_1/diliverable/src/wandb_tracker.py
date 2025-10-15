# Weights & Biases tracking integration for RNA Folding EA

import datetime
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandbTracker:
    """Weights & Biases tracker for RNA Folding EA experiments"""
    
    def __init__(self, experiment_name, device, config, problem_id, run_number=1, enable_wandb=True):
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
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{experiment_name}_problem_{problem_id}_run{run_number:03d}_{timestamp}"
            
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
                    "elite_percentage": 0.01,  # Always 1%
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
            
            # Configure metrics to prevent auto-chart generation
            # Only log to custom charts, not individual metric charts
            self.wandb_run.define_metric("generation")
            self.wandb_run.define_metric("fitness_max", step_metric="generation", summary="max")
            self.wandb_run.define_metric("fitness_avg", step_metric="generation", summary="last")  
            self.wandb_run.define_metric("diversity", step_metric="generation", summary="last")
            
            # Disable auto-chart creation for cleaner dashboard
            if hasattr(self.wandb_run, 'config'):
                self.wandb_run.config.update({"_wandb": {"chart_view": "custom_only"}}, allow_val_change=True)
            
            print(f"Initialized wandb tracking: {run_name}")
            
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            self.wandb_run = None
            self.enabled = False
    
    def log_generation(self, generation, max_fitness, avg_fitness, diversity):
        """Log generation data to wandb with simplified metric names"""
        if not self.enabled or not self.wandb_run:
            return
        
        try:
            # Use simple metric names that group well in custom charts
            self.wandb_run.log({
                "generation": generation,
                "fitness_max": max_fitness,
                "fitness_avg": avg_fitness,
                "diversity": diversity
                # Remove problem_id from metrics to prevent chart proliferation
            })
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")
    
    def log_final_results(self, best_fitness, generation_found, runtime, valid_sequences, fitness_history, early_terminated=False, termination_reason=None):
        """Log final experiment results to wandb"""
        if not self.enabled or not self.wandb_run:
            return
        
        try:
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
        """Finish and cleanup wandb run"""
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
            return lambda *args, **kwargs: None
        
        return self.log_generation


def add_wandb_tracking_to_ea(ea_instance, experiment_name, device, config, problem_id, run_number=1, enable_wandb=True):
    """Add wandb tracking to EA instance"""
    wandb_tracker = WandbTracker(
        experiment_name=experiment_name,
        device=device,
        config=config,
        problem_id=problem_id,
        run_number=run_number,
        enable_wandb=enable_wandb
    )
    
    if wandb_tracker.enabled:
        def wandb_callback(generation, max_fitness, avg_fitness, diversity):
            wandb_tracker.log_generation(generation, max_fitness, avg_fitness, diversity)
        
        ea_instance.add_callback(wandb_callback)
        print("Added wandb logging callback to EA")
    
    ea_instance.wandb_tracker = wandb_tracker
    return ea_instance, wandb_tracker


if __name__ == "__main__":
    print("WandB Tracker module - use add_wandb_tracking_to_ea() to integrate with EA")