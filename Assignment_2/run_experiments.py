"""
Comprehensive Experiments Script
Runs two types of experiments:
1. pos_weight sweep (1-10) with fixed threshold=0.5
2. threshold sweep (0.3-0.7) with fixed pos_weight=5

For each parameter value, runs 3 tests to measure variance
Saves all results to JSON files for later plotting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import os
from datetime import datetime
import json

# Set CUDA memory configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"CUDA memory cleared\n")

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
NUM_EPOCHS = 20
MAX_LEN = 128
BATCH_SIZE = 128
RUNS_PER_VALUE = 3

# Experiment 1: pos_weight sweep
POS_WEIGHT_START = 1
POS_WEIGHT_END = 10
FIXED_THRESHOLD_EXP1 = 0.5

# Experiment 2: threshold sweep
THRESHOLD_START = 0.3
THRESHOLD_END = 0.7
THRESHOLD_STEP = 0.05
FIXED_POS_WEIGHT_EXP2 = 5

print("="*70)
print("EXPERIMENT CONFIGURATION")
print("="*70)
print(f"Epochs per run: {NUM_EPOCHS}")
print(f"Max Length: {MAX_LEN}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Runs per value: {RUNS_PER_VALUE}")
print("\nExperiment 1: pos_weight sweep")
print(f"  pos_weight range: {POS_WEIGHT_START} to {POS_WEIGHT_END}")
print(f"  Fixed threshold: {FIXED_THRESHOLD_EXP1}")
print("\nExperiment 2: threshold sweep")
print(f"  Threshold range: {THRESHOLD_START} to {THRESHOLD_END} (step {THRESHOLD_STEP})")
print(f"  Fixed pos_weight: {FIXED_POS_WEIGHT_EXP2}")
print("="*70 + "\n")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data_from_csv(file_path):
    """Loads sequence and structure data from a CSV file."""
    df = pd.read_csv(file_path)
    data_tuples = [(row['sequence'], row['structure']) for _, row in df.iterrows()]
    return data_tuples

def one_hot_encode(sequence, max_len):
    """One-hot encodes an RNA sequence."""
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    encoded = np.zeros((max_len, 4), dtype=np.float32)
    sequence = sequence[:max_len]
    for i, base in enumerate(sequence):
        if base in base_to_int:
            encoded[i, base_to_int[base]] = 1
    return encoded

def create_contact_map(dot_bracket, max_len):
    """Creates a contact map from a dot-bracket string."""
    dot_bracket = dot_bracket[:max_len]
    contact_map = np.zeros((max_len, max_len), dtype=np.float32)
    stack = []
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            contact_map[i, j] = 1
            contact_map[j, i] = 1
    return contact_map

class RNADataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, structure = self.data[idx]
        seq_encoded = one_hot_encode(sequence, self.max_len)
        contact_map = create_contact_map(structure, self.max_len)
        seq_tensor = torch.from_numpy(seq_encoded).float()
        contact_tensor = torch.from_numpy(contact_map).float()
        return seq_tensor, contact_tensor

# Load data
print("Loading data...")
train_data = load_data_from_csv("./TR0.csv")
val_data = load_data_from_csv("./VL0.csv")
test_data = load_data_from_csv("./TS0.csv")
print(f"Loaded {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test samples\n")

# Create datasets
train_dataset = RNADataset(train_data, MAX_LEN)
val_dataset = RNADataset(val_data, MAX_LEN)
test_dataset = RNADataset(test_data, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class RNAFoldingCNN(nn.Module):
    def __init__(self, input_channels=8):
        super(RNAFoldingCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x_1d):
        batch_size = x_1d.shape[0]
        max_len = x_1d.shape[1]
        
        x_2d_i = x_1d.unsqueeze(2).repeat(1, 1, max_len, 1)
        x_2d_j = x_1d.unsqueeze(1).repeat(1, max_len, 1, 1)
        x_2d = torch.cat([x_2d_i, x_2d_j], dim=-1)
        x_2d = x_2d.permute(0, 3, 1, 2)
        
        x = self.relu1(self.bn1(self.conv1(x_2d)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        
        x = x.squeeze(1)
        
        if self.training:
            x = (x + x.transpose(1, 2)) / 2
            return x
        else:
            x = torch.sigmoid(x)
            x = (x + x.transpose(1, 2)) / 2
            return x

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(y_pred, y_true, threshold):
    """Calculates F1, Recall, and AUC scores."""
    y_pred_flat = y_pred.cpu().detach().numpy().flatten()
    y_true_flat = y_true.cpu().detach().numpy().flatten()
    
    y_pred_binary = (y_pred_flat >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_binary, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(y_true_flat, y_pred_flat)
    except:
        auc = 0.0
    
    return f1, recall, auc, precision

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(pos_weight_value, threshold_value, device, run_number, experiment_name):
    """Train a single model with given pos_weight and threshold."""
    
    print(f"\n{'='*70}")
    print(f"{experiment_name}: pos_weight={pos_weight_value}, threshold={threshold_value}, Run {run_number}/{RUNS_PER_VALUE}")
    print(f"{'='*70}")
    
    # Initialize model
    model = RNAFoldingCNN(input_channels=8).to(device)
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training metrics storage
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_f1_scores': [],
        'val_recall_scores': [],
        'val_auc_scores': [],
        'val_precision_scores': []
    }
    
    best_val_f1 = -1
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f'  Epoch {epoch+1}/{NUM_EPOCHS} [Train]', leave=False)
        for batch_x, batch_y in train_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_losses'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_f1 = []
        all_val_recall = []
        all_val_auc = []
        all_val_precision = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'  Epoch {epoch+1}/{NUM_EPOCHS} [Val]  ', leave=False)
            for batch_x, batch_y in val_bar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                f1, recall, auc, precision = calculate_metrics(outputs, batch_y, threshold_value)
                all_val_f1.append(f1)
                all_val_recall.append(recall)
                all_val_auc.append(auc)
                all_val_precision.append(precision)
                
                val_bar.set_postfix({'loss': f'{loss.item():.4f}', 'f1': f'{f1:.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_f1 = np.mean(all_val_f1)
        avg_val_recall = np.mean(all_val_recall)
        avg_val_auc = np.mean(all_val_auc)
        avg_val_precision = np.mean(all_val_precision)
        
        history['val_losses'].append(avg_val_loss)
        history['val_f1_scores'].append(avg_val_f1)
        history['val_recall_scores'].append(avg_val_recall)
        history['val_auc_scores'].append(avg_val_auc)
        history['val_precision_scores'].append(avg_val_precision)
        
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
        
        # Print epoch summary
        print(f"  Epoch {epoch+1:2d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"Val F1={avg_val_f1:.4f}, Val Recall={avg_val_recall:.4f}, Val AUC={avg_val_auc:.4f}")
        
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n  Training complete! Best Val F1: {best_val_f1:.4f}")
    
    return model, history, best_val_f1

# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_model(model, threshold_value, device):
    """Test the model and return metrics."""
    print(f"\n  Running test evaluation...")
    
    model.eval()
    test_loss = 0.0
    all_test_f1 = []
    all_test_recall = []
    all_test_auc = []
    all_test_precision = []
    
    # For counting pairs
    all_true_pairs = 0
    all_pred_pairs = 0
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='  Testing', leave=False)
        for batch_x, batch_y in test_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
            f1, recall, auc, precision = calculate_metrics(outputs, batch_y, threshold_value)
            all_test_f1.append(f1)
            all_test_recall.append(recall)
            all_test_auc.append(auc)
            all_test_precision.append(precision)
            
            # Count pairs
            all_true_pairs += batch_y.sum().item()
            pred_binary = (outputs >= threshold_value).float()
            all_pred_pairs += pred_binary.sum().item()
            
            test_bar.set_postfix({'f1': f'{f1:.4f}'})
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_f1 = np.mean(all_test_f1)
    avg_test_recall = np.mean(all_test_recall)
    avg_test_auc = np.mean(all_test_auc)
    avg_test_precision = np.mean(all_test_precision)
    
    print(f"  Test Results: F1={avg_test_f1:.4f}, Recall={avg_test_recall:.4f}, AUC={avg_test_auc:.4f}")
    print(f"  True pairs: {int(all_true_pairs)}, Predicted pairs: {int(all_pred_pairs)}")
    
    return {
        'test_loss': avg_test_loss,
        'test_f1': avg_test_f1,
        'test_recall': avg_test_recall,
        'test_auc': avg_test_auc,
        'test_precision': avg_test_precision,
        'true_pairs': int(all_true_pairs),
        'pred_pairs': int(all_pred_pairs)
    }

# ============================================================================
# EXPERIMENT 1: pos_weight SWEEP
# ============================================================================

def run_pos_weight_experiment(device, results_dir):
    """Run experiment 1: sweep pos_weight from 1 to 10 with fixed threshold=0.5"""
    
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT 1: pos_weight SWEEP (threshold={FIXED_THRESHOLD_EXP1})")
    print(f"{'#'*70}\n")
    
    all_runs = []
    
    for pos_weight_value in range(POS_WEIGHT_START, POS_WEIGHT_END + 1):
        
        print(f"\n{'='*70}")
        print(f"Testing pos_weight = {pos_weight_value}")
        print(f"{'='*70}")
        
        for run_num in range(1, RUNS_PER_VALUE + 1):
            
            # Train model
            model, history, best_val_f1 = train_model(
                pos_weight_value, 
                FIXED_THRESHOLD_EXP1, 
                device, 
                run_num,
                f"Exp1 pos_weight={pos_weight_value}"
            )
            
            # Test model
            test_result = test_model(model, FIXED_THRESHOLD_EXP1, device)
            
            # Store results
            result = {
                'experiment': 'pos_weight_sweep',
                'pos_weight': pos_weight_value,
                'threshold': FIXED_THRESHOLD_EXP1,
                'run_number': run_num,
                'train_losses': history['train_losses'],
                'val_losses': history['val_losses'],
                'val_f1_scores': history['val_f1_scores'],
                'val_recall_scores': history['val_recall_scores'],
                'val_auc_scores': history['val_auc_scores'],
                'val_precision_scores': history['val_precision_scores'],
                'best_val_f1': best_val_f1,
                'test_loss': test_result['test_loss'],
                'test_f1': test_result['test_f1'],
                'test_recall': test_result['test_recall'],
                'test_auc': test_result['test_auc'],
                'test_precision': test_result['test_precision'],
                'true_pairs': test_result['true_pairs'],
                'pred_pairs': test_result['pred_pairs']
            }
            
            all_runs.append(result)
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"\n  Completed run {run_num}/{RUNS_PER_VALUE}")
    
    # Save experiment 1 results
    exp1_path = os.path.join(results_dir, 'experiment1_pos_weight_sweep.json')
    with open(exp1_path, 'w') as f:
        json.dump(all_runs, f, indent=2)
    print(f"\n✓ Experiment 1 results saved to: {exp1_path}")
    
    return all_runs

# ============================================================================
# EXPERIMENT 2: THRESHOLD SWEEP
# ============================================================================

def run_threshold_experiment(device, results_dir):
    """Run experiment 2: sweep threshold from 0.3 to 0.7 with fixed pos_weight=5"""
    
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT 2: THRESHOLD SWEEP (pos_weight={FIXED_POS_WEIGHT_EXP2})")
    print(f"{'#'*70}\n")
    
    all_runs = []
    
    # Generate threshold values
    threshold_values = np.arange(THRESHOLD_START, THRESHOLD_END + THRESHOLD_STEP/2, THRESHOLD_STEP)
    threshold_values = [round(t, 2) for t in threshold_values]  # Round to avoid floating point issues
    
    for threshold_value in threshold_values:
        
        print(f"\n{'='*70}")
        print(f"Testing threshold = {threshold_value}")
        print(f"{'='*70}")
        
        for run_num in range(1, RUNS_PER_VALUE + 1):
            
            # Train model
            model, history, best_val_f1 = train_model(
                FIXED_POS_WEIGHT_EXP2, 
                threshold_value, 
                device, 
                run_num,
                f"Exp2 threshold={threshold_value}"
            )
            
            # Test model
            test_result = test_model(model, threshold_value, device)
            
            # Store results
            result = {
                'experiment': 'threshold_sweep',
                'pos_weight': FIXED_POS_WEIGHT_EXP2,
                'threshold': threshold_value,
                'run_number': run_num,
                'train_losses': history['train_losses'],
                'val_losses': history['val_losses'],
                'val_f1_scores': history['val_f1_scores'],
                'val_recall_scores': history['val_recall_scores'],
                'val_auc_scores': history['val_auc_scores'],
                'val_precision_scores': history['val_precision_scores'],
                'best_val_f1': best_val_f1,
                'test_loss': test_result['test_loss'],
                'test_f1': test_result['test_f1'],
                'test_recall': test_result['test_recall'],
                'test_auc': test_result['test_auc'],
                'test_precision': test_result['test_precision'],
                'true_pairs': test_result['true_pairs'],
                'pred_pairs': test_result['pred_pairs']
            }
            
            all_runs.append(result)
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"\n  Completed run {run_num}/{RUNS_PER_VALUE}")
    
    # Save experiment 2 results
    exp2_path = os.path.join(results_dir, 'experiment2_threshold_sweep.json')
    with open(exp2_path, 'w') as f:
        json.dump(all_runs, f, indent=2)
    print(f"\n✓ Experiment 2 results saved to: {exp2_path}")
    
    return all_runs

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("Results", f"dual_experiments_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}\n")
    
    # Save configuration
    config = {
        'num_epochs': NUM_EPOCHS,
        'max_len': MAX_LEN,
        'batch_size': BATCH_SIZE,
        'runs_per_value': RUNS_PER_VALUE,
        'experiment1': {
            'name': 'pos_weight_sweep',
            'pos_weight_range': [POS_WEIGHT_START, POS_WEIGHT_END],
            'fixed_threshold': FIXED_THRESHOLD_EXP1
        },
        'experiment2': {
            'name': 'threshold_sweep',
            'threshold_range': [THRESHOLD_START, THRESHOLD_END],
            'threshold_step': THRESHOLD_STEP,
            'fixed_pos_weight': FIXED_POS_WEIGHT_EXP2
        },
        'timestamp': timestamp
    }
    
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}\n")
    
    # Run experiments
    exp1_results = run_pos_weight_experiment(device, results_dir)
    exp2_results = run_threshold_experiment(device, results_dir)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Total runs: {len(exp1_results) + len(exp2_results)}")
    print(f"Results saved to: {results_dir}")
    print("="*70 + "\n")
