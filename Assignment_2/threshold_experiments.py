"""
Threshold Experiments Script
Runs multiple training runs with different pos_weight values (5 to 15)
For each pos_weight, runs 3 experiments and keeps the best one
Generates comparison plots for training and test metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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
# GLOBAL CONFIGURATION (SAME AS NOTEBOOK)
# ============================================================================
num_epochs = 20
MAX_LEN = 128
BATCH_SIZE = 128
THRESHOLD = 0.5
POS_WEIGHT_START = 2
POS_WEIGHT_END = 15
RUNS_PER_WEIGHT = 3

print("="*70)
print("EXPERIMENT CONFIGURATION")
print("="*70)
print(f"Epochs per run: {num_epochs}")
print(f"Max Length: {MAX_LEN}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Classification Threshold: {THRESHOLD}")
print(f"pos_weight range: {POS_WEIGHT_START} to {POS_WEIGHT_END}")
print(f"Runs per pos_weight: {RUNS_PER_WEIGHT}")
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
# MODEL DEFINITION (SAME AS NOTEBOOK)
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

def calculate_metrics(y_pred, y_true, threshold=THRESHOLD):
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
    
    return f1, recall, auc

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(pos_weight_value, device, run_number):
    """Train a single model with given pos_weight."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING: pos_weight={pos_weight_value}, Run {run_number}/{RUNS_PER_WEIGHT}")
    print(f"{'='*70}")
    
    # Initialize model
    model = RNAFoldingCNN(input_channels=8).to(device)
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_f1_scores = []
    val_recall_scores = []
    val_auc_scores = []
    best_val_f1 = -1
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f'  Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
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
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_f1 = []
        all_val_recall = []
        all_val_auc = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'  Epoch {epoch+1}/{num_epochs} [Val]  ', leave=False)
            for batch_x, batch_y in val_bar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                f1, recall, auc = calculate_metrics(outputs, batch_y)
                all_val_f1.append(f1)
                all_val_recall.append(recall)
                all_val_auc.append(auc)
                
                val_bar.set_postfix({'loss': f'{loss.item():.4f}', 'f1': f'{f1:.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_f1 = np.mean(all_val_f1)
        avg_val_recall = np.mean(all_val_recall)
        avg_val_auc = np.mean(all_val_auc)
        
        val_losses.append(avg_val_loss)
        val_f1_scores.append(avg_val_f1)
        val_recall_scores.append(avg_val_recall)
        val_auc_scores.append(avg_val_auc)
        
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
        
        # Print epoch summary
        print(f"  Epoch {epoch+1:2d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"Val F1={avg_val_f1:.4f}, Val AUC={avg_val_auc:.4f}")
        
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n  Training complete! Best Val F1: {best_val_f1:.4f}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1_scores': val_f1_scores,
        'val_recall_scores': val_recall_scores,
        'val_auc_scores': val_auc_scores,
        'best_val_f1': best_val_f1
    }

# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_model(model, device):
    """Test the model and return metrics."""
    print(f"\n  Running test evaluation...")
    
    model.eval()
    test_loss = 0.0
    all_test_f1 = []
    all_test_recall = []
    all_test_auc = []
    
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
            
            f1, recall, auc = calculate_metrics(outputs, batch_y)
            all_test_f1.append(f1)
            all_test_recall.append(recall)
            all_test_auc.append(auc)
            
            # Count pairs
            all_true_pairs += batch_y.sum().item()
            pred_binary = (outputs >= THRESHOLD).float()
            all_pred_pairs += pred_binary.sum().item()
            
            test_bar.set_postfix({'f1': f'{f1:.4f}'})
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_f1 = np.mean(all_test_f1)
    avg_test_recall = np.mean(all_test_recall)
    avg_test_auc = np.mean(all_test_auc)
    
    print(f"  Test Results: F1={avg_test_f1:.4f}, Recall={avg_test_recall:.4f}, AUC={avg_test_auc:.4f}")
    print(f"  True pairs: {int(all_true_pairs)}, Predicted pairs: {int(all_pred_pairs)}")
    
    return {
        'test_loss': avg_test_loss,
        'test_f1': avg_test_f1,
        'test_recall': avg_test_recall,
        'test_auc': avg_test_auc,
        'true_pairs': int(all_true_pairs),
        'pred_pairs': int(all_pred_pairs)
    }

# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

def run_experiments():
    """Run all experiments and collect results."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("Results", f"threshold_experiments_batch_s_{BATCH_SIZE}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}\n")
    
    # Store all results
    all_results = []
    
    # Loop through pos_weight values
    for pos_weight_value in range(POS_WEIGHT_START, POS_WEIGHT_END + 1):
        
        print(f"\n{'#'*70}")
        print(f"# STARTING EXPERIMENTS FOR pos_weight = {pos_weight_value}")
        print(f"{'#'*70}")
        
        best_run_f1 = -1
        best_run_result = None
        
        # Run multiple experiments for this pos_weight
        for run_num in range(1, RUNS_PER_WEIGHT + 1):
            
            # Train model
            train_result = train_model(pos_weight_value, device, run_num)
            
            # Test model
            test_result = test_model(train_result['model'], device)
            
            # Combine results
            result = {
                'pos_weight': pos_weight_value,
                'run_number': run_num,
                'train_losses': train_result['train_losses'],
                'val_losses': train_result['val_losses'],
                'val_f1_scores': train_result['val_f1_scores'],
                'val_recall_scores': train_result['val_recall_scores'],
                'val_auc_scores': train_result['val_auc_scores'],
                'best_val_f1': train_result['best_val_f1'],
                'test_loss': test_result['test_loss'],
                'test_f1': test_result['test_f1'],
                'test_recall': test_result['test_recall'],
                'test_auc': test_result['test_auc'],
                'true_pairs': test_result['true_pairs'],
                'pred_pairs': test_result['pred_pairs']
            }
            
            # Check if this is the best run for this pos_weight
            if train_result['best_val_f1'] > best_run_f1:
                best_run_f1 = train_result['best_val_f1']
                best_run_result = result
                print(f"\n  ✓ New best run for pos_weight={pos_weight_value}! Val F1={best_run_f1:.4f}")
            
            # Clean up model
            del train_result['model']
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Store the best run for this pos_weight
        all_results.append(best_run_result)
        
        print(f"\n{'#'*70}")
        print(f"# COMPLETED pos_weight = {pos_weight_value}")
        print(f"# Best Val F1: {best_run_f1:.4f}, Test F1: {best_run_result['test_f1']:.4f}")
        print(f"{'#'*70}\n")
    
    return all_results, results_dir

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_over_epochs(all_results, results_dir):
    """Create comparison plots for training metrics over epochs."""
    
    print("\nGenerating training over epochs comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    pos_weights = [r['pos_weight'] for r in all_results]
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for result in all_results:
        epochs = list(range(1, len(result['train_losses']) + 1))
        ax.plot(epochs, result['train_losses'], marker='o', 
                label=f"pos_weight={result['pos_weight']}", linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation F1 Score
    ax = axes[0, 1]
    for result in all_results:
        epochs = list(range(1, len(result['val_f1_scores']) + 1))
        ax.plot(epochs, result['val_f1_scores'], marker='s', 
                label=f"pos_weight={result['pos_weight']}", linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation F1 Score', fontsize=12)
    ax.set_title('Validation F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation Recall
    ax = axes[1, 0]
    for result in all_results:
        epochs = list(range(1, len(result['val_recall_scores']) + 1))
        ax.plot(epochs, result['val_recall_scores'], marker='^', 
                label=f"pos_weight={result['pos_weight']}", linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Recall', fontsize=12)
    ax.set_title('Validation Recall Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Validation AUC
    ax = axes[1, 1]
    for result in all_results:
        epochs = list(range(1, len(result['val_auc_scores']) + 1))
        ax.plot(epochs, result['val_auc_scores'], marker='d', 
                label=f"pos_weight={result['pos_weight']}", linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title('Validation AUC Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_over_epochs.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: training_over_epochs.png")
    plt.close()

def plot_train_comparison(all_results, results_dir):
    """Create comparison plots for validation metrics (F1, Recall, AUC) vs pos_weight."""
    
    print("Generating validation metrics vs pos_weight comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    pos_weights = [r['pos_weight'] for r in all_results]
    
    # Get final validation metrics (last epoch values)
    val_f1s = [r['val_f1_scores'][-1] for r in all_results]
    val_recalls = [r['val_recall_scores'][-1] for r in all_results]
    val_aucs = [r['val_auc_scores'][-1] for r in all_results]
    best_val_f1s = [r['best_val_f1'] for r in all_results]
    
    # Plot 1: Validation F1 Score
    ax = axes[0, 0]
    ax.plot(pos_weights, val_f1s, marker='o', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('pos_weight Value', fontsize=12)
    ax.set_ylabel('Validation F1 Score', fontsize=12)
    ax.set_title('Validation F1 Score vs pos_weight', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pos_weights)
    for i, (x, y) in enumerate(zip(pos_weights, val_f1s)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Validation Recall
    ax = axes[0, 1]
    ax.plot(pos_weights, val_recalls, marker='s', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('pos_weight Value', fontsize=12)
    ax.set_ylabel('Validation Recall', fontsize=12)
    ax.set_title('Validation Recall vs pos_weight', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pos_weights)
    for i, (x, y) in enumerate(zip(pos_weights, val_recalls)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 3: Validation AUC
    ax = axes[1, 0]
    ax.plot(pos_weights, val_aucs, marker='^', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('pos_weight Value', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title('Validation AUC vs pos_weight', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pos_weights)
    for i, (x, y) in enumerate(zip(pos_weights, val_aucs)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 4: Best Validation F1 (across all epochs)
    ax = axes[1, 1]
    ax.plot(pos_weights, best_val_f1s, marker='d', linewidth=2, markersize=8, color='darkgreen')
    ax.set_xlabel('pos_weight Value', fontsize=12)
    ax.set_ylabel('Best Validation F1', fontsize=12)
    ax.set_title('Best Validation F1 vs pos_weight', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pos_weights)
    for i, (x, y) in enumerate(zip(pos_weights, best_val_f1s)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # Highlight the best pos_weight
    best_idx = best_val_f1s.index(max(best_val_f1s))
    best_pos_weight = pos_weights[best_idx]
    best_f1_value = best_val_f1s[best_idx]
    ax.plot(best_pos_weight, best_f1_value, marker='*', markersize=20, 
            color='red', zorder=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'validation_metrics_vs_weight.png'), 
                dpi=150, bbox_inches='tight')
    print(f"  Saved: validation_metrics_vs_weight.png")
    plt.close()

def plot_test_comparison(all_results, results_dir):
    """Create comparison plots for test metrics vs pos_weight."""
    
    print("Generating test metrics vs pos_weight comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    pos_weights = [r['pos_weight'] for r in all_results]
    test_f1s = [r['test_f1'] for r in all_results]
    test_recalls = [r['test_recall'] for r in all_results]
    test_aucs = [r['test_auc'] for r in all_results]
    true_pairs = [r['true_pairs'] for r in all_results]
    pred_pairs = [r['pred_pairs'] for r in all_results]
    
    # Plot 1: Test F1 Score
    ax = axes[0, 0]
    ax.plot(pos_weights, test_f1s, marker='o', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('pos_weight Value', fontsize=12)
    ax.set_ylabel('Test F1 Score', fontsize=12)
    ax.set_title('Test F1 Score vs pos_weight', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pos_weights)
    for i, (x, y) in enumerate(zip(pos_weights, test_f1s)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Test Recall
    ax = axes[0, 1]
    ax.plot(pos_weights, test_recalls, marker='s', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('pos_weight Value', fontsize=12)
    ax.set_ylabel('Test Recall', fontsize=12)
    ax.set_title('Test Recall vs pos_weight', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pos_weights)
    for i, (x, y) in enumerate(zip(pos_weights, test_recalls)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 3: Test AUC
    ax = axes[1, 0]
    ax.plot(pos_weights, test_aucs, marker='^', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('pos_weight Value', fontsize=12)
    ax.set_ylabel('Test AUC', fontsize=12)
    ax.set_title('Test AUC vs pos_weight', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pos_weights)
    for i, (x, y) in enumerate(zip(pos_weights, test_aucs)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 4: True vs Predicted Pairs
    ax = axes[1, 1]
    width = 0.35
    x = np.arange(len(pos_weights))
    ax.bar(x - width/2, true_pairs, width, label='True Pairs', color='blue', alpha=0.7)
    ax.bar(x + width/2, pred_pairs, width, label='Predicted Pairs', color='red', alpha=0.7)
    ax.set_xlabel('pos_weight Value', fontsize=12)
    ax.set_ylabel('Number of Pairs', fontsize=12)
    ax.set_title('True vs Predicted Pairs', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pos_weights)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'test_metrics_vs_weight.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: test_metrics_vs_weight.png")
    plt.close()

def save_results_summary(all_results, results_dir):
    """Save results summary to JSON and CSV."""
    
    print("Saving results summary...")
    
    # Save detailed JSON
    json_path = os.path.join(results_dir, 'detailed_results.json')
    with open(json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in all_results:
            r_copy = r.copy()
            for key in ['train_losses', 'val_losses', 'val_f1_scores', 
                       'val_recall_scores', 'val_auc_scores']:
                if key in r_copy:
                    r_copy[key] = [float(x) for x in r_copy[key]]
            json_results.append(r_copy)
        json.dump(json_results, f, indent=2)
    print(f"  Saved: detailed_results.json")
    
    # Save summary CSV
    summary_data = {
        'pos_weight': [r['pos_weight'] for r in all_results],
        'best_val_f1': [r['best_val_f1'] for r in all_results],
        'test_f1': [r['test_f1'] for r in all_results],
        'test_recall': [r['test_recall'] for r in all_results],
        'test_auc': [r['test_auc'] for r in all_results],
        'true_pairs': [r['true_pairs'] for r in all_results],
        'pred_pairs': [r['pred_pairs'] for r in all_results]
    }
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(results_dir, 'summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: summary.csv")
    
    # Print summary table
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING THRESHOLD EXPERIMENTS")
    print("="*70)
    
    # Run all experiments
    all_results, results_dir = run_experiments()
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    plot_training_over_epochs(all_results, results_dir)
    plot_train_comparison(all_results, results_dir)
    plot_test_comparison(all_results, results_dir)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    save_results_summary(all_results, results_dir)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {results_dir}")
    print("="*70 + "\n")
