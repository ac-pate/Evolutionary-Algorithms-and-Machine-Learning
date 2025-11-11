# Running Assignment 2 on Remote GPU Server (Odin)

## Server Details
- Host: odin@192.168.0.131
- Password: admin1
- GPU: RTX 5070
- Batch Size: 32 (optimized for your GPU)

---

## Option 1: SSH + Command Line (Recommended)

### Step 1: Connect to Server
```powershell
ssh odin@192.168.0.131
# Enter password: admin1
```

### Step 2: Upload Files
From your local machine, upload the necessary files:
```powershell
# Upload the notebook
scp "d:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\COEN432_Assignment2.ipynb" odin@192.168.0.131:~/assignment2/

# Upload the data files
scp "d:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\TR0.csv" odin@192.168.0.131:~/assignment2/
scp "d:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\VL0.csv" odin@192.168.0.131:~/assignment2/
scp "d:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\TS0.csv" odin@192.168.0.131:~/assignment2/
```

### Step 3: Convert Notebook to Python Script
On the server:
```bash
cd ~/assignment2
jupyter nbconvert --to script COEN432_Assignment2.ipynb
```

### Step 4: Run the Script
```bash
python COEN432_Assignment2.py
```

---

## Option 2: Remote Jupyter Server

### Step 1: SSH with Port Forwarding
```powershell
ssh -L 8888:localhost:8888 odin@192.168.0.131
```

### Step 2: Start Jupyter on Remote Server
```bash
cd ~/assignment2
jupyter notebook --no-browser --port=8888
```

### Step 3: Access from Local Browser
1. Copy the URL with token from terminal (looks like: http://localhost:8888/?token=...)
2. Paste in your local browser
3. Navigate to the notebook and run cells

---

## Option 3: VS Code Remote SSH (Easiest)

### Step 1: Install VS Code Extension
- Install "Remote - SSH" extension in VS Code

### Step 2: Configure SSH Connection
1. Press `F1` or `Ctrl+Shift+P`
2. Type "Remote-SSH: Connect to Host"
3. Enter: `odin@192.168.0.131`
4. Enter password: `admin1`

### Step 3: Open Folder
1. File > Open Folder
2. Navigate to where you'll upload files
3. Create folder: `assignment2`

### Step 4: Upload Files
- Drag and drop:
  - COEN432_Assignment2.ipynb
  - TR0.csv
  - VL0.csv
  - TS0.csv

### Step 5: Run Notebook
- Open the notebook in VS Code
- Select Python kernel
- Run cells normally

---

## Quick SCP Commands (Copy-Paste Ready)

```powershell
# Create directory on remote
ssh odin@192.168.0.131 "mkdir -p ~/assignment2"

# Upload all files at once
scp "D:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\COEN432_Assignment2.ipynb" odin@192.168.0.131:~/assignment2/
scp "D:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\TR0.csv" odin@192.168.0.131:~/assignment2/
scp "D:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\VL0.csv" odin@192.168.0.131:~/assignment2/
scp "D:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\TS0.csv" odin@192.168.0.131:~/assignment2/
```

---

## Verify GPU is Working

After connecting to the server, run:
```bash
# Check NVIDIA GPU
nvidia-smi

# In Python/Jupyter
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5070
```

---

## Expected Performance with RTX 5070 and Batch Size 32

- Training time per epoch: ~1-2 minutes
- Total training time (20 epochs): ~20-40 minutes
- Memory usage: ~2-4 GB VRAM
- CPU usage: Minimal (GPU handles computation)

---

## Download Results Back to Local Machine

After training completes:
```powershell
# Download model
scp odin@192.168.0.131:~/assignment2/best_model.pth "D:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\"

# Download plots
scp odin@192.168.0.131:~/assignment2/training_metrics.png "D:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\"
scp odin@192.168.0.131:~/assignment2/contact_map_comparison.png "D:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\"

# Download notebook with outputs
scp odin@192.168.0.131:~/assignment2/COEN432_Assignment2.ipynb "D:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\"
```

---

## Troubleshooting

### If SSH connection fails:
```powershell
# Test connection
ping 192.168.0.131

# Verify SSH service is running on server
# (May need to enable SSH on the remote machine)
```

### If GPU not detected:
```bash
# Check NVIDIA drivers
nvidia-smi

# Check PyTorch CUDA installation
python -c "import torch; print(torch.version.cuda)"
```

### If out of memory with batch size 32:
Edit the notebook and reduce:
```python
BATCH_SIZE = 16  # or 24
```

### If packages missing on server:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib scikit-learn tqdm jupyter
```

---

## Automated Script for Full Workflow

Save this as `run_on_odin.ps1`:

```powershell
# Configuration
$REMOTE_USER = "odin"
$REMOTE_HOST = "192.168.0.131"
$REMOTE_DIR = "~/assignment2"
$LOCAL_DIR = "D:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2"

Write-Host "Creating remote directory..."
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

Write-Host "Uploading files..."
scp "$LOCAL_DIR\COEN432_Assignment2.ipynb" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
scp "$LOCAL_DIR\TR0.csv" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
scp "$LOCAL_DIR\VL0.csv" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
scp "$LOCAL_DIR\TS0.csv" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

Write-Host "Converting notebook to script..."
ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_DIR && jupyter nbconvert --to script COEN432_Assignment2.ipynb"

Write-Host "Running training..."
ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_DIR && python COEN432_Assignment2.py"

Write-Host "Downloading results..."
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/best_model.pth" "$LOCAL_DIR\"
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/training_metrics.png" "$LOCAL_DIR\"
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/contact_map_comparison.png" "$LOCAL_DIR\"

Write-Host "Done!"
```

Run with:
```powershell
powershell -ExecutionPolicy Bypass -File run_on_odin.ps1
```

---

## Notes

1. **Batch Size Changed**: Updated from 16 to 32 to take advantage of RTX 5070
2. **Expected Training Time**: Much faster than CPU, approximately 20-40 minutes total
3. **Memory**: RTX 5070 has sufficient VRAM for this task
4. **Security**: Consider using SSH keys instead of password for repeated access

---

## Quick Start (Simplest Method)

1. Open PowerShell in Windows
2. Run these commands one by one:

```powershell
# Upload files
scp "D:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\*.csv" odin@192.168.0.131:~/assignment2/
scp "D:\Desktop\Classes\Fall_2025\Evolutionary-Algorithms-and-Machine-Learning\Assignment_2\COEN432_Assignment2.ipynb" odin@192.168.0.131:~/assignment2/

# Connect and run
ssh odin@192.168.0.131
# Enter password: admin1

# Then on the server:
cd ~/assignment2
jupyter nbconvert --to script COEN432_Assignment2.ipynb
python COEN432_Assignment2.py
```

3. Wait for training to complete (20-40 minutes)
4. Download results (see "Download Results" section above)
