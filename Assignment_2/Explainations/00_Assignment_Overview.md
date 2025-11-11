# Assignment 2: RNA Secondary Structure Prediction - Overview

## 📋 Assignment Goal

You're building a **deep learning model to predict RNA secondary structure** from RNA sequences.

### Simple Explanation
- **Input**: A string of RNA bases (A, U, G, C) - like "AUGCGAU..."
- **Output**: Which bases pair with each other (contact map) - a 2D grid showing connections

This is fundamentally different from image classification because you're predicting **relationships between positions** in a sequence, not just a category.

---

## 📥 Assignment Instructions

### 1. Download Files
Download from Google Drive:
- Jupyter Notebook: `Assignment2.ipynb`
- Dataset files: `TR0.csv`, `VL0.csv`, `TS0.csv`

### 2. Complete the Notebook
Open in Google Colab (recommended) or local environment. Complete all sections marked with **TODO**:

**Code Tasks:**
- Data encoding
- CNN model building
- Training loop
- Result visualization

**Analysis Tasks:**
- Model architecture explanation (Part 2.2)
- Performance analysis (Part 4.2)
- Discussion and limitations (Part 4.3)

### 3. Run and Save
- Ensure all code runs successfully
- **Recommended**: Click "Runtime" → "Restart and run all" before submitting
- All output must be clearly visible

---

## 📤 Submission Guidelines

### Required Files (2 total)

#### 1. Jupyter Notebook
- **File name**: `Assignment2_[Student_number].ipynb`
- Must include all code and written answers

#### 2. PDF File
- **File name**: `Assignment2_[Student_number].pdf`
- **How to generate**: 
  1. Run all code in notebook
  2. Use browser's print function (Ctrl+P or Cmd+P)
  3. Select "Save as PDF"

⚠️ **Important**: PDF must include:
- All code
- All written answers
- All output (especially graphs in Part 3.3 and contact diagrams in Part 4.2)

---

## 🎯 Core Learning Objectives

1. **Understand bioinformatics data formats**: FASTA and dot-bracket notation
2. **Master data encoding**: One-hot encoding and Contact Maps
3. **Design a 2D CNN**: For non-image tasks
4. **Build ML pipeline**: Training, validation, and testing
5. **Use proper metrics**: F1-score for imbalanced datasets
6. **Visualize and analyze**: Model predictions critically

---

## 📊 Assignment Structure

| Part | Focus | Weight | Key Tasks |
|------|-------|--------|-----------|
| **Part 1** | Data Preprocessing | 35% | Load data, encode sequences, create contact maps |
| **Part 2** | Model Implementation | 20% | Build CNN architecture |
| **Part 3** | Training & Evaluation | 25% | Train model, track metrics, visualize results |
| **Part 4** | Analysis & Reporting | 20% | Evaluate on test set, analyze performance |

---

## ⏱️ Expected Time Investment

| Experience Level | Estimated Time |
|-----------------|----------------|
| **First-time ML/PyTorch** | 10-16 hours |
| **Some ML experience** | 6-10 hours |
| **Comfortable with PyTorch** | 4-6 hours |

---

## 🎓 Important Notes

- **Focus**: This is about learning the ML pipeline, not building state-of-the-art models
- **Expected Performance**: Low F1 scores (0.1-0.3) are normal for this simple CNN
- **High AUC, Low F1**: This is expected due to imbalanced data (you'll analyze why)
- **No Perfect Solutions**: Understanding matters more than perfect metrics
