# Quick test script for RNA Folding EA
# Run this to validate your setup before running full experiments

import sys
import os
sys.path.append('src')

from rna_folding_ea import RNAFoldingEA

def test_basic_functionality():
    """Test basic EA functionality with small parameters"""
    print("=== TESTING BASIC FUNCTIONALITY ===")
    
    # Check Python dependencies first
    try:
        import numpy, matplotlib, seaborn, pandas, yaml
        print("✓ All Python dependencies available")
    except ImportError as e:
        print(f"✗ Missing Python dependencies: {e}")
        print("If using virtual environment, make sure to activate it first:")
        print("source venv/bin/activate")
        return False
    
    # Simple test constraints
    sequence_constraint = "NNNNNNNNNN"  # 10 bases, any nucleotide
    structure_constraint = "((((...))))"  # Simple hairpin
    
    print(f"Test constraint: {sequence_constraint}")
    print(f"Target structure: {structure_constraint}")
    
    # Small test EA
    ea = RNAFoldingEA(
        population_size=10,
        generations=3,
        sequence_constraint=sequence_constraint,
        structure_constraint=structure_constraint
    )
    
    # Test individual components
    print("\n1. Testing sequence generation...")
    seq = ea.generate_random_sequence()
    print(f"Generated sequence: {seq}")
    print(f"Valid: {ea.is_valid_sequence(seq)}")
    
    print("\n2. Testing population initialization...")
    ea.initialize_population()
    print(f"Population size: {len(ea.population)}")
    print(f"Sample individuals: {ea.population[:3]}")
    
    print("\n3. Testing IUPAC validation...")
    test_cases = [
        ("AUCGAUCGAU", True),   # Valid
        ("AUCGBUCGAU", False),  # Invalid character
        ("AUCG", False)         # Wrong length
    ]
    
    for seq, expected in test_cases:
        result = ea.is_valid_sequence(seq)
        status = "✓" if result == expected else "✗"
        print(f"{status} {seq}: {result} (expected: {expected})")
    
    print("\n4. Testing diversity calculation...")
    test_seqs = ["AUCGAUCGAU", "AUCGAUCGAG", "GCGCGCGCGC"]
    diversity = ea.calculate_diversity(test_seqs)
    print(f"Diversity of test sequences: {diversity:.3f}")
    
    print("\n✓ Basic functionality test completed!")

def test_docker_integration():
    """Test Docker and IPknot integration"""
    print("\n=== TESTING DOCKER INTEGRATION ===")
    
    try:
        import subprocess
        
        # Check if Docker is running
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if result.returncode != 0:
            print("✗ Docker is not running or not installed")
            return False
        
        print("✓ Docker is running")
        
        # Check if IPknot container exists
        result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=ipknot_runner'], 
                              capture_output=True, text=True)
        if 'ipknot_runner' in result.stdout:
            print("✓ IPknot container found")
        else:
            print("✗ IPknot container not found. Run setup_machine.sh first.")
            return False
        
        # Test a simple folding operation
        print("Testing RNA folding...")
        ea = RNAFoldingEA(10, 3, "NNNNNNNNNN", "((((...))))")
        test_seq = "GCGCAAAAGG"
        
        try:
            structure = ea.fold_rna_ipknot(test_seq)
            print(f"✓ Folding test successful: {test_seq} -> {structure}")
            return True
        except Exception as e:
            print(f"✗ Folding test failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Docker test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("RNA Folding EA - Quick Validation Test")
    print("=====================================")
    
    # Test basic functionality
    test_basic_functionality()
    
    # Test Docker integration
    docker_ok = test_docker_integration()
    
    print("\n=== TEST SUMMARY ===")
    print("✓ Basic EA functionality: PASSED")
    print(f"{'✓' if docker_ok else '✗'} Docker integration: {'PASSED' if docker_ok else 'FAILED'}")
    
    if docker_ok:
        print("\nAll tests passed! You're ready to run experiments.")
        print("\nNext steps:")
        print("1. Run: python3 src/rna_folding_ea.py")
        print("2. Or run device-optimized: python3 src/ea_runner.py config/device_experiments.yml --device odin")
    else:
        print("\n  Setup incomplete. Please run ./setup_machine.sh first.")

if __name__ == "__main__":
    main()