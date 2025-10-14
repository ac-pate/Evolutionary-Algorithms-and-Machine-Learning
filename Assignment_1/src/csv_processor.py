#!/usr/bin/env python3
"""
CSV Processor for RNA Folding EA
Handles reading input CSV and writing output CSV files
Achal Patel - 40227663
"""

import csv
import os
import pandas as pd
from typing import List, Dict, Tuple

# Add project root to path for config import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class CSVProcessor:
    """Handles CSV input/output for the RNA Folding EA assignment"""
    
    def __init__(self, input_file: str = None, output_file: str = None):
        """
        Initialize CSV processor
        
        Args:
            input_file: Path to input CSV file (default from config)
            output_file: Path to output CSV file (default from config)
        """
        self.input_file = input_file or config.INPUT_CSV_FILE
        self.output_file = output_file or config.OUTPUT_CSV_FILE
        
    def read_problem_instances(self) -> List[Dict]:
        """
        Read problem instances from CSV file
        
        Returns:
            List of dictionaries containing problem instances
            Each dict has: {'id': str, 'structure': str, 'iupac': str}
        """
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input CSV file not found: {self.input_file}")
        
        problem_instances = []
        
        try:
            with open(self.input_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Validate required columns exist
                required_columns = [config.CSV_ID_COLUMN, config.CSV_STRUCTURE_COLUMN, config.CSV_IUPAC_COLUMN]
                if not all(col in reader.fieldnames for col in required_columns):
                    raise ValueError(f"CSV file missing required columns. Expected: {required_columns}, Found: {reader.fieldnames}")
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 because row 1 is header
                    try:
                        # Extract problem instance data
                        problem_id = row[config.CSV_ID_COLUMN].strip()
                        structure = row[config.CSV_STRUCTURE_COLUMN].strip()
                        iupac = row[config.CSV_IUPAC_COLUMN].strip()
                        
                        # Validate data
                        if not problem_id:
                            raise ValueError(f"Row {row_num}: Empty problem ID")
                        if not structure:
                            raise ValueError(f"Row {row_num}: Empty structure")
                        if not iupac:
                            raise ValueError(f"Row {row_num}: Empty IUPAC constraint")
                        if len(structure) != len(iupac):
                            raise ValueError(f"Row {row_num}: Structure length ({len(structure)}) != IUPAC length ({len(iupac)})")
                        
                        problem_instances.append({
                            'id': problem_id,
                            'structure': structure,
                            'iupac': iupac
                        })
                        
                    except Exception as e:
                        print(f"Warning: Skipping row {row_num} due to error: {e}")
                        continue
                        
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file {self.input_file}: {e}")
        
        if not problem_instances:
            raise ValueError(f"No valid problem instances found in {self.input_file}")
        
        print(f"Successfully loaded {len(problem_instances)} problem instances from {self.input_file}")
        return problem_instances
    
    def write_results(self, results: List[Dict]) -> None:
        """
        Write results to output CSV file in assignment format
        
        Args:
            results: List of result dictionaries
                Each dict should have: {'id': str, 'sequences': List[str], 'fitness_scores': List[float]}
        """
        if not results:
            print("Warning: No results to write")
            return
        
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            with open(self.output_file, 'w', newline='') as csvfile:
                # Create fieldnames for assignment format: id,result_1,result_2,result_3,result_4,result_5
                fieldnames = ['id', 'result_1', 'result_2', 'result_3', 'result_4', 'result_5']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Collect all problem IDs and ensure we have all 6 problems
                all_problem_ids = ['1.1', '1.2', '2.1', '2.2', '3.1', '3.2']
                results_by_id = {result['id']: result for result in results}
                
                # Write results for each problem
                for problem_id in all_problem_ids:
                    row_data = {'id': problem_id}
                    
                    if problem_id in results_by_id:
                        result = results_by_id[problem_id]
                        sequences = result.get('sequences', [])
                        
                        # Fill in up to 5 results
                        for i in range(5):
                            if i < len(sequences):
                                row_data[f'result_{i+1}'] = sequences[i]
                            else:
                                row_data[f'result_{i+1}'] = ''
                    else:
                        # No results for this problem - fill with empty strings
                        for i in range(5):
                            row_data[f'result_{i+1}'] = ''
                    
                    writer.writerow(row_data)
                        
        except Exception as e:
            raise RuntimeError(f"Error writing CSV file {self.output_file}: {e}")
        
        print(f"Results successfully written to {self.output_file}")
    
    def validate_problem_instance(self, problem: Dict) -> Tuple[bool, str]:
        """
        Validate a single problem instance
        
        Args:
            problem: Problem instance dictionary
            
        Returns:
            (is_valid, error_message)
        """
        try:
            structure = problem['structure']
            iupac = problem['iupac']
            
            # Check lengths match
            if len(structure) != len(iupac):
                return False, f"Structure and IUPAC lengths don't match ({len(structure)} vs {len(iupac)})"
            
            # Check valid structure characters (including bracket notation for pseudoknots)
            valid_structure_chars = set('().[{}]')
            if not all(c in valid_structure_chars for c in structure):
                return False, f"Invalid characters in structure: {set(structure) - valid_structure_chars}"
            
            # Check valid IUPAC characters
            valid_iupac_chars = set('AUGCRYSWKMBDHVNAUGC')
            if not all(c in valid_iupac_chars for c in iupac.upper()):
                return False, f"Invalid characters in IUPAC: {set(iupac.upper()) - valid_iupac_chars}"
            
            # Check balanced parentheses and brackets
            paren_balance = 0
            bracket_balance = 0
            for c in structure:
                if c == '(':
                    paren_balance += 1
                elif c == ')':
                    paren_balance -= 1
                    if paren_balance < 0:
                        return False, "Unbalanced parentheses in structure"
                elif c == '[':
                    bracket_balance += 1
                elif c == ']':
                    bracket_balance -= 1
                    if bracket_balance < 0:
                        return False, "Unbalanced brackets in structure"
            
            if paren_balance != 0:
                return False, "Unbalanced parentheses in structure"
            if bracket_balance != 0:
                return False, "Unbalanced brackets in structure"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {e}"

def main():
    """Test the CSV processor"""
    processor = CSVProcessor()
    
    # Test reading (if file exists)
    if os.path.exists(processor.input_file):
        try:
            problems = processor.read_problem_instances()
            print(f"Successfully read {len(problems)} problem instances")
            for problem in problems:
                valid, error = processor.validate_problem_instance(problem)
                print(f"Problem {problem['id']}: {'Valid' if valid else f'Invalid - {error}'}")
        except Exception as e:
            print(f"Error reading problems: {e}")
    else:
        print(f"Input file {processor.input_file} not found. Create it to test CSV reading.")

if __name__ == "__main__":
    main()