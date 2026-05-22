import sys
import json
from pathlib import Path

def compare(path1, path2):
    with open(path1, "r") as f:
        data1 = json.load(f)
    with open(path2, "r") as f:
        data2 = json.load(f)
        
    metrics1 = data1.get("metrics", data1)
    metrics2 = data2.get("metrics", data2)
    
    print(f"{'Métrica':<20} | {'Baseline':<10} | {'Optimizado':<10} | {'Cambio':<10} | {'Veredicto'}")
    print("-" * 75)
    
    improvements = 0
    new_failures = len(set(data2.get("failed_case_ids", [])) - set(data1.get("failed_case_ids", [])))
    
    for metric in metrics1:
        v1 = metrics1[metric]
        v2 = metrics2.get(metric, 0)
        diff = v2 - v1
        
        # For error_rate, lower is better
        if metric == "error_rate":
            better = diff < 0
            symbol = "✅" if better else "❌"
        else:
            better = diff > 0
            symbol = "✅" if better else "❌"
            
        if better:
            improvements += 1
            
        print(f"{metric:<20} | {v1:<10.4f} | {v2:<10.4f} | {diff:<+10.4f} | {symbol}")
        
    print("-" * 75)
    if improvements >= 2 and new_failures == 0:
        print("\nOPTIMIZATION SUCCESSFUL ✅")
    else:
        print("\nOPTIMIZATION FAILED or INSUFFICIENT ❌")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python compare_evaluations.py <path_baseline> <path_optimized>")
    else:
        compare(sys.argv[1], sys.argv[2])
