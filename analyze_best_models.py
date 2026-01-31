import json
import os
from pathlib import Path

def load_json_file(filepath):
    """Load JSON file and return as dictionary."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_model_name(params):
    """Create a concise, informative name from parameters."""
    # Map parameter names to short abbreviations
    abbrev_map = {
        'symmetry_break': 'SB',
        'implied_constraint': 'IC',
        'use_int_search': 'IS',
        'use_restart_luby': 'RL',
        'use_relax_and_reconstruct': 'RR',
        'chuffed': 'CH'
    }
    
    # For circle method, also include symm_brake (typo in original)
    if 'symm_brake' in params:
        abbrev_map['symm_brake'] = 'SB'
    if 'ic_diff_match_in_week' in params:
        abbrev_map['ic_diff_match_in_week'] = 'IC'
    
    # Get enabled features
    enabled = []
    for param, abbrev in abbrev_map.items():
        if param in params and params[param]:
            if abbrev not in enabled:  # Avoid duplicates
                enabled.append(abbrev)
    
    if not enabled:
        return "Baseline"
    
    return "+".join(enabled)

def find_best_model(entries, is_circle):
    """Find the best model from a list of entries."""
    if not entries:
        return None
    
    # Sort by time first, then by secondary time metric
    def sort_key(entry):
        time = entry.get('time', float('inf'))
        
        # Secondary metric depends on circle mode
        if is_circle:
            secondary = entry.get('n20_time', entry.get('solve_time', float('inf')))
        else:
            secondary = entry.get('cp_time', entry.get('solve_time', float('inf')))
        
        return (time, secondary)
    
    best_entry = min(entries, key=sort_key)
    return best_entry

def analyze_json_files():
    """Analyze all JSON files in res/CP directory."""
    res_dir = Path('res/CP')
    
    if not res_dir.exists():
        print(f"Directory {res_dir} does not exist!")
        return
    
    json_files = sorted(res_dir.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {res_dir}")
        return
    
    print("=" * 80)
    print("Best Models Analysis for CP Results")
    print("=" * 80)
    print("\nModel Name Legend:")
    print("  SB = Symmetry Breaking")
    print("  IC = Implied Constraints")
    print("  IS = Integer Search (first-fail)")
    print("  RL = Restart Luby")
    print("  RR = Relax & Reconstruct")
    print("  CH = Chuffed Solver")
    print("  Baseline = No optimization techniques enabled")
    print("=" * 80)
    
    for json_file in json_files:
        print(f"\nFile: {json_file.name}")
        print("-" * 80)
        
        data = load_json_file(json_file)
        
        # Group entries by circle mode
        circle_true_entries = []
        circle_false_entries = []
        
        # Iterate through all timestamp entries
        for timestamp, entry in data.items():
            if not isinstance(entry, dict):
                continue
            
            params = entry.get('params', {})
            
            # Only consider optimized solutions (check in params)
            if not params.get('optimized', False):
                continue
            
            # Check if this is circle mode
            is_circle = params.get('circle', False)
            
            if is_circle:
                circle_true_entries.append(entry)
            else:
                circle_false_entries.append(entry)
        
        # Find best models for each mode
        best_canonical = find_best_model(circle_false_entries, is_circle=False)
        best_circle = find_best_model(circle_true_entries, is_circle=True)
        
        # Print results for canonical method
        if best_canonical:
            model_name = create_model_name(best_canonical.get('params', {}))
            time = best_canonical.get('time', 'N/A')
            cp_time = best_canonical.get('cp_time', 'N/A')
            optimizer_time = best_canonical.get('optimizer_time', 'N/A')
            obj = best_canonical.get('obj', 'N/A')
            timeout = best_canonical.get('timeout_reached', False)
            
            print(f"\nBest Canonical Model: {model_name}")
            print(f"  Time: {time}s")
            print(f"  CP Time: {cp_time}s")
            print(f"  Optimizer Time: {optimizer_time}s")
            print(f"  Objective: {obj}")
            print(f"  Timeout: {timeout}")
            print(f"  Parameters: {best_canonical.get('params', {})}")
        else:
            print(f"\nBest Canonical Model: None found")
        
        # Print results for circle method
        if best_circle:
            model_name = create_model_name(best_circle.get('params', {}))
            time = best_circle.get('time', 'N/A')
            n20_time = best_circle.get('n20_time', best_circle.get('solve_time', 'N/A'))
            optimizer_time = best_circle.get('optimizer_time', 'N/A')
            obj = best_circle.get('obj', 'N/A')
            timeout = best_circle.get('timeout_reached', False)
            
            print(f"\nBest Circle Model: {model_name}")
            print(f"  Time: {time}s")
            print(f"  N20 Time: {n20_time}s")
            print(f"  Optimizer Time: {optimizer_time}s")
            print(f"  Objective: {obj}")
            print(f"  Timeout: {timeout}")
            print(f"  Parameters: {best_circle.get('params', {})}")
        else:
            print(f"\nBest Circle Model: None found")
    
    print("\n" + "=" * 80)

def parse_model_name_to_params(model_name, is_circle):
    """Parse a model name (e.g., 'SB+IC+IS') back to parameter dictionary."""
    abbrev_to_param = {
        'SB': 'symmetry_break' if not is_circle else 'symm_brake',
        'IC': 'implied_constraint' if not is_circle else 'ic_diff_match_in_week',
        'IS': 'use_int_search',
        'RL': 'use_restart_luby',
        'RR': 'use_relax_and_reconstruct',
        'CH': 'chuffed'
    }
    
    # Start with all parameters as False
    params = {
        'use_int_search': False,
        'use_restart_luby': False,
        'use_relax_and_reconstruct': False,
        'chuffed': False,
        'circle': is_circle,
        'optimized': True
    }
    
    if is_circle:
        params['symm_brake'] = False
        params['ic_diff_match_in_week'] = False
    else:
        params['symmetry_break'] = False
        params['implied_constraint'] = False
    
    # Handle Baseline (all False)
    if model_name.upper() == "BASELINE":
        return params
    
    # Parse the model name
    codes = [code.strip().upper() for code in model_name.split('+')]
    
    for code in codes:
        if code in abbrev_to_param:
            param_name = abbrev_to_param[code]
            params[param_name] = True
    
    return params

def params_match(entry_params, target_params):
    """Check if entry parameters match target parameters (ignoring circle and optimized)."""
    # Get all relevant parameter keys (excluding circle and optimized)
    keys_to_check = [k for k in target_params.keys() if k not in ['circle', 'optimized']]
    
    for key in keys_to_check:
        if entry_params.get(key) != target_params.get(key):
            return False
    
    return True

def analyze_specific_model():
    """Analyze a specific model across all n values."""
    res_dir = Path('res/CP')
    
    if not res_dir.exists():
        print(f"Directory {res_dir} does not exist!")
        return
    
    json_files = sorted(res_dir.glob('*.json'))
    
    # Ask user for model details
    print("\n" + "=" * 80)
    print("Analyze Specific Model")
    print("=" * 80)
    
    is_circle_str = input("\nIs this a Circle method model? (yes/no): ").strip().lower()
    is_circle = is_circle_str in ['yes', 'y', 'true']
    
    model_name = input("Enter model name (e.g., 'SB+IC+IS' or 'Baseline'): ").strip()
    
    # Parse model name to parameters
    target_params = parse_model_name_to_params(model_name, is_circle)
    
    print("\n" + "=" * 80)
    print(f"Results for {'Circle' if is_circle else 'Canonical'} Model: {model_name}")
    print("=" * 80)
    print(f"Target Parameters: {target_params}")
    print("-" * 80)
    
    # Collect results across all files
    results = []
    
    for json_file in json_files:
        data = load_json_file(json_file)
        
        # Extract n from filename
        n_value = json_file.stem
        
        # Search for matching entry
        for timestamp, entry in data.items():
            if not isinstance(entry, dict):
                continue
            
            entry_params = entry.get('params', {})
            
            # Check if parameters match
            if params_match(entry_params, target_params):
                results.append({
                    'n': entry.get('n', n_value),
                    'time': entry.get('time', 'N/A'),
                    'cp_time': entry.get('cp_time', 'N/A'),
                    'n20_time': entry.get('n20_time', 'N/A'),
                    'optimizer_time': entry.get('optimizer_time', 'N/A'),
                    'obj': entry.get('obj', 'N/A'),
                    'optimal': entry.get('optimal', 'N/A'),
                    'timeout': entry.get('timeout_reached', False)
                })
                break  # Found match for this n, move to next file
    
    # Display results
    if not results:
        print("No matching results found!")
        return
    
    print(f"\nFound {len(results)} results:\n")
    
    for result in results:
        print(f"n = {result['n']}")
        print(f"  Total Time: {result['time']}s")
        if is_circle:
            print(f"  N20 Time: {result['n20_time']}s")
        else:
            print(f"  CP Time: {result['cp_time']}s")
        print(f"  Optimizer Time: {result['optimizer_time']}s")
        print(f"  Objective: {result['obj']}")
        print(f"  Optimal: {result['optimal']}")
        print(f"  Timeout: {result['timeout']}")
        print()
    
    print("=" * 80)

if __name__ == "__main__":
    analyze_json_files()
    
    # Ask if user wants to analyze a specific model
    while True:
        choice = input("\nDo you want to analyze a specific model? (yes/no): ").strip().lower()
        if choice in ['yes', 'y']:
            analyze_specific_model()
        else:
            break
