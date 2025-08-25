#!/usr/bin/env python3
"""
Template per creare manualmente i file JSON STS
Crea JSON nel formato esatto richiesto dalla sezione 2.4
"""

import json
import os

# =============================================================================
# MODIFICA QUESTI DATI CON I TUOI RISULTATI
# =============================================================================

# Per n=6 teams
n = 6

# La tua soluzione da MiniZinc
your_solution = [
    [[1, 2], [1, 3], [2, 6], [6, 3], [4, 5]],
    [[3, 4], [2, 5], [3, 5], [4, 2], [6, 1]],
    [[5, 6], [4, 6], [1, 4], [5, 1], [2, 3]],
]

# I tuoi risultati dei solver
results = {
    "gecode": {
        "time": 90,
        "optimal": True,
        "obj": 1,
        "sol": your_solution
    },
    "chuffed": {
        "time": 300,          # Esempio: timeout
        "optimal": False,
        "obj": None,
        "sol": None
    },
    "gecode_symbreak": {
        "time": 45,           # Esempio: con symmetry breaking più veloce
        "optimal": True,
        "obj": None,          # None per decision version
        "sol": your_solution
    }
}

# =============================================================================
# FORMATTAZIONE JSON CORRETTA
# =============================================================================

def format_solution_array(sol):
    """Formatta la soluzione nel formato compatto richiesto"""
    
    if sol is None:
        return "null"
    
    lines = ["    ["]
    
    for i, period in enumerate(sol):
        # Formatta ogni periodo su una riga
        games_str = ", ".join([f"[{game[0]}, {game[1]}]" for game in period])
        line = f"      [{games_str}]"
        
        # Aggiungi virgola tranne per l'ultimo periodo
        if i < len(sol) - 1:
            line += ","
        
        lines.append(line)
    
    lines.append("    ]")
    
    return "\n" + "\n".join(lines)

def create_json_manual():
    """Crea il file JSON manualmente nel formato esatto"""
    
    # Crea directory
    os.makedirs("res/CP", exist_ok=True)
    
    output_file = f"res/CP/{n}.json"
    
    # Scrivi JSON manualmente
    with open(output_file, 'w') as f:
        f.write("{\n")
        
        solver_entries = []
        
        for solver_name, data in results.items():
            # Formatta obj field
            obj_str = "null" if data["obj"] is None else str(data["obj"])
            
            # Formatta solution array
            sol_str = format_solution_array(data["sol"])
            
            # Crea entry per questo solver
            entry = f'  "{solver_name}": {{\n' + \
                   f'    "time": {data["time"]},\n' + \
                   f'    "optimal": {str(data["optimal"]).lower()},\n' + \
                   f'    "obj": {obj_str},\n' + \
                   f'    "sol": {sol_str}\n' + \
                   f'  }}'
            
            solver_entries.append(entry)
        
        # Unisci tutti i solver
        f.write(",\n".join(solver_entries))
        f.write("\n}")
    
    print(f"✅ File JSON creato: {output_file}")
    
    # Mostra il contenuto
    print(f"\n📄 Contenuto di {output_file}:")
    with open(output_file, 'r') as f:
        content = f.read()
        print(content)
    
    return output_file

def validate_format():
    """Valida che il formato sia corretto"""
    
    periods = n // 2
    weeks = n - 1
    
    print(f"🔍 Validazione per n={n}:")
    print(f"   Atteso: {periods} periods × {weeks} weeks")
    
    for solver_name, data in results.items():
        if data["sol"] is not None:
            solution = data["sol"]
            if len(solution) == periods and len(solution[0]) == weeks:
                print(f"✅ {solver_name}: formato corretto ({len(solution)}×{len(solution[0])})")
            else:
                print(f"❌ {solver_name}: formato sbagliato ({len(solution)}×{len(solution[0])})")
        else:
            print(f"⚠️  {solver_name}: nessuna soluzione")

def preview_expected_format():
    """Mostra come dovrebbe apparire il formato finale"""
    
    print("📋 FORMATO ATTESO:")
    print('''
{
  "gecode": {
    "time": 90,
    "optimal": true,
    "obj": 1,
    "sol": [
      [[1, 2], [1, 3], [2, 6], [6, 3], [4, 5]],
      [[3, 4], [2, 5], [3, 5], [4, 2], [6, 1]],
      [[5, 6], [4, 6], [1, 4], [5, 1], [2, 3]]
    ]
  }
}
    ''')

if __name__ == "__main__":
    print("STS Manual JSON Creator - Fixed Format")
    print("=" * 45)
    print("Crea JSON nel formato ESATTO richiesto dalla sezione 2.4")
    print()
    
    validate_format()
    print()
    
    preview_expected_format()
    print()
    
    create_json_manual()
    
    print(f"\n🎯 RISULTATO:")
    print("✅ Il JSON ora ha il formato compatto corretto!")
    print("✅ Ogni periodo su una riga singola")
    print("✅ Pronto per il checker check_solution.py")
    
    print(f"\n📁 File salvato: res/CP/{n}.json")