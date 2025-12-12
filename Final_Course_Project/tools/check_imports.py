"""
Script para comprobar qué módulos importan los archivos del proyecto
Ejecuta: `python tools/check_imports.py`
Devuelve una lista de módulos que no se pueden importar.
"""
modules = [
    "pandas",
    "numpy",
    "sklearn",
    "joblib",
    "matplotlib",
    "scipy",
    "seaborn",
    "xgboost",
    "smogn",
]

missing = []
for m in modules:
    try:
        __import__(m)
        print(f"OK: {m}")
    except Exception as e:
        print(f"MISSING: {m} -> {e.__class__.__name__}: {e}")
        missing.append(m)

if missing:
    print("\nResumen: módulos faltantes:")
    for m in missing:
        print(f" - {m}")
else:
    print("\nTodos los módulos comprobados están instalados.")
