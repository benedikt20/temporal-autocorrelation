import subprocess

steps = [
    "src/ObtainData.py",
    "src/PrepData_psd.py",
    "src/PrepData_landcover.py",
    "src/PreVisualizations.py",
    "src/ModelingVisualizations.py",
    "src/PostVisualizations.py"
]

for step in steps:
    print(f"Running {step} ...")
    subprocess.run(["python", step], check=True)
    print("--------------------------------")
