import subprocess

steps = [
    #"src/ObtainData.py",    # To get the data, uncomment (not necessary with the processed data)
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
