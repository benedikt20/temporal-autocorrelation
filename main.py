import subprocess

steps = [
    "src/psd_slope_example.py",
    "src/get_data.py",
    "src/psd_proc.py",
    "src/landcover_proc.py",
    "src/corr_matrices.py"
]

for step in steps:
    print(f"Running {step} ...")
    subprocess.run(["python", step], check=True)
    print("--------------------------------")
