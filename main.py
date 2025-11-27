import subprocess

steps = [
    "src/get_data.py",
    "src/psd_proc.py"
]

for step in steps:
    print(f"Running {step} ...")
    subprocess.run(["python", step], check=True)
    print("--------------------------------")
