import subprocess
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

print("Step 1: Generating processed datasets")
data_proc_path = os.path.join(base_dir, 'processed_datasets', 'data_processing.py')
subprocess.run(['python', data_proc_path], check=True)

print("Step 2: Running experiments")
exp_path = os.path.join(base_dir, 'experiments.py')
subprocess.run(['python', exp_path], check=True)