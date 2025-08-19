import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import subprocess
from cycler import cycler

import torch

# Check PyTorch version
print("PyTorch version:", torch.__version__)

# Check CUDA version (PyTorch's compiled CUDA version)
print("CUDA version:", torch.version.cuda)

# Check if GPU is available
print("Is GPU available?", torch.cuda.is_available())

# If GPU is available, get the GPU name
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

plt.rcParams['axes.prop_cycle'] = cycler('color', plt.colormaps['tab20'].colors)
workspace = ""

algorithms = [
    "TUCB", "TTS", "UCB", "TS", "MINI", "TuRBO", "SnAKe", "RS", "LHS", "GS", "LA"
    ]

# Below is copy and pasted from plot.ipynb
def plot_from_files():
    workspace = ""
    desired_order = [
        "GS", "LA", "LHS", "MINI", "RS", "SnAKe", "TS", "TTS", "TUCB", "TuRBO", "UCB"
    ]

    print("--- Generating Regret Plot ---")

    # Get all files from the directory
    all_files_reg = glob.glob(os.path.join(workspace, '*_reg.xlsx'))
    print(f"Found {len(all_files_reg)} regret files.")

    # Create a new list of files sorted according to desired_order
    reordered_files_reg = []
    for algo_name in desired_order:
        expected_file = os.path.join(workspace, f"{algo_name}_reg.xlsx")
        if expected_file in all_files_reg:
            reordered_files_reg.append(expected_file)

    # Plot using the reordered list
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for file in reordered_files_reg:
        df = pd.read_excel(file)[:200]
        cumulative_regret = df.cumsum()
        t = cumulative_regret.index + 1
        average_regret = cumulative_regret.div(t, axis=0)
        mean = 2.95 - average_regret.mean(axis=1)
        std = average_regret.std(axis=1) / np.sqrt(10)
        
        # Extract name and plot with the 'label' keyword
        algorithm_name = os.path.basename(file).replace('_reg.xlsx', '')
        ax1.plot(mean, label=algorithm_name)
        ax1.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    # Format plot
    ax1.set_xlabel('Round (t)', fontsize=16)
    ax1.set_ylabel('Log Average Cumulative Regret log$(R_t/t)$', fontsize=16)
    ax1.legend(fontsize=14, loc='lower left')
    ax1.set_xlim(0, 200)
    ax1.grid(True)
    ax1.set_yscale("log")

    for tick in ax1.get_yticklabels(which='both'):
        tick.set_rotation(90)
        tick.set_ha('center')
        tick.set_va('center')
        
    plt.tight_layout()
    plt.savefig(os.path.join("Figures", os.path.basename(workspace.rstrip('/\\')) + "_regret.pdf"), dpi=600)

    print("\n--- Generating Travel Cost Plot ---")

    # Get all travel files
    all_files_travel = glob.glob(os.path.join(workspace, '*_travel.xlsx'))
    print(f"Found {len(all_files_travel)} travel files.")

    # Create a reordered list for the travel files
    reordered_files_travel = []
    for algo_name in desired_order:
        expected_file = os.path.join(workspace, f"{algo_name}_travel.xlsx")
        if expected_file in all_files_travel:
            reordered_files_travel.append(expected_file)

    # Plot using the reordered list
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for file in reordered_files_travel:
        df = pd.read_excel(file)[:200]
        cumulative_regret = df.cumsum()
        t = cumulative_regret.index + 1
        average_regret = cumulative_regret.div(t, axis=0)
        mean = average_regret.mean(axis=1)
        std = average_regret.std(axis=1) / np.sqrt(10)

        # Extract name and plot with the 'label' keyword
        algorithm_name = os.path.basename(file).replace('_travel.xlsx', '')
        ax2.plot(mean, label=algorithm_name)
        ax2.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    # Format plot
    ax2.set_xlabel('Round (t)', fontsize=16)
    ax2.set_ylabel('Average Movement Cost $C_t/t$', fontsize=16)
    ax2.legend(fontsize=14, loc='lower left')
    ax2.set_xlim(0, 200)
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("Figures", os.path.basename(workspace.rstrip('/\\')) + "_cost.pdf"), dpi=600)

# Create the "Figures" directory if it doesn't exist
os.makedirs("Figures", exist_ok=True)

print("Running simulation scripts...")
for alg in algorithms:
    # Construct the script path safely
    script_path = os.path.join(workspace, f"{alg}.py")
    
    # Check if the script exists before trying to run it
    if os.path.exists(script_path):
        # Use subprocess.run() to execute the script
        # ["python", script_path] is the command as you'd type it in the terminal
        import sys
        subprocess.run([sys.executable, script_path], check=True, text=True)
    else:
        print(f"Warning: Script not found at {script_path}")

    plot_from_files()

print("Simulations finished.")
