import glob
import numpy as np

# read in the testing files from the HEC and let's start generating some plots

# start with a helper function for reading from the file

def read_results_file(filepath):
    results = {
        "problem": None,
        "model_path": None,
        "test_dataset_path": None,
        "seed": None,
        "device": None,
        "samples": [],
        "average_cost": None,
        "std_cost": None,
    }

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Problem:"):
                results["problem"] = line.split("Problem:")[1].strip()
            elif line.startswith("Model Path:"):
                results["model_path"] = line.split("Model Path:")[1].strip()
            elif line.startswith("Test Dataset Path:"):
                results["test_dataset_path"] = line.split("Test Dataset Path:")[1].strip()
            elif line.startswith("Seed:"):
                results["seed"] = int(line.split("Seed:")[1].strip())
            elif line.startswith("Device:"):
                results["device"] = line.split("Device:")[1].strip()
            elif line.startswith("Sample"):
                parts = line.split("Cost =")
                idx_part = parts[0].split()[1].strip(":")
                cost = float(parts[1].strip())
                results["samples"].append((int(idx_part), cost))
            elif line.startswith("Average Cost:"):
                results["average_cost"] = float(line.split("Average Cost:")[1].strip())
            elif line.startswith("Std dev of Costs:"):
                results["std_cost"] = float(line.split("Std dev of Costs:")[1].strip())

    return results

# let's compare results for pdtrp first

# Specify the test file roots you want to compare
test_folders = ['Joshi/recursive', '30-70/recursive', 'dynamic/recursive', 'pomo/recursive']
ablation_roots = ['pdtrp20', 'pdtrp50', 'pdtrp100', 'pdtrp200']

# Read in the test results for each root
all_test_results = []

for root in test_folders:
    for ablation_root in ablation_roots:
        files = glob.glob(f'test_results/pdtrp/{root}/{ablation_root}*')
        results = [read_results_file(f) for f in files]
        all_test_results.append(results)

# Extract all unique test set names
all_names = set()
for results in all_test_results:
    all_names.update(
        res['test_dataset_path'].split('/')[-1].replace('ablation_testset_', '').replace('.txt', '').replace('v4th8_ablation_ortools', '')
        for res in results
    )
all_names = sorted(all_names)

# Gather average costs for each root, aligned by test set name
costs_by_root = []
for results in all_test_results:
    names = [res['test_dataset_path'].split('/')[-1].replace('ablation_testset_', '').replace('.txt', '').replace('v4th8_ablation_ortools', '') for res in results]
    avg_costs = [res['average_cost'] for res in results]
    costs = [avg_costs[names.index(name)] if name in names else np.nan for name in all_names]
    costs_by_root.append(costs)

# Plot grouped bar chart
import matplotlib.pyplot as plt

x = np.arange(len(all_names))
width = 0.8 / len(test_folders)

plt.figure(figsize=(14, 6))
bars = []
for i, (costs, label) in enumerate(zip(costs_by_root, test_folders)):
    bar = plt.bar(x + (i - len(test_folders)/2 + 0.5) * width, costs, width, label=label)
    bars.append(bar)

plt.xlabel('Test Set')
plt.ylabel('Average Cost')
plt.title('Average Cost per Test Set (Grouped by Model)')
plt.xticks(x, all_names, rotation=45, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()

# Annotate bars with values
for bar_group, costs in zip(bars, costs_by_root):
    for bar, cost in zip(bar_group, costs):
        if not np.isnan(cost):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{cost:.2f}',
                     ha='center', va='bottom', fontsize=9)
            
                        
# save plot as svg
plt.savefig('pdtrp_ablation_comparison_training.svg', format='svg')

plt.show()