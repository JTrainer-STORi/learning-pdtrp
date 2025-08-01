from problems import PDTRP, PDCVRP, PDTRPTW, PDCVRPTW
from problems import PDTRP, PDCVRP, PDTRPTW, PDCVRPTW
from problems import PDTRP, PDCVRP, PDTRPTW, PDCVRPTW

from torch.utils.data import DataLoader

from utils import load_model, move_to
from utils.data_utils import dataset_to_input
from train import set_decode_type
import torch 
from tqdm import tqdm
import argparse
import numpy as np
import time
import os

parser = argparse.ArgumentParser(description="Test a model on a dataset")

parser.add_argument("--problem", type=str, default="pdtrp", help="Problem to test on")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU for testing")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model to test")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for testing")
parser.add_argument('--model_mode', type=str, default='recursive', choices=['recursive', 'masked'], help="Encoding mode for the model")

args = parser.parse_args()

use_cuda = args.use_gpu and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model, model_args = load_model(args.model_path)
model.to(device)
set_decode_type(model, "greedy")
model.eval()

# SET SEEDS FOR REPRODUCIBILITY
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.problem == "pdtrp":
    problem = PDTRP()
elif args.problem == "pdtrptw":
    problem = PDTRPTW()
elif args.problem == "pdcvrp":
    problem = PDCVRP()
elif args.problem == "pdcvrptw":
    problem = PDCVRPTW()

ablation_datasets = ['new_data/pdtrp/ablation_testset_20total_0.5dod.txt', 'new_data/pdtrp/ablation_testset_50total_0.2dod.txt', 'new_data/pdtrp/ablation_testset_50total_0.5dod.txt', 'new_data/pdtrp/ablation_testset_50total_0.8dod.txt', 'new_data/pdtrp/ablation_testset_100total_0.5dod.txt', 'new_data/pdtrp/ablation_testset_100total_0.8dod.txt', 'new_data/pdtrp/ablation_testset_200total_0.5dod.txt', 'new_data/pdtrp/ablation_testset_200total_0.8dod.txt']

model_name = args.model_path.split("/")[-1]

for dataset_path in ablation_datasets:
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    dataset = problem.make_dataset(filename=dataset_path, 
                                batch_size=args.batch_size)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    run_times = []
    costs = []

    for bat in tqdm(dataloader):
        input = dataset_to_input(bat, problem.NAME, device)
        start_time = time.perf_counter()
        cost, log_likelihood, info = model(input, mode=args.model_mode)
        cost = cost.cpu().numpy()
        costs.append(cost[0])
        end_time = time.perf_counter()
        run_times.append(end_time - start_time)

    os.makedirs("runtime_stats/", exist_ok=True)

    with open("runtime_stats/" + model_name + dataset_name + "_run_times.txt", "w") as f:
        f.write("Run times for each sample:\n")
        for i, run_time in enumerate(run_times):
            f.write(f"Sample {i+1}: {run_time:.4f} seconds\n")
        f.write(f"Average run time: {np.mean(run_times):.4f} seconds\n")
        f.write(f"Std dev of run times: {np.std(run_times):.4f} seconds\n")

    os.makedirs("test_results/", exist_ok=True)

    with open("test_results/" + model_name + dataset_name + "_results.txt", "w") as f:
        f.write("Test results:\n")
        f.write(f"Problem: {args.problem}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Test Dataset Path: {dataset_path}\n")
        f.write(f"Seed: {args.seed}\n")
        for i, cost in enumerate(costs):
            f.write(f"Sample {i+1}: Cost = {cost}\n")
        f.write(f"Average Cost: {np.mean(costs)}\n")
        f.write(f"Std dev of Costs: {np.std(costs)}\n")
        