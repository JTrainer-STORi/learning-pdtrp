from problems import PDTRP, PDCVRP, PDTRPTW, PDCVRPTW

from torch.utils.data import DataLoader

from utils import load_problem, move_to
from train import set_decode_type
import torch 
from tqdm import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Test a model on a dataset")

parser.add_argument("--problem", type=str, default="pdtrp", help="Problem to test on")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU for testing")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model to test")
parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to the dataset to test on")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

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

dataset = problem.load_dataset(filename=args.test_dataset_path,)