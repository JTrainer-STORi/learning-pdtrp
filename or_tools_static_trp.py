# Simple script for solving static TSP problems using OR-Tools, to refresh my basic knowledge of the library.

from problems.pdtrp.problem_pdtrp import PDTRP

from torch.utils.data import DataLoader
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import argparse
import numpy as np
import torch
import sys
from functools import partial
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description='OR-Tools Static TSP Solver')

parser.add_argument('--n_samples', type=int, default=1280,
                    help='The number of samples to generate')
parser.add_argument('--batch_size', type=int, default=128,
                    help='The batch size for the dataloader')
parser.add_argument('--min_total', type=int, default=40, 
                        help="The minimum number of total nodes")
parser.add_argument('--max_total', type=int, default=100, 
                    help="The maximum number of static nodes")
parser.add_argument('--speed', type=float, default=4.0,
                    help='The speed of the vehicle in units/h')
parser.add_argument('--n_subregions', type=int, default=9,
                        help='Number of subregions to generate the customers in. Passing 1 will generate customers uniformly in the unit square.')
parser.add_argument('--filename', type=str, default=None,
                    help='The filename to save the data to')
parser.add_argument('--seed', type=int, default=41394125,
                    help='The seed for the problem')

args = parser.parse_args()

dataset = PDTRP.make_dataset(num_samples=args.n_samples, min_total=args.min_total, max_total=args.max_total, min_dod=0.0, max_dod=0.0, speed=args.speed, n_subregions=args.n_subregions, seed=args.seed)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

