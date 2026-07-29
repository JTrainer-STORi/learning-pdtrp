"""Legacy module path retained for deserializing existing checkpoints."""

from problems.pdctrp_tw.problem_pdctrptw import PDCTRPTW, PDCTRPTWDataset

PDCVRPTW = PDCTRPTW
PDCVRPTWDataset = PDCTRPTWDataset

__all__ = ["PDCVRPTW", "PDCVRPTWDataset"]
