"""Legacy module path retained for deserializing existing checkpoints."""

from problems.pdctrp.problem_pdctrp import PDCTRP, PDCTRPDataset

PDCVRP = PDCTRP
PDCVRPDataset = PDCTRPDataset

__all__ = ["PDCVRP", "PDCVRPDataset"]
