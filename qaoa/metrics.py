from __future__ import annotations
from typing import List, Tuple
import numpy as np
import networkx as nx

Edge = Tuple[int, int]

def bitstring_to_cut_value(bitstr: str, edges: List[Edge]) -> int:
    return sum(1 for (i,j) in edges if bitstr[i] != bitstr[j])

def approx_ratio(best_cut: int, optimum_cut: int) -> float:
    if optimum_cut <= 0:
        return 0.0
    return best_cut / float(optimum_cut)

def brute_force_maxcut(G: nx.Graph) -> int:
    n = G.number_of_nodes()
    edges = list(G.edges())
    best = 0
    for mask in range(1 << n):
        s = format(mask, f"0{n}b")
        cut = sum(1 for (i,j) in edges if s[i] != s[j])
        if cut > best:
            best = cut
    return best
