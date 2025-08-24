from __future__ import annotations
from typing import List, Tuple,Union
import numpy as np
import networkx as nx

Edge = Tuple[int, int]

def bitstring_to_cut_value(bitstr: Union[str, int], edges: List[Edge]) -> int:
   
    n = max(max(i, j) for (i, j) in edges) + 1
    s = bitstr if isinstance(bitstr, str) else format(int(bitstr), f"0{n}b")
    s = s[::-1]
    return sum(1 for (i, j) in edges if s[i] != s[j])

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
