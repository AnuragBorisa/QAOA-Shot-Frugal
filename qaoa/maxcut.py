from __future__ import annotations
from typing import List , Dict , Tuple
from qiskit.quantum_info import SparsePauliOp
import numpy as np

Edge = Tuple[int,int]

def max_cut_hamiltonian(edges:List[Edge],n:int)->SparsePauliOp:
    terms = []

    for i,j in edges:
        lables = ["I"] * n
        lables[i] = "Z"
        lables[j] = "Z"
        pauli = "".join(lables[::-1])
        terms.append((pauli,-0.5))
    
    const = 0.5 * len(edges)
    terms.append(("I"*n,const))
    return SparsePauliOp.from_list(terms)



