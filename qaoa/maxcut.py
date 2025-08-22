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

def expectation_from_counts(counts:Dict[str,float],H:SparsePauliOp)->float:
    n = H.num_qubits
    total = sum(counts.values())
    if total == 0:
        return 0.0
    exp = 0.0
    H_terms = H.to_list()
    for bitstr , weight in counts.items():

        s = bitstr if isinstance(bitstr,str) else format(bitstr, f"0{n}b")

        E_x=0.0
        
        z = [1 if s[q]== "0" else -1 for q in range(n)]
        for pauli , coeff in H_terms:
            eig = 1 
            for q , p in enumerate(pauli[::-1]):
                if p == "Z":
                    eig *= z[q]
            E_x += coeff * eig
        
        exp += (weight/total) * E_x
    
    return float(exp)