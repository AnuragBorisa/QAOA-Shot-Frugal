from __future__ import annotations
from typing import List,Tuple
import numpy as np
from qiskit import QuantumCircuit

Edge = Tuple[int,int]

def qaoa_circuit(
        n:int,
        edges:List[Edge],
        gammas : np.ndarray,
        betas : np.ndarray,
)->QuantumCircuit:
    
    assert gammas.shape == betas.shape

    p = len(gammas)
    qc = QuantumCircuit(n)

    qc.h(range(n))

    for l in range(p):
        gamma = gammas[l]
        beta = betas[l]

        for i,j in edges:
            qc.cx(i,j)
            qc.rz(2*gamma,j)
            qc.cx(i,j)
        
        for q in range(n):
            qc.rx(2*beta,q)
    
    qc.measure_all()
    return qc


