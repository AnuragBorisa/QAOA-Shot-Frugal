from __future__ import annotations
from typing import List, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qaoa.circuits import Edge, qaoa_stateprep_circuit


def plus_statevector(n: int) -> np.ndarray:
   
    qc = QuantumCircuit(n)
    qc.h(range(n))
    sv = Statevector.from_instruction(qc)
    return np.asarray(sv.data)


def qaoa_statevector(
    n: int,
    edges: List[Edge],
    gammas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    
    qc = qaoa_stateprep_circuit(n, edges, gammas, betas)
    sv = Statevector.from_instruction(qc)
    return np.asarray(sv.data)


def overlap(psi_ref: np.ndarray, psi: np.ndarray):
   
    amp = np.vdot(psi_ref, psi)  # conjugate dot product
    F = float(np.abs(amp) ** 2)
    return F, amp


def geometric_fraction(psi_ref: np.ndarray, psi: np.ndarray):
    
    F, amp = overlap(psi_ref, psi)
    fg = 1.0 - F
    return fg, F, amp


def fs_step(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
   
    F_ab, _ = overlap(psi_a, psi_b)
    F_ab = float(np.clip(F_ab, 0.0, 1.0))
    return float(2.0 * np.arccos(np.sqrt(F_ab)))


def path_length(states: List[np.ndarray]):
    
    if len(states) < 2:
        return 0.0, []
    steps = [fs_step(states[k], states[k + 1]) for k in range(len(states) - 1)]
    return float(np.sum(steps)), steps


def circuitousness(states: List[np.ndarray]):
    
    L, steps = path_length(states)
    if len(states) < 2:
        return L, 0.0, 0.0, 0.0, steps

    F_0T, _ = overlap(states[0], states[-1])
    F_0T = float(np.clip(F_0T, 0.0, 1.0))
    D = float(2.0 * np.arccos(np.sqrt(F_0T)))

    if D < 1e-9:
        C_add = L
        C_ratio = 0.0
    else:
        C_add = L - D
        C_ratio = L / D

    return L, D, C_add, C_ratio, steps
