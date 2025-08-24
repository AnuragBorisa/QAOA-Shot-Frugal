from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from qiskit_aer.primitives import Sampler as AerSampler

from qaoa.maxcut import max_cut_hamiltonian,expectation_from_counts
from qaoa.circuits import qaoa_circuit
from qaoa.spsa import ShotFrugalSPSA,default_seed_schedule,default_shots_schedule
from qaoa.metrics import brute_force_maxcut, bitstring_to_cut_value, approx_ratio


def make_aer_sampler(shots:int,seed:int,method:str="density_matrix")->AerSampler:

    return AerSampler(
        backend_options={"method":method,"seed_simulator":seed},
        run_options={"shots":shots}
    )

def evaluate_params(params:np.ndarray,n:int,edges,sampler:AerSampler,H):

    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    qc = qaoa_circuit(n,edges,gammas,betas)
    result = sampler.run(qc).result()
    counts = result.quasi_dists[0]
    val = -expectation_from_counts(counts, H) 

    best_str = max(counts,key=counts.get) if len(counts)>0 else "0"*n
    return val , {"counts":counts,"best_str":best_str}

def make_eval_current(n,edges,H,method:str="density_matrix"):

    def _eval(params:np.ndarray,k:int):
        shots = max(256,default_shots_schedule(k)//2)
        seed = default_seed_schedule(k) + 991
        sampler = make_aer_sampler(shots,seed,method)
        return evaluate_params(params,n,edges,sampler,H)
    return _eval

def objective_pair(
        theta_plus:np.ndarray,
        theta_minus:np.ndarray,
        k:int,
        *,
        n:int,
        edges,
        H,
        shots_schedule=default_shots_schedule,
        seed_schedule=default_seed_schedule,
        method:str = "density_matrix",
        **kwargs,
)->tuple[float,float,dict,dict]:
    p = len(theta_plus) // 2 
    gammas_p , betas_p = theta_plus[:p] , theta_plus[p:]
    gammas_n , betas_n = theta_minus[:p] ,theta_minus[p:]

    qcp = qaoa_circuit(n,edges,gammas_p,betas_p)
    qcm = qaoa_circuit(n,edges,gammas_n,betas_n)

    shots = shots_schedule(k)
    seed = seed_schedule(k)

    sampler = make_aer_sampler(shots,seed,method)

    res = sampler.run([qcp,qcm]).result()
    counts_p = res.quasi_dists[0]
    counts_m = res.quasi_dists[1]

    fp = -expectation_from_counts(counts_p,H)
    fm = -expectation_from_counts(counts_m,H)

    auxp = {"counts": counts_p, "shots": shots, "seed": seed}
    auxm = {"counts": counts_m, "shots": shots, "seed": seed}

    return fp,fm,auxp,auxm

def display_str(k, n: int) -> str:
    s = k if isinstance(k, str) else format(int(k), f"0{n}b")
    return s[::-1]

def main():
    n = 6
    G = nx.random_regular_graph(d=3, n=n, seed=42)
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    edges = [(min(i, j), max(i, j)) for (i, j) in G.edges()]

    H = max_cut_hamiltonian(edges,n)

    opt_cut = brute_force_maxcut(G)
    print(f"Brute-force optimum cut: {opt_cut}")

    p = 4

    x0 = np.concatenate([np.full(p,0.8),np.full(p,0.4)])

    spsa = ShotFrugalSPSA(a=0.2,c=0.1,A=10.0,alpha=0.602,gamma=0.101,seed=0)

    obj_kwargs = dict (
        n = n,
        edges=edges,
        H=H,
        shots_schedule = default_shots_schedule,
        seed_schedule = default_seed_schedule,
        method="density_matrix",
        eval_current = make_eval_current(n,edges,H,method="density_matrix")
    )

    best_params , info = spsa.minimise_pair(objective_pair,x0,maxiter=150,**obj_kwargs)
    values = info["history"]["value"]
    values_plot = [-v for v in values] 

    final_sampler = make_aer_sampler(shots=4096, seed=777, method="density_matrix")
    final_val, final_aux = evaluate_params(best_params, n, edges, final_sampler, H)
    raw_best = final_aux["best_str"]                  
    best_str_pretty = display_str(raw_best, n) 
    # best_str = final_aux["best_str"]
    best_cut = bitstring_to_cut_value(raw_best, edges)
    ratio = approx_ratio(best_cut, opt_cut)
    
    print(f"Best objective (-E[cut]): {final_val:.4f}")
    print(f"Best bitstring: {best_str_pretty}  cut={best_cut}  approx ratio={ratio:.3f}")
    est_Ecut = -final_val
    print(f"Estimated E[cut]: {est_Ecut:.3f}")

    plt.plot(values_plot, marker=".")
    plt.xlabel("Iteration")
    plt.ylabel("E[cut] (higher is better)")
    plt.axhline(opt_cut, ls="--", lw=1, alpha=0.6, label=f"opt cut = {opt_cut}")
    plt.legend()
    plt.title(f"QAOA p={p} on 3-regular n={n} (Shot-frugal SPSA)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    