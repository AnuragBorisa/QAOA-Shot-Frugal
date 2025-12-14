from __future__ import annotations
import argparse
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from qiskit_aer.primitives import Sampler as AerSampler

from qaoa.maxcut import max_cut_hamiltonian, expectation_from_counts
from qaoa.circuits import qaoa_circuit
from qaoa.spsa import ShotFrugalSPSA, default_seed_schedule, default_shots_schedule
from qaoa.metrics import brute_force_maxcut, bitstring_to_cut_value, approx_ratio
from qaoa.geometry import plus_statevector, qaoa_statevector, geometric_fraction, circuitousness


def make_aer_sampler(shots: int, seed: int, method: str = "density_matrix") -> AerSampler:
    return AerSampler(
        backend_options={"method": method, "seed_simulator": seed},
        run_options={"shots": shots},
    )


def evaluate_params(params: np.ndarray, n: int, edges, sampler: AerSampler, H):
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]
    qc = qaoa_circuit(n, edges, gammas, betas)
    result = sampler.run(qc).result()
    counts = result.quasi_dists[0]
    val = -expectation_from_counts(counts, H)
    best_str = max(counts, key=counts.get) if len(counts) > 0 else "0" * n
    return val, {"counts": counts, "best_str": best_str}


def make_eval_current(n, edges, H, shots_schedule, seed_schedule, method: str = "density_matrix"):
    def _eval(params: np.ndarray, k: int):
        s = int(shots_schedule(k))
        shots = max(256, s // 2)
        seed = int(seed_schedule(k) + 991)
        sampler = make_aer_sampler(shots, seed, method)
        val, aux = evaluate_params(params, n, edges, sampler, H)
        aux["shots"] = shots
        aux["seed"] = seed
        return val, aux
    return _eval


def objective_pair(
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    k: int,
    *,
    n: int,
    edges,
    H,
    shots_schedule=default_shots_schedule,
    seed_schedule=default_seed_schedule,
    method: str = "density_matrix",
    **kwargs,
) -> tuple[float, float, dict, dict]:
    p = len(theta_plus) // 2
    gammas_p, betas_p = theta_plus[:p], theta_plus[p:]
    gammas_m, betas_m = theta_minus[:p], theta_minus[p:]

    qcp = qaoa_circuit(n, edges, gammas_p, betas_p)
    qcm = qaoa_circuit(n, edges, gammas_m, betas_m)

    shots = int(shots_schedule(k))
    seed = int(seed_schedule(k))

    sampler = make_aer_sampler(shots, seed, method)
    res = sampler.run([qcp, qcm]).result()
    counts_p = res.quasi_dists[0]
    counts_m = res.quasi_dists[1]

    fp = -expectation_from_counts(counts_p, H)
    fm = -expectation_from_counts(counts_m, H)

    auxp = {"counts": counts_p, "shots": shots, "seed": seed}
    auxm = {"counts": counts_m, "shots": shots, "seed": seed}

    return fp, fm, auxp, auxm


def display_str(k, n: int) -> str:
    s = k if isinstance(k, str) else format(int(k), f"0{n}b")
    return s[::-1]


def fixed_shots_schedule_factory(shots: int):
    def _sched(k: int) -> int:
        return int(shots)
    return _sched


def make_geometry_aware_shots_schedule(state: dict, base: int = 256, growth: float = 0.5, cap: int = 4096):
    def _sched(k: int) -> int:
        fg = float(state["fg"])
        baseline = int(min(cap, base * ((1 + k) ** growth)))

        if fg < 0.90:
            return int(max(128, min(512, baseline)))
        if fg < 0.97:
            return int(max(128, min(1024, baseline)))
        return int(max(128, baseline))
    return _sched


def compute_total_shots_from_history_aux(history_aux: list[dict]) -> int:
    total = 0
    for aux in history_aux:
        shots = int(aux["auxp"]["shots"])
        total += 2 * shots
        total += max(256, shots // 2)
    return int(total)


def run_one(
    *,
    n: int,
    d: int,
    p: int,
    maxiter: int,
    graph_seed: int,
    spsa_seed: int,
    shots_schedule,
    seed_schedule,
    method: str,
    do_plots: bool,
    tag: str,
    fg_state: dict | None = None,
):
    G = nx.random_regular_graph(d=d, n=n, seed=int(graph_seed))
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    edges = [(min(i, j), max(i, j)) for (i, j) in G.edges()]
    H = max_cut_hamiltonian(edges, n)
    opt_cut = brute_force_maxcut(G)

    x0 = np.concatenate([np.full(p, 0.8), np.full(p, 0.4)])
    spsa = ShotFrugalSPSA(a=0.2, c=0.1, A=10.0, alpha=0.602, gamma=0.101, seed=int(spsa_seed))

    psi0 = plus_statevector(n)
    state = fg_state if fg_state is not None else {"fg": 0.0}
    psi_x0 = qaoa_statevector(n, edges, x0[:p], x0[p:])
    fg0, _, _ = geometric_fraction(psi0, psi_x0)
    state["fg"] = float(fg0)

    eval_current = make_eval_current(n, edges, H, shots_schedule, seed_schedule, method=method)

    obj_kwargs = dict(
        n=n,
        edges=edges,
        H=H,
        shots_schedule=shots_schedule,
        seed_schedule=seed_schedule,
        method=method,
        eval_current=eval_current,
    )

    geom_data = {
        "iters": [],
        "Ecut": [],
        "fg": [],
        "shots": [],
        "states": [],
    }

    def geom_callback(k: int, params: np.ndarray, val: float, aux_step: dict):
        p_local = len(params) // 2
        gammas_k = params[:p_local]
        betas_k = params[p_local:]
        psi_k = qaoa_statevector(n, edges, gammas_k, betas_k)
        fg_k, _, _ = geometric_fraction(psi0, psi_k)
        state["fg"] = float(fg_k)

        geom_data["iters"].append(int(k))
        geom_data["Ecut"].append(float(-val))
        geom_data["fg"].append(float(fg_k))
        geom_data["shots"].append(int(aux_step["auxp"]["shots"]))
        geom_data["states"].append(psi_k)

    best_params, info = spsa.minimise_pair(
        objective_pair,
        x0,
        maxiter=int(maxiter),
        callback=geom_callback,
        **obj_kwargs,
    )

    final_sampler = make_aer_sampler(shots=4096, seed=777, method=method)
    final_val, final_aux = evaluate_params(best_params, n, edges, final_sampler, H)

    raw_best = final_aux["best_str"]
    best_str_pretty = display_str(raw_best, n)
    best_cut = bitstring_to_cut_value(raw_best, edges)
    ratio = approx_ratio(best_cut, opt_cut)
    est_Ecut = -float(final_val)

    history_aux = info["history"]["aux"]
    total_shots = compute_total_shots_from_history_aux(history_aux)

    L = D = C_add = C_ratio = None
    if len(geom_data["states"]) >= 2:
        L, D, C_add, C_ratio, _ = circuitousness(geom_data["states"])

    if do_plots:
        iters = np.array(geom_data["iters"])
        Ecut = np.array(geom_data["Ecut"])
        fg = np.array(geom_data["fg"])
        shots = np.array(geom_data["shots"])

        plt.figure()
        plt.plot(iters, Ecut, marker=".")
        plt.axhline(opt_cut, ls="--", lw=1, alpha=0.6, label=f"opt cut = {opt_cut}")
        plt.xlabel("Iteration")
        plt.ylabel("E[cut]")
        plt.title(f"{tag} | n={n} d={d} p={p} seedG={graph_seed} seedS={spsa_seed}")
        plt.legend()
        plt.tight_layout()

        plt.figure()
        ax1 = plt.gca()
        ax1.plot(iters, Ecut, marker=".", label="E[cut]")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("E[cut]")
        ax2 = ax1.twinx()
        ax2.plot(iters, fg, marker="x", label="fg")
        ax2.set_ylabel("fg")
        plt.title(f"{tag} | E[cut] + fg")
        plt.tight_layout()

        plt.figure()
        ax1 = plt.gca()
        ax1.plot(iters, shots, marker=".", label="shots")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("shots")
        ax2 = ax1.twinx()
        ax2.plot(iters, fg, marker="x", label="fg")
        ax2.set_ylabel("fg")
        plt.title(f"{tag} | shots + fg")
        plt.tight_layout()

        plt.show()

    return {
        "tag": tag,
        "n": n,
        "d": d,
        "p": p,
        "graph_seed": int(graph_seed),
        "spsa_seed": int(spsa_seed),
        "opt_cut": int(opt_cut),
        "best_cut": int(best_cut),
        "approx_ratio": float(ratio),
        "est_Ecut": float(est_Ecut),
        "best_str": str(best_str_pretty),
        "total_shots": int(total_shots),
        "L": None if L is None else float(L),
        "D": None if D is None else float(D),
        "C_add": None if C_add is None else float(C_add),
        "C_ratio": None if C_ratio is None else float(C_ratio),
    }


def ablation(
    *,
    trials: int,
    n: int,
    d: int,
    p: int,
    maxiter: int,
    method: str,
    out_csv: str,
):
    rows = []

    for t in range(int(trials)):
        graph_seed = 1000 + t
        spsa_seed = 2000 + t

        fixed_sched = fixed_shots_schedule_factory(1024)
        rows.append(
            run_one(
                n=n, d=d, p=p, maxiter=maxiter,
                graph_seed=graph_seed, spsa_seed=spsa_seed,
                shots_schedule=fixed_sched, seed_schedule=default_seed_schedule,
                method=method, do_plots=False, tag="fixed1024",
            )
        )

        rows.append(
            run_one(
                n=n, d=d, p=p, maxiter=maxiter,
                graph_seed=graph_seed, spsa_seed=spsa_seed,
                shots_schedule=default_shots_schedule, seed_schedule=default_seed_schedule,
                method=method, do_plots=False, tag="default",
            )
        )

        state = {"fg": 0.0}
        geom_sched = make_geometry_aware_shots_schedule(state)
        rows.append(
            run_one(
                n=n, d=d, p=p, maxiter=maxiter,
                graph_seed=graph_seed, spsa_seed=spsa_seed,
                shots_schedule=geom_sched, seed_schedule=default_seed_schedule,
                method=method, do_plots=False, tag="geom-aware",
                fg_state=state,
            )
        )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    def summarize(tag: str):
        sub = [r for r in rows if r["tag"] == tag]
        ar = np.array([r["approx_ratio"] for r in sub], dtype=float)
        ts = np.array([r["total_shots"] for r in sub], dtype=float)
        succ = float(np.mean((ar >= 1.0 - 1e-12).astype(float)))
        return {
            "tag": tag,
            "mean_ratio": float(np.mean(ar)),
            "std_ratio": float(np.std(ar)),
            "success_rate": succ,
            "mean_total_shots": float(np.mean(ts)),
            "std_total_shots": float(np.std(ts)),
        }

    s1 = summarize("fixed1024")
    s2 = summarize("default")
    s3 = summarize("geom-aware")

    print("\nAblation summary")
    for s in [s1, s2, s3]:
        print(
            f"{s['tag']:>10} | mean_ratio={s['mean_ratio']:.4f} ± {s['std_ratio']:.4f} | "
            f"success={s['success_rate']:.2f} | mean_shots={s['mean_total_shots']:.1f} ± {s['std_total_shots']:.1f}"
        )
    print(f"\nWrote CSV: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--method", type=str, default="density_matrix")
    ap.add_argument("--out_csv", type=str, default="ablation_results.csv")
    args = ap.parse_args()

    if args.ablation:
        ablation(
            trials=args.trials,
            n=args.n,
            d=args.d,
            p=args.p,
            maxiter=args.maxiter,
            method=args.method,
            out_csv=args.out_csv,
        )
        return

    graph_seed = 42
    spsa_seed = 0

    print("Single run: default schedule")
    run_one(
        n=args.n, d=args.d, p=args.p, maxiter=args.maxiter,
        graph_seed=graph_seed, spsa_seed=spsa_seed,
        shots_schedule=default_shots_schedule, seed_schedule=default_seed_schedule,
        method=args.method, do_plots=True, tag="default",
    )


if __name__ == "__main__":
    main()
