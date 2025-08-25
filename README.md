# QAOA‑Shot‑Frugal

**Symmetry‑aware, shot‑frugal QAOA for Max‑Cut (Qiskit).**

> A compact research‑grade playground to study QAOA on small graphs with:
>
> * **p‑layer circuits** and clean parameterization
> * **SPSA** (2 evaluations/step) and baseline optimizers
> * Hooks for **symmetry‑tying** parameters
> * **Benchmarks** vs. classical exact & heuristic solvers
> * **NISQ realism**: Aer noise models, simple mitigation, optional hardware run
>
> Built to align with theory‑aware interests (e.g., parameter landscapes, symmetry, and rigorous baselines).

---

##  Features

* **Max‑Cut Hamiltonian** builder using `SparsePauliOp`.
* **General (n, p)** QAOA circuit: cost (ZZ) + mixer (X) layers.
* **Shot‑frugal SPSA** optimizer; easy to swap in others (COBYLA, Nelder‑Mead, …).
* **Metrics**: expectation ⟨H⟩, best bitstring, cut value, approximation ratio.
* **Baselines (planned/partial)**: brute force opt, Goemans–Williamson (SDP via CVXPY), simulated annealing.
* **Symmetry tying** (edge‑group hooks) to reduce parameters/improve trainability.
* **Noise**: Aer device noise models; measurement error mitigation.
* **Optional hardware**: run final parameters on an IBM backend.

---

##  Repository structure

```
qaoa-shot-frugal/
├─ qaoa/
│  ├─ maxcut.py          # Hamiltonian + expectation from counts
│  ├─ circuits.py        # p‑layer QAOA circuit (n, edges, gammas, betas)
│  ├─ spsa.py            # SPSA optimizer (2 evals/iter)
│  ├─ metrics.py         # cut value, approximation ratio, brute‑force optimum
│  └─ symmetry.py        # edge grouping hooks (symmetry‑tying)
├─ classical/
│  └─ baselines.py       # (WIP) SA and Goemans–Williamson via CVXPY
├─ experiments/
│  └─ run_maxcut_qaoa.py # main runner: 3‑regular ER demo + plots
├─ theory/
│  └─ K3_p1.md           # (WIP) analytic p=1 landscape on triangle
├─ README.md
├─ requirements.txt
└─ .gitignore
```

---

##  Quick start

```bash
# 1) Create & activate a virtual env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the demo experiment (6‑node 3‑regular graph, p=2, SPSA)
python experiments/run_maxcut_qaoa.py
```

You should see:

* the exact optimum cut (via brute force for small n),
* the best bitstring found by QAOA,
* the approximation ratio, and
* a convergence plot of ⟨H⟩ across SPSA iterations.

---

##  Configuration (common knobs)

Inside `experiments/run_maxcut_qaoa.py`:

* **Graph family**: `nx.random_regular_graph(d=3, n=n)` (try ER, cycles, planar, …)
* **Problem size**: `n = 6` (keep small if you want exact optimum as baseline)
* **Layers**: `p = 1,2,3,…`
* **Initialization**: `x0 = [γ₁…γ_p | β₁…β_p]` (e.g., `0.8/0.4` per layer)
* **Shots**: sampler `run_options={"shots": 2048}`
* **Optimizer**: SPSA hyper‑params `(a, c, A, α, γ)` (sensible defaults included)

---

##  Symmetry‑tying (edge groups)

**Why?** Graph symmetries can induce redundant parameters; tying them improves landscapes & sample efficiency.

**How?** Use `qaoa/symmetry.py` to group edges (e.g., by degree signatures or automorphism orbits) and map a smaller set of γ’s onto edges per layer. A thin wrapper can expand `γ_grouped -> γ_per_edge` before circuit construction. Start with regular graphs (all edges equivalent), then try ER/planar with degree‑based grouping.

---

##  Baselines

* **Brute force optimum**: `qaoa/metrics.py::brute_force_maxcut` (n ≲ 16).
* **Simulated annealing (SA)**: to be added in `classical/baselines.py`.
* **Goemans–Williamson (GW)** via SDP:

  * `pip install cvxpy` (solver like ECOS/OSQP)
  * Implement SDP relaxation + randomized rounding, then compare approximation ratios vs QAOA.

Reporting suggestion:

* Table/plot: approximation ratio vs p (QAOA) + GW + SA on the same instances.

---

##  Noise & mitigation (NISQ realism)

* Use **Aer noise models** from a real backend (fake backends or retrieved props) and rerun the SPSA loop.
* Add **measurement error mitigation** (Qiskit `LocalReadoutMitigator`).
* **Optional hardware**: run the *final* parameterized circuit once on a small IBM backend and compare distributions.

> Tip: For hardware, set your token once via `IBMQRuntimeService(channel="ibm_quantum", token=...)` or the IBM Quantum dashboard.

---

##  Outputs

* **Console**: optimum cut, best bitstring, approximation ratio, best ⟨H⟩.
* **Figures**: convergence of ⟨H⟩; bar/line plots comparing optimizers, depths p, noise vs noiseless.
* **Reproducibility**: seeds are set where relevant; add `--seed` CLI flags if you generalize runners.

---

##  Roadmap

* [ ] Symmetry‑tying wrapper + examples (regular, cycle, ER)
* [ ] Simulated annealing baseline
* [ ] Goemans–Williamson (SDP) baseline (CVXPY)
* [ ] Aer noise models + measurement mitigation demo
* [ ] Single hardware run (final parameters)
* [ ] 6–8 page write‑up (incl. small analytic K₃, p=1 note)

---

##  Background reading (friendly)

* QAOA (orig. idea), SPSA (shot‑frugal stochastic optimization), GW SDP (approximation baseline for Max‑Cut), and recent notes on symmetry in variational circuits/landscapes.

> This repository is intentionally minimal: it’s a didactic scaffold you can extend into a small, professor‑friendly study.

---

##  License

TO be choosed license (e.g., MIT or Apache‑2.0) 

---

##  Acknowledgments

Thanks to open‑source quantum toolkits (Qiskit, NetworkX, CVXPY) and the broader community discussions on symmetry, noise, and variational algorithm trainability.
