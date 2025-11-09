# Shot‑Frugal QAOA — SPSA + Pairing + Adaptive Shots

*A simple, story‑style notebook of what we built and why. Future‑me friendly.*

---

## 0) The goal in one line

We want **good QAOA parameters** (γ’s, β’s) for Max‑Cut **with the fewest circuit evaluations/shots**. We do that with **SPSA** (only 2 evaluations per step) plus **shot‑frugal tricks** (pairing + adaptive shots).

---

## 1) Max‑Cut energy — quick refresher

Graph with edges **E**, qubits = nodes. The cost Hamiltonian:

[ H = \tfrac{|E|}{2} - \tfrac{1}{2}\sum_{(i,j)\in E} Z_i Z_j. ]

Two equivalent ways to get the **mean energy** ⟨H⟩ from samples:

* **ZZ view:** compute each (\langle Z_i Z_j\rangle), then
  [\langle H\rangle = \tfrac{|E|}{2} - \tfrac{1}{2}\sum_{(i,j)} \langle Z_i Z_j\rangle.]
* **Bitstring view:** for each measured bitstring *x*, compute **H(x)** and average with its probability: (\langle H\rangle=\sum_x p(x)H(x)).
  *(We use this view to compute variance/SE easily.)*

---

## 2) Why not finite differences?

If we have **D parameters** (D = 2p for QAOA), coordinate finite differences needs **2D evaluations per step**. Example D=10 ⇒ 20 evaluations *per step* → expensive.

---

## 3) SPSA in one page

**Idea:** instead of wiggling parameters one‑by‑one, **wiggle all at once** in a random ±1 pattern.

At step *k*:

1. Draw Δ ∈ {−1,+1}^D (Rademacher each entry).
2. Pick a tiny perturbation size **c_k**.
3. Build two parameter sets:
   (\theta^{+}=\theta + c_k,\Delta),  (\theta^{-}=\theta - c_k,\Delta).
4. Evaluate the objective **f** (mean energy) at both.
5. Gradient estimate (for each i):
   [ \hat g_i = \frac{f(\theta^{+}) - f(\theta^{-})}{2,c_k,\Delta_i}. ]
6. Update with a step size **a_k**:
   (\theta \leftarrow \theta - a_k,\hat g).

**Why SPSA is cheap:** regardless of D, it’s only **2 evaluations per step**.

Schedules we used:

* (a_k = a_0/(k+1+A)^{\alpha}),  (c_k = c_0/(k+1)^{\gamma}).
  (Classic choices: (\alpha≈0.602), (\gamma≈0.101).)

---

## 4) Shot‑frugal trick #1 — *pairing* (make the difference cleaner)

Each evaluation f(θ) is computed from **shots** (repeated measurements) and is therefore **noisy**. SPSA needs the **difference** (f(\theta^{+}) - f(\theta^{-})). If the two sides share the **same randomness**, much of the noise cancels in the subtraction.

### How we force “shared randomness”

* **Simulator (Qiskit Aer):** run ([\text{qc}*+,\text{qc}*-])) **in one batch** with the **same `seed_simulator`** and the **same number of shots**.
  → Their sampling noise becomes **positively correlated**, so the difference is less jittery.
* **Real hardware:** submit θ⁺ and θ⁻ in a **single job** or **back‑to‑back** with the same transpilation/layout/readout‑mitigation to share drift and conditions.

**Bottom line:** cleaner difference ⇒ you can use **fewer shots** per SPSA step for the same gradient quality.

---

## 5) Shot‑frugal trick #2 — *adaptive shots* (don’t overspend early)

Early on we only need a **rough** downhill direction; near the minimum we need **precision**. So we:

* Start with **few shots** (e.g., 256–512 per eval).
* **Increase** shots later (e.g., toward 1024–2048) only when needed.

Two controllers we used:

1. **Iteration‑based:** `shots_k = min(cap, base * (1+k)^growth)`
   (e.g., base=256, growth≈0.35–0.5, cap=2048.)
2. **Noise‑aware (SE‑target):** estimate the **standard error** of the **difference** after each paired step and set
   [ S_{k+1} = \mathrm{clip}\big(S_k, (\text{SE}*{\text{diff}}/\text{SE}*{\text{target}})^2\big). ]
   Rule of thumb: if the difference is **2× too noisy**, multiply shots by **4×** next time.

**Mini‑glossary:**

* **Variance** = average squared spread around the mean.
* **SD** (standard deviation) = √variance (spread of individual samples).
* **SE** (standard error) ≈ SD/√shots (wiggle of the **average**; shrinks with more shots).

We compute variance/SE straight from bitstring counts by first computing H(x) for each bitstring.

---

## 6) The recipe we coded (SIM‑friendly)

**Per SPSA iteration k:**

1. Draw Δ, set (\theta^{±}=\theta \pm c_k\Delta).
2. **Pairing:** build circuits for θ⁺ and θ⁻; run **one** `sampler.run([qc_plus, qc_minus])` with

   * `seed_simulator = base_seed + k`,
   * `shots = shots_schedule(k)`.
3. From both results, compute **f⁺, f⁻** (mean energies).
   Also compute **variance** and **SE** from counts if using the noise‑aware controller.
4. Gradient via (\hat g) (formula above); update θ with **a_k**.
5. (Optional) Update **shots** for next iteration using SE‑target.
6. Log: value, shots used, best bitstring, and cumulative shots.

At the **end**, do one **high‑shot** evaluation (e.g., 4096–8192) for a clean final ⟨H⟩ and a reliable best bitstring.

---

## 7) Tiny code snippets (shape, not full repo)

**SPSA core:**

```python
Delta = rng.choice([-1.0, 1.0], size=D)
ck = c0 / (k+1)**gamma
fp, auxp = obj(theta + ck*Delta)
fm, auxm = obj(theta - ck*Delta)
ghat = (fp - fm) / (2*ck) * (1.0/Delta)
ak = a0 / (k+1+A)**alpha
theta = theta - ak*ghat
```

**Paired objective (Aer):**

```python
shots = shots_schedule(k)
seed  = seed_schedule(k)
sampler = AerSampler(
    backend_options={"method": "density_matrix", "seed_simulator": seed},
    run_options={"shots": shots},
)
res = sampler.run([qc_plus, qc_minus]).result()
counts_p, counts_m = res.quasi_dists[0], res.quasi_dists[1]
fp = expectation_from_counts(counts_p, H)
fm = expectation_from_counts(counts_m, H)
```

**Energy stats from counts (for SE):**

```python
def energy_stats_from_counts(counts, edges):
    # map bits → z∈{+1,−1}, compute H(x) for each bitstring
    Es, ps = [], []
    for bitstr, p in counts.items():
        z = [1 if b=='0' else -1 for b in reversed(bitstr)]  # q0 is rightmost
        zz = sum(z[i]*z[j] for (i,j) in edges)
        Hx = 0.5*len(edges) - 0.5*zz
        Es.append(Hx); ps.append(p)
    Es = np.array(Es, float); ps = np.array(ps, float)
    mu = float((ps*Es).sum())
    var = float((ps*(Es**2)).sum()) - mu*mu
    return mu, max(var, 0.0)
```

**SE‑target shots update:**

```python
SEp = (var_p / shots)**0.5
SEm = (var_m / shots)**0.5
rho = 0.5  # pairing correlation guess
SEdiff = max((SEp**2 + SEm**2 - 2*rho*SEp*SEm)**0.5, 1e-12)
shots_next = clip(shots * (SEdiff/SE_target)**2, min_shots, max_shots)
```

---

## 8) What numbers we observed (rule‑of‑thumb)

* Without pairing, hitting a target noise on (f^+ - f^-) might need ~**1536 shots** per eval.
* With pairing (same batch + seed), similar quality at ~**256–512 shots** per eval.
* With adaptive shots, total budget across 150 steps can drop from ~**6.1e5** shots (finite diff D=10) or **6.1e5 → 1.5e5** (SPSA fixed 2048 → SPSA+pairing avg ~500).

*(Exact numbers depend on n, p, graph, backend/noise.)*

---

## 9) Sanity checks & pitfalls

* **q0 bit order:** rightmost bit is qubit 0 when mapping to Z eigenvalues.
* **Same seed & same batch** for θ⁺/θ⁻ on simulator; on hardware, **same job or back‑to‑back**.
* Start with modest **a0, c0**; if updates are jumpy, lower a0 or raise shots slightly.
* Always do a **final high‑shot** evaluation for reporting.
* Log **shots per iteration** and **total shots**; these plots tell the story.

---

## 10) One‑minute executive summary

* **SPSA** cuts evaluations/step from **2D → 2** by wiggling all params together.
* **Pairing** (same randomness) makes the SPSA **difference** less noisy ⇒ fewer shots.
* **Adaptive shots** avoids wasting shots early; add them only when the gradient needs it.
* Together: **shot‑frugal QAOA** that converges with a much smaller sampling budget — and a clean, reproducible code path to show it.

---

*End of notes.*
