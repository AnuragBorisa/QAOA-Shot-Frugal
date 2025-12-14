# 10) How the ZZ phase is built (CNOT–RZ–CNOT) and why we need it

**Goal:** give one phase to **equal** bits (00, 11) and the opposite phase to **different** bits (01, 10). That is a ZZ rotation.

### Why not “just RZ on one qubit”?

An (R_z(\theta)) on a single qubit only looks at **that qubit**. It cannot tell `01` from `10` because both share the same value on that qubit. Max‑Cut needs a phase that depends on **both bits together** (parity: same vs different).

### The trick: encode parity with CNOT, phase with RZ, then undo

**Decomposition (up to global phase):**

* (\mathrm{RZZ}(\theta) = e^{-i\frac{\theta}{2} Z\otimes Z}).
* Implement as: **CNOT**(i→j) → **RZ(θ)** on j → **CNOT**(i→j).

**Intuition:**

1. First CNOT writes the **parity** ((b_i \oplus b_j)) into the target j (equal→0, different→1).
2. (R_z(\theta)) adds a phase based on that parity (0 → (e^{-i\theta/2}), 1 → (e^{+i\theta/2})).
3. Second CNOT restores the bits; the **parity‑dependent phase remains**.

**Phase pattern (ignoring global phase):**

* 00 → (e^{-i\theta/2})
* 01 → (e^{+i\theta/2})
* 10 → (e^{+i\theta/2})
* 11 → (e^{-i\theta/2})

This is exactly (e^{-i\frac{\theta}{2} Z_i Z_j}): equal bits get one phase, different bits the opposite.

---

# 11) Why we use a mixer after the ZZ phase

After the phase separator (U_C(\gamma)=e^{-i\gamma C}), the **magnitudes** of amplitudes haven’t changed—only their **phases** have. If you measured now (from the initial uniform superposition), you’d still see a **uniform distribution**. The **mixer** turns those phase differences into **probability** differences via interference.

**Standard mixer:** (U_B(\beta)=\prod_i e^{-i\beta X_i}). In Qiskit: `rx(2*beta)` on every qubit.

**Role of the mixer:**

* **Moves amplitude** between neighboring bitstrings (flip one bit at a time).
* Makes strings whose phases “align” (from (U_C)) interfere **constructively** (probability up), and misaligned ones interfere **destructively** (probability down).
* Stacking layers ([U_C(\gamma_\ell),U_B(\beta_\ell)]) concentrates probability on high‑cut strings.

**Mini 2‑qubit picture:**

* Start with (|++\rangle) (uniform over 00, 01, 10, 11).
* (U_C) gives phase 1 to 00/11 and phase (e^{-i\gamma}) to 01/10 (the “good” ones).
* (U_B) mixes amplitudes; because 01/10 share a phase, they add up more after mixing → their probabilities increase.

**Without the mixer:** phases alone don’t change measurement probabilities; you wouldn’t bias toward good cuts.

---

# 12) One‑layer template you can reuse

For a graph with edges (E) and parameters ((\gamma,\beta)):

1. `h(q)` on every qubit (q) (make (|+\rangle^{\otimes n})).
2. For each edge ((i,j)): apply **RZZ** on ((i,j)). (Native `rzz(γ)` or CNOT–RZ(γ)–CNOT.)
3. For each qubit (q): apply `rx(2*beta, q)`.
4. Measure in Z. Estimate (\langle C \rangle = \sum_{(i,j)\in E} (1-\langle Z_iZ_j\rangle)/2). Optimize (maximize (\langle C\rangle) or minimize (-\langle C\rangle)).

> **Sign note:** toolkits differ by a sign/global phase in `RZZ` definitions. If results look “mirrored,” flip (\gamma\to-\gamma); physics of Max‑Cut is unchanged.
