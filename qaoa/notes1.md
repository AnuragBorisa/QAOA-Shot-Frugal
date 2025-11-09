# Max‑Cut & Expectations — Simple Notes

These are memory‑friendly notes to revisit months later. Plain language, minimal symbols, with tiny worked examples.

---

## 0) The story in one line

We split graph nodes into **two groups** (0 or 1). An **edge** scores **1** when its endpoints land on **opposite sides**, else **0**. We build a quantum circuit that produces bitstrings (assignments). Because measurements are random, we optimize the **expected** total score.

---

## 1) Classical Max‑Cut for one edge

Nodes: 0 and 1. Bits (x_0,x_1\in{0,1}) say which side (0=A, 1=B).

**Edge score (classical):**

* 1 if bits differ, 0 if same.
* Three equivalent formulas:

  * XOR: (x_0 \oplus x_1)
  * Squared difference: ((x_0 - x_1)^2)
  * Algebraic: (x_0 + x_1 - 2x_0x_1)

For many edges, just **sum over edges**.

---

## 2) From classical to Pauli‑Z (why Z?)

Quantum computers measure in the **computational (Z) basis** by default:

* (|0\rangle) is Z‑eigenvalue **+1**
* (|1\rangle) is Z‑eigenvalue **−1**

Switch to **spin variables** (s_i\in{+1,-1}) via (x_i=(1-s_i)/2). Then
[x_i \oplus x_j = \frac{1 - s_i s_j}{2}.]
On qubits, identify (s_i \leftrightarrow Z_i). So the **edge operator** is
[\boxed{;\text{edgeScore}_{(i,j)} = \frac{1 - Z_i Z_j}{2};}]
Sum this over all edges to get the **cost Hamiltonian** (C).

* For a **line of 3 nodes** (0–1–2):
  [C = \tfrac{1 - Z_0Z_1}{2} + \tfrac{1 - Z_1Z_2}{2}.]
* For a **triangle** (0–1, 1–2, 0–2):
  [C = \tfrac{1 - Z_0Z_1}{2} + \tfrac{1 - Z_1Z_2}{2} + \tfrac{1 - Z_0Z_2}{2}.]

**Operator meaning:** On a basis bitstring, each (Z_iZ_j) evaluates to **+1** if bits are equal (00 or 11) and **−1** if different (01 or 10). So ((1−Z_iZ_j)/2) is exactly 0/1 for “not cut / cut”.

---

## 3) Why we use **expectations**

A single measurement returns one random bitstring. We need a **stable objective** for optimization (VQE/QAOA/SPSA). That is the **expected value** of the cost operator:
[\boxed{;\langle C \rangle = \sum_{(i,j)\in E} \frac{1 - \langle Z_i Z_j \rangle}{2};}]
This equals the **average number of cut edges** you’d see over many shots.

Reasons:

* Quantum outcomes are random; expectations are smooth/deterministic for fixed parameters.
* Variational principle (for energies) and linearity (for sums of Pauli terms) fit expectations.

---

## 4) Expectation from eigenvalues & probabilities

Think “weighted average of eigenvalues.”

**Single qubit:** Z has eigenvalues +1 (for 0), −1 (for 1). If (p_0,p_1) are probabilities,
[\boxed{;\langle Z \rangle = (+1)p_0 + (-1)p_1 = p_0 - p_1;}]

**Two qubits:** (Z_iZ_j) eigenvalues on 00,01,10,11 are +1,−1,−1,+1. Let (p_{00},p_{01},p_{10},p_{11}) be probabilities for those two positions (marginalizing other qubits if any):
[\boxed{;\langle Z_i Z_j \rangle = (p_{00}+p_{11}) - (p_{01}+p_{10});}]
Then the **edge expectation** is
[\boxed{;\mathbb E[\text{edge }(i,j)] = \frac{1 - \langle Z_iZ_j \rangle}{2} = p_{01}+p_{10}; }]
(“probability the two bits differ”).

---

## 5) Concrete 2‑node refresher (1 edge)

* Bitstrings: 00, 01, 10, 11.
* **Per shot edge score:** 1 for 01/10, 0 for 00/11.
* **Expected edge score:** (p_{01}+p_{10} = (1-\langle Z_0Z_1\rangle)/2).
* **Max‑cut value** here is **1** (best bitstrings: 01 or 10).

---

## 6) Concrete 3‑node refresher

You have **3 qubits → 8 bitstrings**. The **total cut** is the sum of contributions from each edge.

### Line 0–1–2

[\langle \text{Cut} \rangle = \frac{1-\langle Z_0Z_1\rangle}{2} + \frac{1-\langle Z_1Z_2\rangle}{2}.]

* Max possible = **2** (e.g., bitstrings 010 or 101 cut both edges).

### Triangle (complete graph)

[\langle \text{Cut} \rangle = \frac{1-\langle Z_0Z_1\rangle}{2} + \frac{1-\langle Z_1Z_2\rangle}{2} + \frac{1-\langle Z_0Z_2\rangle}{2}.]

* Max possible = **2** (you can’t cut all three edges at once).

**How to compute from counts:**

1. Gather shot counts into probabilities (p_b) over bitstrings.
2. For each pair ((i,j)) form the pairwise marginals (p^{(i,j)}*{00},p^{(i,j)}*{01},p^{(i,j)}*{10},p^{(i,j)}*{11}) by summing over the other qubits.
3. Plug into (\langle Z_iZ_j\rangle=(p_{00}+p_{11})-(p_{01}+p_{10})).
4. Sum edge expectations ((1-\langle Z_iZ_j\rangle)/2) to get (\langle C\rangle).

---

## 7) What does optimization try to do?

* **Goal (Max‑Cut):** increase (\langle C\rangle) (or equivalently minimize (-\langle C\rangle)).
* Intuition: push probability mass onto **good** bitstrings (those that cut many edges). For the 3‑node line, push towards 010 and 101.
* After training, **sample** the circuit and report the **best bitstring** observed (the concrete cut).

---

## 8) Tiny SPSA note (why we take differences)

For gradient estimates we evaluate the cost at (\theta\pm c\Delta) and use the **difference** to estimate a slope (central difference). Using the **same seed** for the two evaluations pairs shot noise so their **difference** is less noisy → fewer shots needed.

---

## 9) Quick checklists

* One qubit per **node**.
* Per edge (i,j) use ((1 - Z_iZ_j)/2).
* Expectation over shots = average cut value.
* Maximize (\langle C\rangle) (or minimize (-\langle C\rangle)).
* Final answer: the high‑scoring sampled bitstring(s).

---

### Micro‑examples to memorize

* 2 nodes: best = **01** or **10**, value = **1**.
* 3‑node line: best = **010** or **101**, value = **2**.
* 3‑node triangle: any bitstring with exactly **one bit different** from the other two has value **2**.

That’s the whole pipeline, end‑to‑end, in plain language.
