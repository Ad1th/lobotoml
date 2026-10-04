# Neurofuck Memory Model & Tape Architecture

This document formalizes the memory tape layout, register allocation strategy, and lifetime management used by the Neurofuck compiler to execute neural network inference on a 1D linear Brainfuck tape.

---

## 1. Abstract Execution Environment

Brainfuck executes on an unbounded or bounded 1D array of integer memory cells:

$$\text{Tape} = [c_0, c_1, c_2, \dots, c_N], \quad c_i \in \mathbb{Z}_{\ge 0}$$

with a single movable data pointer $\text{ptr} \in \mathbb{N}$ initialized to $\text{ptr} = 0$.

### Non-Negative Invariant

Because standard Brainfuck interpreters define cell values as unsigned integers ($c_i \ge 0$) and standard decrement loops `[-]` rely on zero-termination, **all physical tape cells in Neurofuck are guaranteed to be non-negative integers ($\ge 0$) at all times**.

Signed integer values ($x \in \mathbb{Z}$) are represented using **Dual-Rail Register Pairs**.

---

## 2. Dual-Rail Representation

Every logical variable $X$ representing a signed tensor element or pre-activation is mapped to two adjacent or tracked physical memory cells:

$$X = (X_{\text{pos}}, X_{\text{neg}}), \quad X_{\text{pos}} \ge 0, \; X_{\text{neg}} \ge 0$$

The mathematical value of $X$ is defined as:

$$\text{val}(X) = X_{\text{pos}} - X_{\text{neg}}$$

### Canonical Form (Normalized Dual-Rail)

A dual-rail register is in **canonical form** when at most one of the two cells is non-zero:

$$\min(X_{\text{pos}}, X_{\text{neg}}) = 0$$

- If $\text{val}(X) > 0 \implies X_{\text{pos}} = \text{val}(X), \; X_{\text{neg}} = 0$
- If $\text{val}(X) < 0 \implies X_{\text{pos}} = 0, \; X_{\text{neg}} = -\text{val}(X)$
- If $\text{val}(X) = 0 \implies X_{\text{pos}} = 0, \; X_{\text{neg}} = 0$

---

## 3. Tape Memory Layout

The tape is partitioned into distinct functional regions:

```
┌──────────────┬──────────────────┬────────────────────────┬──────────────────────┐
│  Cell 0..7   │    Cell 8..11    │      Cell 12..19       │      Cell 20..N      │
├──────────────┼──────────────────┼────────────────────────┼──────────────────────┤
│ Global       │ Model Input      │ Layer 0 Neurons        │ Layer 1 Neurons &    │
│ Scratchpad   │ Dual-Rail Regs   │ Accumulators & Outputs │ Network Output Regs  │
│ (s0 .. s7)   │ (in_0, in_1)     │ (l0_n0 .. l0_n3)       │ (l1_n0 .. out)       │
└──────────────┴──────────────────┴────────────────────────┴──────────────────────┘
```

### 3.1 Global Scratchpad (`s0` .. `s7`)
- Cells $0$ to $7$ are reserved as a fixed scratch frame for fundamental arithmetic macros (division remainders, branch flags, inversion tokens, temporary copies).
- Any routine that utilizes scratch cells cleans them to $0$ upon completion.

### 3.2 Input Registers
- Cells allocated for each network input dimension.
- Initialized either at startup via constant embedded Brainfuck increments (`+`) or interactively at runtime via `,` instructions.

### 3.3 Layer Neuron Workspace
- For each layer $l$ and neuron $j$:
  - `acc_pos`, `acc_neg`: High-capacity accumulator cells (scale $S^2$).
  - `q_pos`, `q_neg`: Rescaled pre-activation quotient cells (scale $S$).
  - `out_pos`, `out_neg`: Final activated output cells (scale $S$).

---

## 4. Register Allocation & Lifetime Analysis

The Neurofuck compiler uses a deterministic linear-scan tape allocator:

1. **Sequential Allocation**: Variables are assigned monotonic cell indices.
2. **Intermediate Cleanup**: Temporary accumulators and quotient registers are freed immediately after their values are consumed by the subsequent activation phase.
3. **Pointer Tracking**: The compiler tracks the absolute data pointer position $\text{ptr}_{\text{current}}$ at compile-time and emits optimal relative navigation strings (`>` or `<`).
