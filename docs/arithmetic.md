# Fixed-Point Arithmetic & Brainfuck Algorithms

This document details the mathematical theory, quantization scheme, and Brainfuck algorithmic implementations for neural network inference.

---

## 1. Fixed-Point Quantization Theory

In fixed-point arithmetic with scale factor $S \in \mathbb{N}_{>0}$:

$$\text{Fixed}(x) = \lfloor x \cdot S + 0.5 \rfloor \in \mathbb{Z}$$

$$\text{Float}(X) = \frac{X}{S} \in \mathbb{R}$$

### Resolution and Dynamic Range
- For scale $S = 16$: Quantization step is $\Delta = \frac{1}{16} = 0.0625$.
- For scale $S = 64$: Quantization step is $\Delta = \frac{1}{64} = 0.015625$.

---

## 2. Neural Layer Computation in Fixed Point

For a dense layer computing $y = \sigma(x W + b)$:

### Step 1: Integer Dot Product Accumulation
Given inputs $X_i = \text{Fixed}(x_i)$ and weights $W_{ij} = \text{Fixed}(w_{ij})$:

$$\text{Acc}_j = \sum_{i=1}^{d_{\text{in}}} X_i \cdot W_{ij}$$

The resulting integer $\text{Acc}_j$ has fixed-point scale $S^2$.

### Step 2: Integer Rescaling and Bias Addition
To bring the pre-activation back to scale $S$:

$$Z_j = \left\lfloor \frac{\text{Acc}_j}{S} \right\rfloor + B_j$$

where $B_j = \text{Fixed}(b_j) = \lfloor b_j \cdot S + 0.5 \rfloor$.

### Step 3: Activation Evaluation
$$Y_j = \text{Activation}_{\text{fixed}}(Z_j, S)$$

---

## 3. Brainfuck Algorithmic Implementations

### 3.1 Non-Destructive Addition: `dst += src`
```brainfuck
# src[dst+ temp+ src-] temp[src+ temp-]
```

### 3.2 Constant Multiplication: `dst += src * W`
```brainfuck
# src[dst +*W temp+ src-] temp[src+ temp-]
```

### 3.3 Integer Division by Constant $D$: `dst = src // D`
Maintains an internal countdown counter initialized to $D$:
```brainfuck
# Initialize rem = D
# For each decrement of src:
#   rem--
#   if rem == 0:
#     rem = D
#     dst++
```

### 3.4 Dual-Rail Canonical Normalization
To transform $(P, N) \to (\max(0, P - N), \max(0, N - P))$:
```brainfuck
# while N != 0:
#   N--
#   if P != 0:
#     P--
#   else:
#     N++
#     break
```

### 3.5 ReLU Activation
In normalized dual-rail format:
- If $Z \ge 0 \implies (Z_{\text{pos}}, 0)$
- If $Z < 0 \implies (0, Z_{\text{neg}})$

$$\text{ReLU}(Z) = (Z_{\text{pos}}, 0)$$

In Brainfuck, ReLU is a single destructive clear `[-]` applied to the negative cell $Z_{\text{neg}}$.

### 3.6 Hard-Sigmoid Activation
$$\sigma_{\text{hard}}(z) = \text{clamp}\left(\left\lfloor \frac{z}{4} \right\rfloor + \left\lfloor \frac{S}{2} \right\rfloor, 0, S\right)$$

In Brainfuck:
1. Divide $Z_{\text{pos}} // 4$ and $Z_{\text{neg}} // 4$.
2. Add offset $S // 2$ to positive quotient.
3. Normalize dual-rail.
4. Zero the negative rail (clamp lower bound at 0).
5. Clamp upper bound to $S$ using dual-rail subtraction.
