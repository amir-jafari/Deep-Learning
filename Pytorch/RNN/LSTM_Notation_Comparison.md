# LSTM Notation Comparison: Lecture vs. Colah Blog

## Overview

This document provides a detailed comparison between the LSTM notation used in the Deep Learning Lecture and the popular Colah blog post on LSTMs. Understanding these notational differences is crucial for reading different LSTM implementations and research papers.

**Key Difference:** The lecture uses explicit **feedback connections with delays**, while Colah's blog uses a more compact representation without explicit feedback notation.

---

## Understanding the Lecture LSTM Architecture

### Layer Structure (from the diagram on page 28)

The lecture LSTM has **7 layers** total:

**Layer 1 - Input Layer (Candidate Cell State):**
- Location: Top row, leftmost section labeled "Input"
- Inputs: p¹(t) via IW^{1,1}, and a⁷(t-1) via delay D and LW^{1,7}
- Output: a¹(t) = tanh(n¹(t))
- Function: Creates candidate values to add to cell state
- Colah equivalent: C̃_t

**Layer 2 - Input Gate:**
- Location: Bottom row, leftmost gate
- Inputs: p¹(t) via IW^{2,1}, and a⁷(t-1) via delay D and LW^{2,7}
- Output: a²(t) = σ(n²(t))
- Function: Controls what portion of candidate goes into cell
- Colah equivalent: i_t

**Layer 3 - Feedback (Forget) Gate:**
- Location: Bottom row, middle gate
- Inputs: p¹(t) via IW^{3,1}, and a⁷(t-1) via delay D and LW^{3,7}
- Output: a³(t) = σ(n³(t))
- Function: Controls what portion of previous cell state to keep
- Colah equivalent: f_t
- **Important**: b³ typically initialized to ones to keep gate ON initially

**Layer 4 - Output Gate:**
- Location: Bottom row, rightmost gate
- Inputs: p¹(t) via IW^{4,1}, and a⁷(t-1) via delay D and LW^{4,7}
- Output: a⁴(t) = σ(n⁴(t))
- Function: Controls what portion of cell state to output
- Colah equivalent: o_t

**Layer 5 - Constant Error Carousel (CEC):**
- Location: Top row, middle section labeled "Constant Error Carousel"
- Inputs: a¹(t) via LW^{5,1}, a³(t) via LW^{5,2}, a²(t) via LW^{5,3}, a⁵(t-1) via delay D and LW^{5,5}
- Output: a⁵(t) = a³(t) ◦ a⁵(t-1) + a²(t) ◦ a¹(t)
- Function: Maintains cell state with minimal gradient decay
- **Fixed weights**: LW^{5,5} = I (identity for feedback loop)
- Colah equivalent: C_t

**Layer 6 - Output Section:**
- Location: Top row, third section labeled "Output"
- Inputs: a⁵(t) via LW^{6,5}
- Output: a⁶(t) = tanh(a⁵(t))
- Function: Applies tanh to cell state before gating
- Colah equivalent: tanh(C_t)

**Layer 7 - Gating Layer (Final Output):**
- Location: Top row, rightmost section labeled "Gating Layer"
- Inputs: a⁶(t) via LW^{7,6}, a⁴(t) via LW^{7,4}
- Output: a⁷(t) = a⁴(t) ◦ a⁶(t) = a⁴(t) ◦ tanh(a⁵(t))
- Function: Applies output gate to produce final hidden state
- **Fixed weights**: LW^{7,6} and LW^{7,4} are typically identity matrices
- Colah equivalent: h_t

---

## Notation Systems

### Lecture Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| **a^m(t)** | Output of layer m at time t | a¹(t), a²(t), a⁷(t) |
| **n^m(t)** | Net input to layer m at time t | n¹(t), n²(t) |
| **IW^{m,l}** | Input weight from external input to layer m | IW^{1,1}, IW^{2,1} |
| **LW^{m,l}** | Layer weight from layer l to layer m | LW^{1,7}, LW^{2,7} |
| **b^m** | Bias for layer m | b¹, b² |
| **D** | Unit delay operator (z^{-1}) | Explicit in diagrams |
| **◦** | Hadamard product (element-wise) | Used in gating |
| **p¹(t)** | External input at time t | Network input |

### Colah Blog Notation

| Symbol | Meaning | Lecture Equivalent |
|--------|---------|-------------------|
| **x_t** | Input at time t | p¹(t) |
| **h_t** | Hidden state at time t | a⁷(t) |
| **h_{t-1}** | Previous hidden state | a⁷(t-1) via delay D |
| **C_t** | Cell state at time t | a⁵(t) |
| **C_{t-1}** | Previous cell state | a⁵(t-1) via delay D |
| **f_t** | Forget gate | a³(t) |
| **i_t** | Input gate | a²(t) |
| **o_t** | Output gate | a⁴(t) |
| **C̃_t** | Candidate cell state | a¹(t) |
| **σ** | Sigmoid activation | logsig transfer function |
| **tanh** | Hyperbolic tangent | tansig transfer function |
| **⊙** | Element-wise multiplication | ◦ (Hadamard product) |

---

## Component-by-Component Comparison

### 1. Candidate Cell State (Input Layer)

**Lecture Notation:**
```
Layer 1: Input Layer  
n¹(t) = IW^{1,1}p¹(t) + LW^{1,7}a⁷(t-1) + b¹
a¹(t) = tanh(n¹(t))
```
- Receives: External input p¹(t) and delayed output a⁷(t-1) through **D**
- **Feedback**: LW^{1,7} connects from output layer 7 back through delay
- Uses tanh activation

**Colah Notation:**
```
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```

**Conversion:**
```
Colah: C̃_t ←→ Lecture: a¹(t)
Colah: x_t ←→ Lecture: p¹(t)  
Colah: h_{t-1} ←→ Lecture: a⁷(t-1) [obtained through delay D]
Colah: W_C ←→ Lecture: [IW^{1,1} | LW^{1,7}] [concatenated weights]
```

---

### 2. Input Gate

**Lecture Notation:**
```
Layer 2: Input Gate
n²(t) = IW^{2,1}p¹(t) + LW^{2,7}a⁷(t-1) + b²
a²(t) = σ(n²(t))
```
- **Feedback**: LW^{2,7} connects from delayed output a⁷(t-1)

**Colah Notation:**
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
```

**Conversion:**
```
Colah: i_t ←→ Lecture: a²(t)
Colah: W_i ←→ Lecture: [IW^{2,1} | LW^{2,7}]
```

---

### 3. Forget Gate (Feedback Gate)

**Lecture Notation:**
```
Layer 3: Feedback (Forget) Gate
n³(t) = IW^{3,1}p¹(t) + LW^{3,7}a⁷(t-1) + b³
a³(t) = σ(n³(t))
```
- **Feedback**: LW^{3,7} connects from delayed output
- Typical initialization: b³ = [1, 1, ..., 1]ᵀ (gates ON initially)

**Colah Notation:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

**Conversion:**
```
Colah: f_t ←→ Lecture: a³(t)
Colah: W_f ←→ Lecture: [IW^{3,1} | LW^{3,7}]
```

---

### 4. Output Gate

**Lecture Notation:**
```
Layer 4: Output Gate  
n⁴(t) = IW^{4,1}p¹(t) + LW^{4,7}a⁷(t-1) + b⁴
a⁴(t) = σ(n⁴(t))
```
- **Feedback**: LW^{4,7} connects from delayed output

**Colah Notation:**
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
```

**Conversion:**
```
Colah: o_t ←→ Lecture: a⁴(t)
Colah: W_o ←→ Lecture: [IW^{4,1} | LW^{4,7}]
```

---

### 5. Cell State Update (Constant Error Carousel)

**Lecture Notation:**
```
Layer 5: Constant Error Carousel (CEC)
n⁵(t) = LW^{5,1}a¹(t) + LW^{5,2}a³(t) + LW^{5,3}a²(t) + LW^{5,5}a⁵(t-1)
```

**IMPORTANT**: The actual operation shown in the diagram is:
```
a⁵(t) = a³(t) ◦ a⁵(t-1) + a²(t) ◦ a¹(t)
        └─ forget gate ─┘    └─ input gate ─┘
```

Where:
- LW^{5,2} implements the Hadamard product with forget gate a³(t)
- LW^{5,3} implements the Hadamard product with input gate a²(t)
- LW^{5,1} brings in the candidate values a¹(t)
- LW^{5,5} = **I** (identity, for feedback loop through **D**)

**Colah Notation:**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```

**Conversion:**
```
Colah: C_t ←→ Lecture: a⁵(t)
Colah: C_{t-1} ←→ Lecture: a⁵(t-1) [through delay D]
Colah: f_t ⊙ C_{t-1} ←→ Lecture: a³(t) ◦ a⁵(t-1)
Colah: i_t ⊙ C̃_t ←→ Lecture: a²(t) ◦ a¹(t)
```

**Key Point:** 
- Lecture explicitly shows **LW^{5,5} = I** with delay **D** creating feedback
- Colah implicitly passes C_{t-1} as an input (no feedback arrow needed)
- The linear layer weights are used to implement the gating operations

---

### 6. Output Section

**Lecture Notation:**
```
Layer 6: Output Section
n⁶(t) = LW^{6,5}a⁵(t)
a⁶(t) = tanh(a⁵(t))
```

Where:
- LW^{6,5} is typically the identity matrix
- This just applies tanh to the cell state

**Colah Notation:**
```
tanh(C_t)  [intermediate, not assigned to variable]
```

**Conversion:**
```
Colah: tanh(C_t) ←→ Lecture: a⁶(t)
```

---

### 7. Hidden State Output (Gating Layer)

**Lecture Notation:**
```
Layer 7: Gating Layer
n⁷(t) = LW^{7,6}a⁶(t) + LW^{7,4}a⁴(t)
a⁷(t) = a⁴(t) ◦ a⁶(t) = a⁴(t) ◦ tanh(a⁵(t))
```

Where:
- LW^{7,6} and LW^{7,4} are typically identity matrices
- Output gate a⁴(t) modulates the tanh of cell state via Hadamard product

**Colah Notation:**
```
h_t = o_t ⊙ tanh(C_t)
```

**Conversion:**
```
Colah: h_t ←→ Lecture: a⁷(t)
Colah: o_t ←→ Lecture: a⁴(t)
Colah: tanh(C_t) ←→ Lecture: a⁶(t)
```

---

## Complete LSTM Equations Side-by-Side

| Colah Blog | Lecture Notation | Layer |
|------------|------------------|-------|
| `C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)` | `a¹(t) = tanh(IW^{1,1}p¹(t) + LW^{1,7}a⁷(t-1) + b¹)` | Layer 1 |
| `i_t = σ(W_i·[h_{t-1}, x_t] + b_i)` | `a²(t) = σ(IW^{2,1}p¹(t) + LW^{2,7}a⁷(t-1) + b²)` | Layer 2 |
| `f_t = σ(W_f·[h_{t-1}, x_t] + b_f)` | `a³(t) = σ(IW^{3,1}p¹(t) + LW^{3,7}a⁷(t-1) + b³)` | Layer 3 |
| `o_t = σ(W_o·[h_{t-1}, x_t] + b_o)` | `a⁴(t) = σ(IW^{4,1}p¹(t) + LW^{4,7}a⁷(t-1) + b⁴)` | Layer 4 |
| `C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t` | `a⁵(t) = a³(t) ◦ a⁵(t-1) + a²(t) ◦ a¹(t)` | Layer 5 |
| `tanh(C_t)` | `a⁶(t) = tanh(a⁵(t))` | Layer 6 |
| `h_t = o_t ⊙ tanh(C_t)` | `a⁷(t) = a⁴(t) ◦ a⁶(t)` | Layer 7 |

---

## Understanding Feedback in the Lecture Diagrams

### What the Arrows Mean

In the lecture diagrams, **arrows with delay blocks (D)** indicate **feedback connections**:

```
Example from Input Gate (Layer 2):

        ┌──────────────────┐
        │                  │
   a⁷(t)│                  ↓
        └──→ [D] ──→ LW^{2,7} ──→ (+) ──→ Layer 2
                                    ↑
                            IW^{2,1}p¹(t)
```

**Interpretation:** 
- The output a⁷(t) from time t flows into the delay D
- The delay D produces a⁷(t-1) at time t+1
- This delayed output feeds back into Layer 2 via weight LW^{2,7}
- **This is what makes it a recurrent network**

### All Feedback Connections in LSTM

Every gate and the input layer have **two** feedback connections:

1. **From Final Output a⁷(t) through delay:**
   - LW^{1,7}: Feeds back to Input Layer (candidate)
   - LW^{2,7}: Feeds back to Input Gate
   - LW^{3,7}: Feeds back to Feedback (Forget) Gate
   - LW^{4,7}: Feeds back to Output Gate

2. **From CEC a⁵(t) through delay:**
   - LW^{5,5} = I: Feeds back to itself (the carousel!)

**This is the key architectural feature**: The network's own outputs influence its next computations through these delay-based feedback paths.

---

## Visual Representation Differences

### Lecture Diagram Style

```
Key Features:
1. **Delay Blocks (D)**: Explicitly shown where feedback occurs
2. **Layer Numbers**: Each component numbered 1-7
3. **Weight Matrices**: Labeled as IW (input) and LW (layer/feedback)
4. **Temporal Flow**: D blocks show time delays explicitly
5. **Separate Gate Boxes**: Three gates shown separately at bottom

Example of Feedback:
        ┌──────────────────┐
        │                  │
   a⁷(t)│                  ↓
        └──→ [D] ──→ LW ──→ Gate
                             ↑
                           Input
```

**Interpretation:** 
- Arrows going backwards with D blocks = recurrent connections
- These create temporal dependencies
- The network processes sequences step-by-step

### Colah Blog Diagram Style

```
Key Features:
1. **No Explicit Delays**: Previous states (h_{t-1}, C_{t-1}) are inputs
2. **Compact Cell**: All gates shown within one "LSTM cell" box
3. **State Lines**: Horizontal lines show cell state C_t propagation
4. **No Feedback Arrows**: State passing is implicit
5. **Flow Diagram**: Shows information flow through gates

Example:
   [h_{t-1}] ──────┬──────┐
   [C_{t-1}] ──────┤      │
                   │ LSTM │──→ [h_t]
   [x_t]    ───────┤ Cell │──→ [C_t]
                   └──────┘
```

**Interpretation:**
- Previous states are simply inputs to the current cell
- No explicit "feedback loop" shown
- Sequential processing is implicit in the t subscripts

---

## Practical Conversion Guide

### Converting Colah to Lecture Notation

**Step 1: Split concatenated weights**
```python
# Colah style
W_i = weights for input gate (matrix)
[h_{t-1}, x_t] = concatenated input vector

# Lecture style  
# Split W_i into two separate weight matrices:
IW^{2,1} = weights for external input p¹(t)
LW^{2,7} = weights for feedback from a⁷(t-1)
```

**Step 2: Make delays explicit**
```python
# Colah: h_{t-1} is just "previous hidden state"
h_{t-1}

# Lecture: Show it comes through delay
a⁷(t) → [D] → a⁷(t-1) → LW^{2,7} → Layer 2
```

**Step 3: Expand to separate net input and activation**
```python
# Colah compact form
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

# Lecture expanded form
n²(t) = IW^{2,1}p¹(t) + LW^{2,7}a⁷(t-1) + b²
a²(t) = σ(n²(t))
```

**Step 4: Assign layer numbers**
```
Colah doesn't number - uses variable names
Lecture numbers all 7 layers explicitly
```

**Step 5: Map gate outputs correctly**
```python
# Colah → Lecture mapping:
C̃_t → a¹(t)  (Layer 1: candidate)
i_t  → a²(t)  (Layer 2: input gate)
f_t  → a³(t)  (Layer 3: forget gate)
o_t  → a⁴(t)  (Layer 4: output gate)
C_t  → a⁵(t)  (Layer 5: cell state)
     → a⁶(t)  (Layer 6: tanh(cell state))
h_t  → a⁷(t)  (Layer 7: final output)
```

### Converting Lecture to Colah Notation

**Step 1: Combine feedback and input weights**
```python
# Lecture: separate matrices
IW^{2,1} ← from p¹(t)
LW^{2,7} ← from a⁷(t-1)

# Colah: concatenate horizontally
W_i = [IW^{2,1} | LW^{2,7}]
```

**Step 2: Replace delay notation with subscripts**
```python
# Lecture
a⁷(t-1) through delay D

# Colah  
h_{t-1}  # Just subscript, no delay operator shown
```

**Step 3: Use compact gate notation**
```python
# Lecture layers → Colah symbols
a¹(t) → C̃_t  (candidate cell state)
a²(t) → i_t   (input gate)
a³(t) → f_t   (forget gate)
a⁴(t) → o_t   (output gate)
a⁵(t) → C_t   (cell state)
a⁶(t) → [not assigned in Colah, just tanh(C_t)]
a⁷(t) → h_t   (hidden state)
```

**Step 4: Combine net input and activation**
```python
# Lecture: two steps
n²(t) = IW^{2,1}p¹(t) + LW^{2,7}a⁷(t-1) + b²
a²(t) = σ(n²(t))

# Colah: one step
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
```

---

## Practical Example: Forward Pass

Let's trace one time step through both notations.

### Given:
- Input: x_t = p¹(t) = [0.5]
- Previous hidden: h_{t-1} = a⁷(t-1) = [0.3]
- Previous cell: C_{t-1} = a⁵(t-1) = [0.8]
- Assume all weights are 1.0 and biases are 0 for simplicity

### Colah Notation Steps:

```python
# 1. Candidate
C̃_t = tanh(W_C · [0.3, 0.5] + 0)
     = tanh(0.3 + 0.5) = tanh(0.8) = 0.664

# 2. Input gate  
i_t = σ(W_i · [0.3, 0.5] + 0)
    = σ(0.8) = 0.689

# 3. Forget gate
f_t = σ(W_f · [0.3, 0.5] + 0)
    = σ(0.8) = 0.689

# 4. Cell update
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
    = 0.689 × 0.8 + 0.689 × 0.664
    = 0.551 + 0.458 = 1.009

# 5. Output gate
o_t = σ(W_o · [0.3, 0.5] + 0)
    = σ(0.8) = 0.689

# 6. Hidden state
h_t = o_t ⊙ tanh(C_t)
    = 0.689 × tanh(1.009)
    = 0.689 × 0.765 = 0.527
```

### Lecture Notation Steps:

```python
# Layer 1: Input Layer (candidate)
n¹(t) = 1.0 × 0.5 + 1.0 × 0.3 + 0 = 0.8
a¹(t) = tanh(0.8) = 0.664

# Layer 2: Input gate
n²(t) = 1.0 × 0.5 + 1.0 × 0.3 + 0 = 0.8
a²(t) = σ(0.8) = 0.689

# Layer 3: Forget gate
n³(t) = 1.0 × 0.5 + 1.0 × 0.3 + 0 = 0.8
a³(t) = σ(0.8) = 0.689

# Layer 4: Output gate
n⁴(t) = 1.0 × 0.5 + 1.0 × 0.3 + 0 = 0.8
a⁴(t) = σ(0.8) = 0.689

# Layer 5: CEC (cell state)
a⁵(t) = a³(t) ◦ a⁵(t-1) + a²(t) ◦ a¹(t)
      = 0.689 × 0.8 + 0.689 × 0.664
      = 0.551 + 0.458 = 1.009

# Layer 6: Output section
a⁶(t) = tanh(a⁵(t)) = tanh(1.009) = 0.765

# Layer 7: Gating Layer
a⁷(t) = a⁴(t) ◦ a⁶(t)
      = 0.689 × 0.765 = 0.527
```

**Result:** Both notations produce the same output: **h_t = a⁷(t) = 0.527**

---

## Key Conceptual Differences

### 1. Feedback Philosophy

**Lecture Approach:**
- **Explicit recurrent architecture** with delay operators
- Emphasizes that outputs feed back to inputs through time
- Delays (D) are physical architectural components
- Shows how information flows backward in network structure
- Better for understanding backpropagation through time

**Colah Approach:**
- **Sequential state processing**
- Previous states are regular inputs (no special "feedback" status)
- Time dependency shown through subscripts, not architecture
- Emphasizes information flow through gates
- Better for intuitive understanding

### 2. Weight Matrix Representation

**Lecture:**
- **Separate matrices**: IW (input weights) vs LW (layer/recurrent weights)
- Clear architectural distinction
- Explicitly shows which connections are recurrent
- Mirrors neural network layer notation

**Colah:**
- **Combined matrices**: Single W for all inputs
- Concatenates [h_{t-1}, x_t] before multiplication
- More compact, implementation-oriented
- Common in modern frameworks

### 3. Layer Granularity

**Lecture:**
- **Fine-grained** (7 distinct numbered layers)
- Each functional component is separate
- Explicit intermediate values (a⁶(t) = tanh(a⁵(t)))
- Useful for detailed backpropagation analysis

**Colah:**
- **Coarse-grained** ("LSTM cell" concept)
- Gates are operations, not layers
- Some intermediate values not assigned to variables
- Useful for understanding overall function

---

## Summary Comparison Table

| Aspect | Lecture Notation | Colah Notation |
|--------|-----------------|----------------|
| **Feedback** | Explicit with D blocks | Implicit in state passing |
| **Recurrence** | Shown as architectural feedback loops | Shown as sequential states |
| **Weights** | Separate IW and LW | Combined W matrices |
| **Time** | Explicit delays D | Subscript t-1 |
| **Layers** | 7 numbered layers | Gates + cell operations |
| **Candidate** | a¹(t) | C̃_t |
| **Input Gate** | a²(t) | i_t |
| **Forget Gate** | a³(t) | f_t |
| **Output Gate** | a⁴(t) | o_t |
| **Cell State** | a⁵(t) | C_t |
| **tanh(Cell)** | a⁶(t) | tanh(C_t) |
| **Hidden State** | a⁷(t) | h_t |
| **Input** | p¹(t) | x_t |
| **Operations** | Net input n, activation a | Direct computation |
| **Focus** | Architecture & backprop structure | Information flow & gating |

---

## When to Use Each Notation

**Use Lecture Notation When:**
- Implementing LSTM from scratch with explicit layer structure
- Analyzing backpropagation through time in detail
- Understanding recurrent network architecture fundamentals
- Working with systems that separate input and recurrent weights
- Teaching the mechanics of temporal processing

**Use Colah Notation When:**
- Reading modern deep learning papers
- Working with PyTorch, TensorFlow, or other frameworks
- Focusing on the gating mechanism and information flow
- Quick mathematical description
- Explaining LSTM intuition to beginners

---

## Common Misconceptions

### Misconception 1: "Colah doesn't have feedback"
**Reality:** It does! The h_{t-1} and C_{t-1} are the feedback. The notation just doesn't show it as explicitly as the lecture's delay blocks.

### Misconception 2: "The architectures are different"
**Reality:** They're identical. Only the notation differs. Both implement the same computations.

### Misconception 3: "Lecture notation has more layers"
**Reality:** Colah just doesn't assign separate variable names to every intermediate value. Layer 6 (a⁶(t) = tanh(a⁵(t))) exists in both but Colah writes it inline as tanh(C_t).

### Misconception 4: "Different gate orderings"
**Reality:** Gate order doesn't matter since they compute in parallel. The lecture shows all three gates at the bottom, while Colah shows them sequentially in diagrams, but this is just presentation.

---

## Conclusion

Both notations describe the **same LSTM architecture**. The key differences are:

- **Lecture**: Emphasizes recurrent architecture with explicit feedback paths shown via delay blocks
- **Colah**: Emphasizes sequential information processing with implicit state passing

**The Lecture's strength** is showing the recurrent network structure explicitly - you can see how outputs feed back through delays to become inputs. This is invaluable for understanding backpropagation through time.

**Colah's strength** is the intuitive flow diagram showing how information is gated and combined. This helps understand what the LSTM is doing functionally.

Understanding both perspectives will make you proficient at reading LSTM resources across academia and industry!

---

## References

1. Lecture: "Nonlinear Sequence Processing" - Deep Learning Lecture 10
2. Colah Blog: "Understanding LSTM Networks" - https://colah.github.io/posts/2015-08-Understanding-LSTMs/
3. Original LSTM Paper: Hochreiter & Schmidhuber (1997)
4. "Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling" - Sak et al. (2014)

---

**Quick Reference Cheat Sheet:**

```
Colah  →  Lecture      | Colah  →  Lecture
C̃_t   →  a¹(t)        | IW     →  IW (input weights)
i_t    →  a²(t)        | LW     →  LW (recurrent weights)  
f_t    →  a³(t)        | [h,x]  →  separate h and x inputs
o_t    →  a⁴(t)        | t-1    →  delays shown with D blocks
C_t    →  a⁵(t)        | ⊙      →  ◦ (Hadamard product)
tanh(C)→  a⁶(t)        | W      →  [IW | LW] concatenated
h_t    →  a⁷(t)        | cell   →  7 numbered layers
```
