# 🧠 Geometric Median Matching for Robust k-Subset Selection

---

Modern deep learning thrives on massive datasets. But training on all data is expensive — both computationally and financially. What if we could pick just a **small, representative subset** of the data and still train great models?

Welcome to the world of **data pruning**, where the goal is to select a "coreset" — a subset of training examples — that retains the essence of the entire dataset.

The problem? Real-world datasets are often noisy, and most pruning strategies crumble in the presence of corrupted or adversarial data.

In our new work, we propose a robust solution: **Geometric Median (GM) Matching** — a theoretically grounded and practically scalable method that **selects high-quality subsets even when up to 50% of the data is adversarially corrupted**.

---

## 🚧 The Challenge: Robustness vs. Diversity

Most existing pruning methods rely on **importance scores** — think of examples closest to the class centroid or 
hardest to learn. These methods perform well in clean settings.

But under **gross corruption** — mislabeled data, noisy features, or adversarial examples — these strategies fail. 

Why?
Because they typically compute the **empirical mean** to determine centroids. A single outlier can 
**completely distort** this mean — leading to bad selections.

This leads to a trade-off:

- **Robustness**: Retain only the easiest, most prototypical examples → safe but not diverse.
- **Diversity**: Include hard examples → informative but vulnerable to noise.

**How can we break this trade-off?**

---

## 💡 Robust Moment Matching with Geometric Median

### 🎯 Robust Moment Matching

A principled approach to data pruning is **moment matching** — selecting a subset of examples such that the 
**empirical mean of the subset** approximates that of the full dataset. This strategy works well in clean settings.

However, under data corruption (e.g., mislabeled or adversarial examples), the **empirical mean is unreliable** 
— even a single outlier can arbitrarily distort it.

To counter this, we introduce **Robust Moment Matching** — where we match the subset’s mean to a 
**robust estimator** of the dataset’s central tendency, rather than the standard empirical mean.

---

### 🛡️ Geometric Median: A Robust Estimator

The **Geometric Median (GM)** is a classic robust estimator that remains **resilient to up to 50% corrupted data**. 
Unlike the mean (which minimizes squared distances), the GM minimizes the **sum of Euclidean distances** to all points.

#### 📐 Definition
Given a set of points $\{x_1, x_2, \dots, x_n\} \subset \mathbb{R}^d$, the Geometric Median $\mu_{\text{GM}}$ 
is defined as:

$$
\\ \mu_{GM} = \underset{z \in \mathbb{R}^d}{\arg\min} \sum_{i=1}^n || z - x_i ||
$$

This estimator is **translation invariant**, **resistant to outliers**, 
and lies within the **convex hull** of the clean samples.


![Geometric Median vs Mean](gm.png)

### ⚙️ Algorithm: GM Matching (Greedy Subset Selection)

We now describe the **GM Matching** algorithm that selects a subset of \( k \) points whose empirical mean best approximates the Geometric Median of the full dataset in an embedding space.

```python
# GM Matching (Simplified Pseudocode)

Inputs:
- D: dataset of n examples
- ϕ: encoder mapping inputs to embedding space
- γ: fraction for GM estimation
- B: number of batches
- k: size of subset to select

# Step 1: Compute embeddings
Φ = [ϕ(x) for x in D]

# Step 2: Subsample γ-fraction for robust GM estimation
Φ_GM = random_subset(Φ, fraction=γ)

# Step 3: Compute ε-approximate Geometric Median
μ_GM = geometric_median(Φ_GM)

# Step 4: Partition data into B batches
batches = partition(Φ, B)

# Step 5: Greedy selection loop
θ = μ_GM
DS = []

for Φ_b in batches:
    for _ in range(k // B):
        # Select point closest to residual direction
        ω = argmax(dot(θ, ω_i) for ω_i in Φ_b)
        DS.append(ω)
        
        # Update direction vector
        θ = θ + (μ_GM - ω)
        
        # Remove selected point
        Φ_b.remove(ω)

return DS
