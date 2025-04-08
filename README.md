# 🧠 Geometric Median Matching for Robust k-Subset Selection

---

Modern deep learning thrives on massive datasets. But training on all data is expensive — both computationally and financially. What if we could pick just a **small, representative subset** of the data and still train great models?

Welcome to the world of **data pruning**, where the goal is to select a "coreset" — a subset of training examples — that retains the essence of the entire dataset.

The problem? Real-world datasets are often noisy, and most pruning strategies crumble in the presence of corrupted or adversarial data.

In our new work, we propose a robust solution: **Geometric Median (GM) Matching** — a theoretically grounded and practically scalable method that **selects high-quality subsets even when up to 50% of the data is adversarially corrupted**.

---

## 🚧 The Challenge: Robustness vs. Diversity

Most existing pruning methods rely on **importance scores** — think of examples closest to the class centroid or hardest to learn. These methods perform well in clean settings.

But under **gross corruption** — mislabeled data, noisy features, or adversarial examples — these strategies fail. Why?

Because they typically compute the **empirical mean** to determine centroids. A single outlier can **completely distort** this mean — leading to bad selections.

This leads to a trade-off:

- **Robustness**: Retain only the easiest, most prototypical examples → safe but not diverse.
- **Diversity**: Include hard examples → informative but vulnerable to noise.

**How can we break this trade-off?**

---

## 💡 Our Key Insight: Use the Geometric Median

We propose to **replace the empirical mean with a robust alternative — the Geometric Median (GM)**.

- The GM minimizes the **sum of distances** (not squared) to all points.
- It has a **breakdown point of 1/2** — meaning it can tolerate up to 50% adversarial data.
- Unlike the mean, it resists being pulled by outliers.

---

## 🧪 The Method: Geometric Median Matching (GM Matching)

We formulate k-subset selection as a **robust moment matching** problem:

> Find a subset whose **empirical mean** matches the **GM** of the dataset in a meaningful embedding space.

### Step-by-Step

1. **Embed** the data using a pretrained encoder (e.g., CLIP).
2. **Compute** the GM of a subsampled set of embeddings (for scalability).
3. **Iteratively select** points that align with the GM using a herding-style greedy algorithm.

### Algorithm Sketch

```python
# GM Matching (Simplified)
for t in range(k):
    x_t = argmax_x dot(theta_t, embedding(x))
    theta_{t+1} = theta_t + (GM - embedding(x_t))
