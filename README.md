# Practical Lightweight Data Prefetchers for CPU Architectures

## 1. Problem Statement

Modern CPUs use *data prefetchers* to guess what data will be needed next and load it early.
This helps reduce memory delay (which is a big bottleneck).

But current approaches fall into two extremes:

* **Simple prefetchers (rule-based)**
  * Work well only when memory access patterns are clean and repetitive
  * Fail when patterns are irregular (common in real-world + ML workloads)

* **ML / Neural prefetchers**
  * Can predict complex patterns
  * But:
    * Require large training datasets and pre-training
    * Have many parameters (heavy memory + heavy computation (heavy multiplications and activation functions))
    * Are mostly fixed after training (not adaptive at runtime)
    * Almost impractical to implement alongside CPU due to size and computation constraints

So there is a gap:

We need something that is **adaptive like ML**, but **lightweight like simple prefetchers**, and **practical to run on real CPUs**.

Though there is some work happening in this direction like Pathfinder which uses SNN. The goal is to find new methods that bridge this gap.

---

## 2. Project Goal

Design and evaluate **data prefetching algorithms that are**:
* **Lightweight** (low computation + low storage)
* **Adaptive** (learns from current program behavior, not heavily pre-trained)
* **Practical for CPU implementation** (low latency, hardware-friendly)

Use **LLMs as an offline analysis tool** to study memory access patterns and *help design smarter heuristics*.
