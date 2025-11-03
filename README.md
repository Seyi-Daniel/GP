# Multi-Head Genetic Programming (DEAP) — 12-in, 9-out (demo)

This is a **minimal** end-to-end demo of **multi-head Genetic Programming** built on **DEAP**.  
An **individual** is a **list of 9 GP trees** ("heads"), each producing one output. Together, the 9 heads form a 9-logit vector you could pass through a softmax for multiclass work.

> ⚠️ For this demo the **fitness is random** every generation. That’s on purpose so you can verify the full EA/GP pipeline (init → variation → evaluation → selection → stats) without caring about a dataset.

---

## What’s inside

- **Representation:** `Individual` = Python `list` of 9 `PrimitiveTree`s. Each tree consumes 12 inputs (`X0..X11`) and returns a scalar.
- **Primitives:** `+`, `-`, `*`, **protected** division, **square** (`x^2`), **protected** sqrt, and an **ephemeral constant** in `[-1, 1]`.
- **Evolution:** `eaSimple` with tournament selection, one-point crossover and uniform subtree mutation applied to **one random head** per operation.
- **Run size:** `population=10`, `generations=10` (tweakable in code).
- **Output:** Prints the best individual (all 9 heads), then compiles it and runs a sample forward pass on a random 12-vector, showing logits and softmax probabilities.

---

## Install & run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python gp_multihead.py
```

You should see per-generation stats (min/avg/max fitness), the best individual’s 9 heads, and a sample forward pass like:

```
=== Sample forward pass ===
Input x (12): [ ... 12 floats ... ]
Logits (9): [ ... 9 floats ... ]
Softmax (9): [ ... 9 probs summing to 1.0 ... ]
```

Because fitness is random, don’t expect monotonic improvement—this is a plumbing test.

---

## How it works (short version)

### Primitive set & terminals
We define a DEAP `PrimitiveSet` with 12 arguments and add our math ops. Division and sqrt are protected to avoid NaNs. An ephemeral constant supplies a random scalar terminal per tree.

### Multi-head individual
DEAP individuals can be any Python container; we use a list of 9 trees. Variation operators are thin wrappers that pick a random head and call DEAP’s standard GP operators on that tree (one-point crossover; uniform subtree mutation).

### Random fitness
`evaluate(individual)` returns a fresh `random.random()` each generation for every individual. Since DEAP clears invalidated fitness between gens, this reassigns fitness continually.

### Compile & infer
Each head is compiled with `gp.compile(pset=...)`. Given a 12-vector, we evaluate all 9 heads to produce logits, then apply a numerically stable softmax.

---

## Sample output snippet

```
Starting evolution: pop=10, gens=10, heads=9, inputs=12
gen  nevals  min     avg     max
0    10      0.0324  0.5151  0.9591
1    9       0.0159  0.5078  0.9854
...
=== Best individual (list of 9 heads) ===
Head 0: add(X0, CONST0)
Head 1: mul(X3, X7)
...
=== Sample forward pass ===
Input x (12): [-0.24 ...]
Logits (9): [0.12 ...]
Softmax (9): [0.11 ...]
```

Actual expressions, logits, and probabilities will differ run-to-run because evolution is stochastic and fitness is random, but the structure will look like this.
