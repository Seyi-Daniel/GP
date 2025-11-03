#!/usr/bin/env python3
"""
Multi-head Genetic Programming demo with DEAP.

- 12 inputs, 9 outputs (9 heads = 9 trees per individual)
- Population: 10, Generations: 10
- Random fitness assigned each generation to each individual (for E2E sanity)
- Primitive set: +, -, *, protected_div, square (x^2), protected_sqrt, ephemeral constant
- After evolution: compile best individual, run a sample forward pass, print logits + softmax.
"""
import operator
import random
import math
from typing import List

import numpy as np
from deap import base, creator, tools, gp, algorithms

# -----------------------
# Safe / helper primitives
# -----------------------
EPS = 1e-12


def protected_div(a: float, b: float) -> float:
    return a / b if abs(b) > EPS else a


def protected_sqrt(x: float) -> float:
    return math.sqrt(x) if x >= 0.0 else math.sqrt(abs(x))


def square(x: float) -> float:
    return x * x


def rand_const() -> float:
    # Ephemeral constant in [-1, 1]
    return random.uniform(-1.0, 1.0)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)


# -----------------------
# GP configuration
# -----------------------
N_INPUTS = 12
N_HEADS = 9
POP_SIZE = 10
N_GEN = 10
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Primitive set: 12 inputs X0..X11
pset = gp.PrimitiveSet("MAIN", N_INPUTS, prefix="X")
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(square, 1)
pset.addPrimitive(protected_sqrt, 1)
pset.addEphemeralConstant("CONST", rand_const)  # ephemeral constant

# Individuals: list of 9 trees (multi-head)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Tree generators
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
# Build one head (one tree)
toolbox.register("head", gp.PrimitiveTree, toolbox.expr())
# Build individual as 9 independent heads
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.head, n=N_HEADS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compile a single head (tree) to a Python function
toolbox.register("compile", gp.compile, pset=pset)


# -----------------------
# Variation operators for multi-head individuals
# We operate on a randomly chosen head index.
# -----------------------
def cx_one_point_multihead(ind1, ind2):
    i = random.randrange(len(ind1))
    # Apply one-point crossover on the selected head (tree) in each individual
    ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2


def mut_uniform_multihead(individual):
    i = random.randrange(len(individual))
    # Mutate the selected head (tree) via uniform subtree mutation
    individual[i], = gp.mutUniform(individual[i], expr=toolbox.expr_mut, pset=pset)
    return (individual,)


toolbox.register("mate", cx_one_point_multihead)
toolbox.register("mutate", mut_uniform_multihead)
toolbox.register("select", tools.selTournament, tournsize=3)


# -----------------------
# Evaluation: random fitness each generation
# -----------------------
def evaluate(individual) -> tuple:
    # For demonstration only: assign a random fitness in [0,1).
    # Lower is better (FitnessMin).
    return (random.random(),)


# For completeness: show a compiled forward pass using 12 inputs -> 9 outputs
def forward_pass(individual, x: List[float]) -> (np.ndarray, np.ndarray):
    assert len(x) == N_INPUTS
    funcs = [toolbox.compile(expr=head) for head in individual]
    logits = np.array([f(*x) for f in funcs], dtype=float)
    probs = softmax(logits)
    return logits, probs


toolbox.register("evaluate", evaluate)


def main():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print(f"Starting evolution: pop={POP_SIZE}, gens={N_GEN}, heads={N_HEADS}, inputs={N_INPUTS}")
    pop, logbook = algorithms.eaSimple(
        population=pop,
        toolbox=toolbox,
        cxpb=0.5,
        mutpb=0.3,
        ngen=N_GEN,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    # Report best individual
    best = hof[0]
    print("\n=== Best individual (list of 9 heads) ===")
    for i, head in enumerate(best):
        print(f"Head {i}: {head}")

    # Sample inference
    x = np.random.randn(N_INPUTS).tolist()
    logits, probs = forward_pass(best, x)
    print("\n=== Sample forward pass ===")
    print(f"Input x (12): {np.array(x)}")
    print(f"Logits (9): {logits}")
    print(f"Softmax (9): {probs}  (sum={probs.sum():.6f})")

    # Show final stats
    print("\n=== Evolution stats (last gen) ===")
    print(logbook.stream)


if __name__ == "__main__":
    main()
