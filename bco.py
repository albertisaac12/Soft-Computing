import random

# ---------------------------------------------
# Function to minimize (from PDF)
# f(x) = x^2 + 4
# ---------------------------------------------
def func(x):
    return x*x + 4

# ---------------------------------------------
# Fitness function for minimization (PDF formula)
# fitness = 1 / (1 + f(x))
# ---------------------------------------------
def fitness(x):
    return 1 / (1 + func(x))


# ---------------------------------------------
# Generate a neighbor using BCO formula:
# v = x_i + φ (x_i - x_k)
# φ ~ U[-1, 1]
# ---------------------------------------------
def generate_neighbor(x_i, x_k):
    phi = random.uniform(-1, 1)
    return x_i + phi * (x_i - x_k)


# ---------------------------------------------
# BCO PARAMETERS (MATCH PDF)
# ---------------------------------------------
limit = 2              # abandonment limit
max_cycles = 5         # PDF uses 5 cycles
search_min = -5
search_max = 5

# Initial food sources (PDF example)
sources = [4, -2, 1]   # S1, S2, S3
trials = [0, 0, 0]     # no fails yet

print("\nINITIAL SOURCES:", sources)
print("-----------------------------\n")

# ---------------------------------------------
# BCO MAIN LOOP
# ---------------------------------------------
for cycle in range(1, max_cycles + 1):
    print(f"\n===== CYCLE {cycle} =====")

    # --- 1. Employed Bee Phase ---
    for i in range(len(sources)):
        x_i = sources[i]

        # choose random neighbor (k != i)
        k = random.choice([j for j in range(len(sources)) if j != i])
        x_k = sources[k]

        # generate neighbor
        v = generate_neighbor(x_i, x_k)

        # clip to search range
        v = max(search_min, min(search_max, v))

        # evaluate
        if fitness(v) > fitness(x_i):
            print(f"Source S{i+1} improved: {x_i} → {v}")
            sources[i] = v
            trials[i] = 0
        else:
            print(f"Source S{i+1} failed to improve (trial+1): v={v}")
            trials[i] += 1

    # --- 2. Onlooker Bee Phase ---
    # Probability selection (proportional to fitness)
    fits = [fitness(x) for x in sources]
    total_fit = sum(fits)
    probs = [f / total_fit for f in fits]

    print("\nOnlooker probabilities:", probs)

    # Each onlooker chooses a source
    for _ in range(len(sources)):  # one onlooker per source (simple choice)
        # roulette wheel selection
        r = random.random()
        cumulative = 0
        chosen = 0
        for idx, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen = idx
                break

        i = chosen
        x_i = sources[i]

        # choose random neighbor x_k (k != i)
        k = random.choice([j for j in range(len(sources)) if j != i])
        x_k = sources[k]

        # generate neighbor
        v = generate_neighbor(x_i, x_k)
        v = max(search_min, min(search_max, v))

        # evaluate
        if fitness(v) > fitness(x_i):
            print(f"Onlooker improved S{i+1}: {x_i} → {v}")
            sources[i] = v
            trials[i] = 0
        else:
            trials[i] += 1

    # --- 3. Scout Bee Phase (abandonment) ---
    for i in range(len(sources)):
        if trials[i] >= limit:
            new_source = random.uniform(search_min, search_max)
            print(f"SOURCE S{i+1} ABANDONED → New scout value = {new_source}")
            sources[i] = new_source
            trials[i] = 0

    # Print status this cycle
    print("\nSources after cycle", cycle, ":", sources)
    print("Trials:", trials)
    print("Best:", min(sources, key=lambda x: func(x)), "with f(x) =", func(min(sources, key=lambda x: func(x))))

# --- Final Output ---
best = min(sources, key=lambda x: func(x))
print("\n====================================")
print("FINAL BEST SOLUTION:", best)
print("FINAL BEST VALUE f(x):", func(best))
print("====================================\n")
