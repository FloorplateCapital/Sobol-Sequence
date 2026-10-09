import numpy as np
import pandas as pd
from scipy.stats import qmc

DIMENSIONS = 4      # one per source of randomness per trial (+1 if you use the t-copula W draw)
POWER      = 10      # points per seed = 2**POWER  (4 -> 16, 12 -> 4096, 13 -> 8192)
SEEDS      = 3      # independent scrambles (1 is enough to run; more lets you estimate error)
OUTPUT     = "sobol_points V2"

blocks = []
for seed in range(SEEDS):
    sampler = qmc.Sobol(d=DIMENSIONS, scramble=True, seed=seed)  # scrambled = randomised, no (0,0,0) point
    u = sampler.random_base2(POWER)                               # exactly 2**POWER points
    u = np.clip(u, 1e-10, 1 - 1e-10)                              # keep inside (0,1) for NORM.S.INV

    block = pd.DataFrame(u, columns=[f"dim{k+1}" for k in range(DIMENSIONS)])
    block.insert(0, "trial", np.arange(1, 2**POWER + 1))
    block.insert(0, "seed", seed)
    blocks.append(block)

table = pd.concat(blocks, ignore_index=True)
table.to_csv(f"{OUTPUT}.csv", index=False)
table.to_excel(f"{OUTPUT}.xlsx", index=False, sheet_name="Sobol")


