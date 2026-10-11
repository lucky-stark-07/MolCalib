"""Fixed configuration (see docs/IMPLEMENTATION_PLAN.md)."""

TASKS = ("caco2_wang", "lipophilicity_astrazeneca", "solubility_aqsoldb")
SEEDS = (1, 2, 3, 4, 5)
ALPHA = 0.1

# Model and training (docs/IMPLEMENTATION_PLAN.md, "Fixed configuration")
HIDDEN = 300
DEPTH = 3
FFN_HIDDEN = 300
FFN_LAYERS = 1
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS = 60
N_MEMBERS = 5
MC_T = 30


def member_seed(split_seed: int, k: int) -> int:
    """Init/shuffle seed of ensemble member k (0-based) for a given split seed."""
    return split_seed * 100 + k


def mcd_seed(split_seed: int) -> int:
    """Seed of the single MC dropout model for a given split seed."""
    return split_seed * 100 + 99

# Statistics (T9): scaffold-cluster bootstrap over test scaffold groups
N_BOOT = 1000
BOOT_SEED = 0
