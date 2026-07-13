import random
import numpy as np
import torch

def set_seed(seed=0):  # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)