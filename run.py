import argparse
import os
import random
import numpy as np
import torch
from exp.exp_AttentionLSTM import Exp_AttentionLSTM


# parser = argparse.ArgumentParser(description="AttentionLSTM")
# parser.add_argument("--is_trianing", type=bool, default=True, help="Training mode")
# parser.add_argument("--loro_num", type=int, default=3, help="Number of LORO iterations")

# args = parser.parse_args()
# fix_seed = args.random_seed
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)

# Exp = Exp_AttentionLSTM
# if args.is_trianing:
#     for loro_itr in range(args.loro_num):
#         exp = Exp(args)


def main():
    # Set random seed for reproducibility
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Initialize the experiment
    exp = Exp_AttentionLSTM(model_name="AttentionLSTM")

    for loro_itr in range(3):  # Assuming 3 iterations for LORO
        print(f"Running LORO iteration {loro_itr + 1}")
        # Run the experiment
        exp.train(loro_itr)


if __name__ == "__main__":
    main()
