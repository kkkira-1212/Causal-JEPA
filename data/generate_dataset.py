# CLI script to pre-generate and save an SCM dataset to disk as a .npz file.

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.cauker_scm import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_series",      type=int,   default=30)
    parser.add_argument("--num_vars",      type=int,   default=5)
    parser.add_argument("--num_timesteps", type=int,   default=1500)
    parser.add_argument("--max_parents",   type=int,   default=2)
    parser.add_argument("--noise_std",     type=float, default=0.1)
    parser.add_argument("--dag_seed",      type=int,   default=42)
    parser.add_argument("--output",        type=str,   default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = (f"data/scm_d{args.num_vars}_n{args.n_series}"
                       f"_t{args.num_timesteps}_dag{args.dag_seed}.npz")

    print(f"D={args.num_vars}  N={args.n_series}  T={args.num_timesteps}  "
          f"max_parents={args.max_parents}  dag_seed={args.dag_seed}  →  {args.output}\n")

    data, true_adj = generate_dataset(
        n_series=args.n_series, num_vars=args.num_vars, num_timesteps=args.num_timesteps,
        max_parents=args.max_parents, noise_std=args.noise_std, dag_seed=args.dag_seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, data=data, true_adj=true_adj,
                        dag_seed=args.dag_seed, n_series=args.n_series)
    print(f"\nSaved to {args.output}  |  shape={data.shape}  |  "
          f"true edges={int(true_adj.sum())}/{args.num_vars*(args.num_vars-1)}")
    print(true_adj)


if __name__ == "__main__":
    main()
