#!/usr/bin/env python

import argparse
import numpy as np


def create_data(n: int, t_max: float, a: float, b: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, t_max, n) + np.random.normal(scale=sigma, size=n)
    t[0] = 0.0
    x = a + b*np.sqrt(t) + np.random.normal(scale=sigma, size=n)
    return t, x


def main():
    arg_parser = argparse.ArgumentParser(description='create noisy data set for a + b*sqrt(t)')
    arg_parser.add_argument('--n', type=int, default=100,
                            help='number of points')
    arg_parser.add_argument('--t_max', type=float, default=30.0,
                            help='maximum time')
    arg_parser.add_argument('--a', type=float, default=0.1,
                            help='additive constant')
    arg_parser.add_argument('--b', type=float, default=0.5,
                            help='multiplication factor')
    arg_parser.add_argument('--sigma', type=float, default=0.0,
                            help='noise level')
    args = arg_parser.parse_args()

    for t, x in zip(*create_data(n=args.n, t_max=args.t_max, a=args.a, b=args.b, sigma=args.sigma)):
        print(f'{t},{x}')

if __name__ == '__main__':
    main()
