#!/usr/bin/env python
"""CLI wrapper: generate an HTML tuning report from a completed analyze_unit_tuning.py output dir."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.visualization import generate_tuning_report


def main():
    parser = argparse.ArgumentParser(
        description='Generate an HTML report from a tuning analysis output directory.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory produced by analyze_unit_tuning.py '
                             '(must contain tuning_maps.npz)')
    parser.add_argument('--title', default=None,
                        help='Optional report title (default: output_dir basename)')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f'Error: {output_dir} is not a directory.', file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(os.path.join(output_dir, 'tuning_maps.npz')):
        print(f'Error: {output_dir}/tuning_maps.npz not found. '
              'Run analyze_unit_tuning.py first.', file=sys.stderr)
        sys.exit(1)

    report_path = generate_tuning_report(output_dir, title=args.title)
    print(f'Report written to {report_path}')


if __name__ == '__main__':
    main()
