#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import argparse
import datetime as dt

import pandas as pd
import matplotlib.pyplot as plt


def valid_date(date: str) -> pd.Timestamp:
    """
    validate date
    """
    try:
        return pd.Timestamp(dt.datetime.strptime(date, "%Y-%m-%d"))
    except ValueError:
        msg = f"Not a valid date: '{date}'."
        raise argparse.ArgumentTypeError(msg)


def argument_parser():
    parser = argparse.ArgumentParser(description='process time series data')
    parser.add_argument(
        'filename', type=argparse.FileType('r'),
        help='name of the file to convert')
    parser.add_argument(
        '--timestamp', '-t', help='colume for time stamp', default='timestamp'
    )
    parser.add_argument(
        '--value', '-v', help='colume for value', default='value'
    )
    parser.add_argument(
        '--plot', '-p', action='store_true'
    )
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), help='output file'
    )
    parser.add_argument(
        '--start', '-sd', type=valid_date, help='start timestamp'
    )
    parser.add_argument(
        '--end', '-ed', type=valid_date, help='end timestamp'
    )
    return parser


def main(argv=None):
    args = argument_parser().parse_args(argv)
    print(args)
    df = pd.read_csv(
        args.filename,
        parse_dates=[args.timestamp]
    )
    df = df.rename(
        columns={
            args.timestamp: 'timestamp',
            args.value: 'value'
        })[['timestamp', 'value']]
    df.sort_values(by=['timestamp'])
    if args.start:
        df = df[df['timestamp'] >= args.start]
    if args.end:
        df = df[df['timestamp'] < args.end]
    if args.output:
        df.to_csv(args.output)
    if args.plot:
        plt.plot(df['timestamp'], df['value'])
        plt.show()


if __name__ == '__main__':
    main()
