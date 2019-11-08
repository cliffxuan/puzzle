import argparse

from scipy import signal
import pandas as pd
import numpy as np


def argument_parser():
    parser = argparse.ArgumentParser(description='resample time series data')
    parser.add_argument(
        'filename', type=argparse.FileType('r'),
        help='name of the file to convert')
    parser.add_argument('rate', type=int, help='sampling rate in second')
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), help='output file'
    )
    parser.add_argument(
        '--format', '-f', help='timestamp format'
    )
    return parser


def main(argv=None):
    args = argument_parser().parse_args(argv)
    df = pd.read_csv(
        args.filename,
        parse_dates=['timestamp']
    )
    interval = (df.loc[1].timestamp - df.loc[0].timestamp).seconds
    new_size = len(df) * interval // args.rate
    new_timestamp = pd.date_range(
        df.loc[0].timestamp,
        periods=new_size,
        freq=f'{args.rate}S'
    )
    if args.format:
        new_timestamp = new_timestamp.map(lambda x: x.strftime(args.format))
    new_value = signal.resample(df['value'], new_size)
    new_df = pd.DataFrame({'timestamp': new_timestamp, 'value': new_value})
    if args.output:
        new_df.to_csv(args.output)


if __name__ == '__main__':
    main()
