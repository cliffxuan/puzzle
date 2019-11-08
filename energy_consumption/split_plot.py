#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

PWD = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PWD, 'data')


def main():
    filename = 'EKPC_hourly'
    df = pd.read_csv(
        os.path.join(DATA_DIR, f'{filename}.csv'),
        parse_dates=['Datetime']
    )
    df = df.sort_values(by=['Datetime'])
    count = len(df)
    training_size = int(count * 0.9)
    training = df.iloc[:training_size]
    training.to_csv(f'{filename}_traning.csv')
    test = df.iloc[training_size:]
    test.to_csv(f'{filename}_test.csv')
    plt.plot(training['Datetime'], training['EKPC_MW'])
    plt.plot(test['Datetime'], test['EKPC_MW'])
    plt.show()


if __name__ == '__main__':
    main()
