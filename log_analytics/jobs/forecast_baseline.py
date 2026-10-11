"""Holt-Winters vs a seasonal-naive baseline on daily request counts.
(Run on the driver with pandas/statsmodels; the series is only ~60 points.)

Usage: python jobs/forecast_baseline.py --reports output/reports
"""
import argparse
import glob

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def mape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.mean(np.abs((y - f) / y)) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="output/reports")
    ap.add_argument("--season", type=int, default=7)
    ap.add_argument("--holdout", type=int, default=7)
    a = ap.parse_args()
    f = glob.glob(f"{a.reports}/requests_per_day/*.csv")[0]
    s = pd.read_csv(f, parse_dates=["date"]).set_index("date")["requests"].asfreq("D")
    s = s.interpolate()  # only for this baseline; no Spark job interpolates
    train, test = s[:-a.holdout], s[-a.holdout:]
    hw = ExponentialSmoothing(train, trend="add", seasonal="add",
                              seasonal_periods=a.season).fit()
    naive = train[-a.season:].values[: a.holdout]
    print(f"Holt-Winters MAPE      : {mape(test, hw.forecast(a.holdout)):.1f}%")
    print(f"Seasonal-naive MAPE    : {mape(test, naive):.1f}%")


if __name__ == "__main__":
    main()
