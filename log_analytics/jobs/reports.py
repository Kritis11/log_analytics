"""Eleven Spark SQL report tables on the curated NASA-HTTP data.

Usage: python jobs/reports.py --curated data/curated/nasa --out output/reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import get_spark  # noqa: E402

REPORTS = {
    "requests_per_day": "SELECT date, COUNT(*) AS requests FROM logs GROUP BY date ORDER BY date",
    "bytes_per_day": "SELECT date, SUM(bytes) AS bytes FROM logs GROUP BY date ORDER BY date",
    "requests_per_hour": "SELECT hour(ts) AS hour, COUNT(*) AS requests FROM logs GROUP BY 1 ORDER BY 1",
    "status_codes": "SELECT status, COUNT(*) AS requests FROM logs GROUP BY status ORDER BY requests DESC",
    "methods": "SELECT method, COUNT(*) AS requests FROM logs GROUP BY method ORDER BY requests DESC",
    "top_urls": "SELECT url, COUNT(*) AS requests FROM logs GROUP BY url ORDER BY requests DESC LIMIT 20",
    "top_hosts": "SELECT host, COUNT(*) AS requests, SUM(bytes) AS bytes FROM logs GROUP BY host ORDER BY requests DESC LIMIT 20",
    "top_404_urls": "SELECT url, COUNT(*) AS misses FROM logs WHERE status = 404 GROUP BY url ORDER BY misses DESC LIMIT 20",
    "error_rate_per_day": "SELECT date, AVG(CAST(status >= 400 AS DOUBLE)) AS error_rate FROM logs GROUP BY date ORDER BY date",
    "file_types": ("SELECT lower(regexp_extract(url, '\\\\.([A-Za-z0-9]+)(\\\\?.*)?$', 1)) AS ext, "
                   "COUNT(*) AS requests FROM logs GROUP BY 1 ORDER BY requests DESC LIMIT 15"),
    "unique_visitors_per_day": "SELECT date, COUNT(DISTINCT host) AS visitors FROM logs GROUP BY date ORDER BY date",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="data/curated/nasa")
    ap.add_argument("--out", default="output/reports")
    a = ap.parse_args()
    spark = get_spark("reports")
    spark.read.parquet(a.curated).createOrReplaceTempView("logs")
    for name, sql in REPORTS.items():
        df = spark.sql(sql)
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{a.out}/{name}")
        print(f"== {name}")
        df.show(5, truncate=60)
    spark.stop()


if __name__ == "__main__":
    main()
