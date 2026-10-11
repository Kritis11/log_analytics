"""NASA-HTTP (NCSA Common Log Format) -> date-partitioned Parquet.

Line format:
  host - - [01/Jul/1995:00:00:01 -0400] "GET /path HTTP/1.0" 200 6245

Usage: python jobs/etl.py --src data/nasa --out data/curated/nasa
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import get_spark  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

CLF = (r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+)\s*(\S*)\s*(\S*)" (\d{3}) (\S+)$')


def parse(spark, src):
    raw = spark.read.text(src)
    p = raw.select(
        F.regexp_extract("value", CLF, 1).alias("host"),
        F.regexp_extract("value", CLF, 2).alias("ts_raw"),
        F.regexp_extract("value", CLF, 3).alias("method"),
        F.regexp_extract("value", CLF, 4).alias("url"),
        F.regexp_extract("value", CLF, 6).alias("status"),
        F.regexp_extract("value", CLF, 7).alias("bytes_raw"),
        "value")
    ok = p.where(F.col("host") != "")
    df = ok.select(
        "host",
        F.to_timestamp("ts_raw", "dd/MMM/yyyy:HH:mm:ss Z").alias("ts"),
        "method", "url",
        F.col("status").cast("int").alias("status"),
        F.when(F.col("bytes_raw") == "-", 0).otherwise(F.col("bytes_raw")).cast("long").alias("bytes"),
    ).where(F.col("ts").isNotNull())
    df = df.withColumn("date", F.to_date("ts"))
    return raw.count(), df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/nasa")
    ap.add_argument("--out", default="data/curated/nasa")
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N parsed rows (0 = all)")
    a = ap.parse_args()
    spark = get_spark("nasa_etl")
    total, df = parse(spark, a.src)
    if a.limit:
        df = df.limit(a.limit)
    df.write.mode("overwrite").partitionBy("date").parquet(a.out)
    n = spark.read.parquet(a.out).count()
    print(f"parsed {n} of {total} lines ({100.0 * n / max(total, 1):.2f}% parse rate)")
    spark.stop()


if __name__ == "__main__":
    main()
