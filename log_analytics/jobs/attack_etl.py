"""Labelled attack corpus -> Parquet with the 42 features attached.

Usage: python jobs/attack_etl.py [--src data/payload_full.csv] [--out data/curated/attacks]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import IDX, LABELS, featurise, get_spark  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

ALIASES = {"norm": "norm", "normal": "norm", "benign": "norm",
           "sqli": "sqli", "xss": "xss", "cmdi": "cmdi",
           "path-traversal": "path-traversal", "path_traversal": "path-traversal",
           "pathtraversal": "path-traversal"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/payload_full.csv")
    ap.add_argument("--out", default="data/curated/attacks")
    ap.add_argument("--label-col", default="attack_type")
    a = ap.parse_args()

    spark = get_spark("attack_etl")
    # 2. Parse: read the CSV, normalise labels, drop empty payloads
    df = (spark.read.option("header", True).option("multiLine", False)
          .option("escape", '"').csv(a.src))
    df = (df.select(F.col("payload"),
                    F.lower(F.trim(F.col(a.label_col))).alias("label_name"))
          .where(F.col("payload").isNotNull() & (F.length(F.trim("payload")) > 0)))
    mapping = F.create_map(*[x for k, v in ALIASES.items()
                             for x in (F.lit(k), F.lit(v))])
    df = df.withColumn("label_name", mapping[F.col("label_name")])
    df = df.where(F.col("label_name").isin(LABELS))
    idx_map = F.create_map(*[x for k, v in IDX.items()
                             for x in (F.lit(k), F.lit(float(v)))])
    df = df.withColumn("label", idx_map[F.col("label_name")])
    # 3. Featurise: UDF turns each request into 42 numbers
    df = featurise(df)
    # 4. Curate: written as Parquet so features are computed once
    df.write.mode("overwrite").parquet(a.out)
    print("rows written:", spark.read.parquet(a.out).count())
    spark.read.parquet(a.out).groupBy("label_name").count().show()
    spark.stop()


if __name__ == "__main__":
    main()
