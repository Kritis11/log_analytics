"""Shared Spark helpers used by every job."""
import json
import os
import sys
import time

import features
from pyspark.ml.functions import array_to_vector
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, DoubleType

LABELS = ["norm", "sqli", "xss", "path-traversal", "cmdi"]
IDX = {l: i for i, l in enumerate(LABELS)}


def get_spark(name, shuffle_partitions=16):
    # make workers use the same interpreter as the driver
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (SparkSession.builder.appName(name)
             .config("spark.sql.shuffle.partitions", shuffle_partitions)
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    # the UDF runs in Python workers on every executor, so features.py
    # must travel with the job or it fails with ModuleNotFoundError
    spark.sparkContext.addPyFile(features.__file__)
    return spark


def featurise(df, col="payload"):
    """Add a `feats` column: array of the 42 features for each request."""
    udf = F.udf(features.build, ArrayType(DoubleType()))
    return df.withColumn("feats", udf(F.col(col)))


def to_vector(df):
    """array<double> `feats` -> ML Vector `raw` (what the pipelines expect)."""
    return df.withColumn("raw", array_to_vector("feats"))


def macro_f1(pred_df, label_col="label", pred_col="prediction", n_classes=5):
    """Macro-averaged F1 plus accuracy and per-class recall/support."""
    rows = (pred_df.groupBy(label_col, pred_col).count().collect())
    cm = {(int(r[label_col]), int(r[pred_col])): r["count"] for r in rows}
    total = sum(cm.values())
    f1s, per_class = [], {}
    for c in range(n_classes):
        tp = cm.get((c, c), 0)
        fp = sum(cm.get((o, c), 0) for o in range(n_classes) if o != c)
        fn = sum(cm.get((c, o), 0) for o in range(n_classes) if o != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        per_class[LABELS[c]] = {"support": tp + fn, "recall": r, "precision": p}
    acc = sum(cm.get((c, c), 0) for c in range(n_classes)) / total
    return {"macro_f1": sum(f1s) / n_classes, "accuracy": acc,
            "per_class": per_class}


def latency_ms_per_1k(model, test_df):
    """Wall-clock time to score the whole test set, scaled to 1,000 rows."""
    test_df = test_df.cache()
    n = test_df.count()
    t0 = time.perf_counter()
    model.transform(test_df).select("prediction").count()
    dt = time.perf_counter() - t0
    return dt * 1000.0 / n * 1000.0


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print("wrote", path)
