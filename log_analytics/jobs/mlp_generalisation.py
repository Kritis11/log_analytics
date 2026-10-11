"""Independent-corpus test (Fwaf) and the fix.

1. Score models trained on the original corpus against Fwaf (collected by
   different people for a different project, never used in training).
2. Split Fwaf in half: half A may be used for training, half B is never
   trained on and is where every model is scored.
3. Retrain with Fwaf-A benign traffic added and score on the same half B.

Fwaf files: goodqueries.txt / badqueries.txt (one request per line).
Usage: python jobs/mlp_generalisation.py --good data/goodqueries.txt --bad data/badqueries.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import models  # noqa: E402
from common import featurise, get_spark, save_json, to_vector  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def attack_metrics(pred_df):
    """Binary view: predicted class != norm (0) means 'attack'."""
    r = pred_df.select(
        F.col("is_attack"), (F.col("prediction") != 0).cast("int").alias("flag")
    ).groupBy("is_attack", "flag").count().collect()
    c = {(int(x["is_attack"]), int(x["flag"])): x["count"] for x in r}
    tp, fn = c.get((1, 1), 0), c.get((1, 0), 0)
    fp, tn = c.get((0, 1), 0), c.get((0, 0), 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"detection": rec, "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "precision": prec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "n": tp + fn + fp + tn}


def load_fwaf(spark, good, bad):
    g = spark.read.text(good).select(F.col("value").alias("payload"), F.lit(0).alias("is_attack"))
    b = spark.read.text(bad).select(F.col("value").alias("payload"), F.lit(1).alias("is_attack"))
    df = g.union(b).where(F.length(F.trim("payload")) > 0).dropDuplicates(["payload"])
    return to_vector(featurise(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="data/curated/attacks")
    ap.add_argument("--good", default="data/goodqueries.txt")
    ap.add_argument("--bad", default="data/badqueries.txt")
    ap.add_argument("--test-size", type=int, default=44327)
    ap.add_argument("--results", default="output/generalisation_results.json")
    a = ap.parse_args()

    spark = get_spark("mlp_generalisation")
    orig = to_vector(spark.read.parquet(a.curated)).select("label", "raw")
    fwaf = load_fwaf(spark, a.good, a.bad).cache()
    half_a, half_b = fwaf.randomSplit([0.5, 0.5], seed=21)
    n_b = half_b.count()
    test = half_b.sample(False, min(1.0, a.test_size / max(n_b, 1)), seed=5).cache()
    print("held-out Fwaf rows:", test.count())

    builders = [("Multilayer Perceptron", models.mlp),
                ("Factorization Machines", models.factorization_machines),
                ("Random Forest", lambda: models.random_forest(160))]
    # realistic benign traffic from half A, labelled 'norm' (class 0)
    extra = (half_a.where("is_attack = 0")
             .select(F.lit(0.0).alias("label"), "raw"))

    out = {"original_corpus_only": {}, "plus_realistic_benign": {}}
    for name, build in builders:
        base = build().fit(orig)
        out["original_corpus_only"][name] = attack_metrics(base.transform(test))
        fixed = build().fit(orig.unionByName(extra))
        out["plus_realistic_benign"][name] = attack_metrics(fixed.transform(test))
        print(name, out["original_corpus_only"][name], "->",
              out["plus_realistic_benign"][name])
    save_json(out, a.results)
    spark.stop()


if __name__ == "__main__":
    main()
