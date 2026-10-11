"""Unsupervised bot detection: sessionise -> 9 behavioural features per host
-> Gaussian Mixture Model (k = 3). No labels are used to fit anything.

Validation: hosts that request /robots.txt are crawlers declaring themselves.
They are held out of the model entirely, then scored afterwards to see how
many the automated component captured (lift over random selection).
K-Means (k = 2 and k = 8) is run on the same table as the comparison.

Usage: python jobs/bot_detection.py --curated data/curated/nasa
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import get_spark, save_json  # noqa: E402
from pyspark.ml.clustering import GaussianMixture, KMeans  # noqa: E402
from pyspark.ml.evaluation import ClusteringEvaluator  # noqa: E402
from pyspark.ml.feature import StandardScaler, VectorAssembler  # noqa: E402
from pyspark.sql import Window, functions as F  # noqa: E402

NINE = ["head_ratio", "med_gap", "image_ratio", "n_sessions", "hits_per_session",
        "n_topdirs", "unique_ratio", "error_ratio", "gap_cv"]
SKEWED = ["med_gap", "n_sessions", "hits_per_session", "n_topdirs"]  # log1p
IMG = r"(?i)\.(gif|jpe?g|xbm|png|bmp)(\?.*)?$"


def sessionise(df):
    """A gap of more than 30 minutes between requests starts a new session."""
    w = Window.partitionBy("host").orderBy("ts")
    df = df.withColumn("gap", F.unix_timestamp("ts") - F.lag(F.unix_timestamp("ts")).over(w))
    df = df.withColumn("new_session",
                       (F.col("gap").isNull() | (F.col("gap") > 1800)).cast("int"))
    return df.withColumn("session_id", F.sum("new_session").over(w))


def host_features(df, min_requests):
    df = sessionise(df)
    df = df.withColumn("topdir", F.regexp_extract("url", r"^(/[^/?]*)", 1))
    agg = df.groupBy("host").agg(
        F.count("*").alias("n_requests"),
        F.avg((F.col("method") == "HEAD").cast("double")).alias("head_ratio"),
        F.expr("percentile_approx(gap, 0.5)").alias("med_gap"),
        F.avg(F.col("url").rlike(IMG).cast("double")).alias("image_ratio"),
        F.max("session_id").alias("n_sessions"),
        F.countDistinct("topdir").alias("n_topdirs"),
        (F.countDistinct("url") / F.count("*")).alias("unique_ratio"),
        F.avg((F.col("status") >= 400).cast("double")).alias("error_ratio"),
        (F.stddev("gap") / F.when(F.avg("gap") > 0, F.avg("gap"))).alias("gap_cv"),
        F.max((F.col("url") == "/robots.txt").cast("int")).alias("asked_robots"),
    )
    agg = agg.withColumn("hits_per_session", F.col("n_requests") / F.col("n_sessions"))
    agg = agg.where(F.col("n_requests") >= min_requests).fillna(0.0, subset=NINE)
    for c in SKEWED:  # one host with 50,000 requests must not dominate
        agg = agg.withColumn(c, F.log1p(F.col(c)))
    return agg


def automated_component(profile):
    """Pick the automated cluster from its own centroid, never from a label:
    high HEAD ratio, long regular gaps, few images."""
    score = {k: v["head_ratio"] * 50 + v["med_gap"] * 0.5 - v["image_ratio"] * 2
             for k, v in profile.items()}
    return max(score, key=score.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="data/curated/nasa")
    ap.add_argument("--min-requests", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="output/dashboard.json")
    a = ap.parse_args()

    spark = get_spark("bot_detection")
    logs = spark.read.parquet(a.curated)
    total_requests = logs.count()
    hosts = host_features(logs, a.min_requests).cache()
    n_hosts = hosts.count()
    print("hosts clustered:", n_hosts, "of", logs.select("host").distinct().count())

    # hold out self-declared crawlers: they are never shown to the model
    train_hosts = hosts.where("asked_robots = 0")
    robots_hosts = hosts.where("asked_robots = 1")
    assembler = VectorAssembler(inputCols=NINE, outputCol="feats")
    # scaling is mandatory: clustering measures distance, so the feature with
    # the largest raw numbers would otherwise decide the clusters on its own
    scaler = StandardScaler(inputCol="feats", outputCol="scaled",
                            withMean=True, withStd=True).fit(assembler.transform(train_hosts))
    prep = lambda d: scaler.transform(assembler.transform(d))  # noqa: E731
    train, held = prep(train_hosts).cache(), prep(robots_hosts)

    def profile_of(pred_df):
        rows = pred_df.groupBy("cluster").agg(
            F.count("*").alias("hosts"), F.sum("n_requests").alias("requests"),
            *[F.avg(c).alias(c) for c in NINE]).collect()
        return {r["cluster"]: r.asDict() for r in rows}

    out = {"total_requests": total_requests, "hosts_clustered": n_hosts}

    # ---- chosen model: Gaussian Mixture Model --------------------------------
    gmm = GaussianMixture(k=a.k, seed=42, featuresCol="scaled").fit(train)
    pred = gmm.transform(train).withColumnRenamed("prediction", "cluster")
    prof = profile_of(pred)
    auto = automated_component(prof)
    silhouette = ClusteringEvaluator(featuresCol="scaled", predictionCol="cluster").evaluate(pred)
    n_train = train.count()
    tot_req = pred.agg(F.sum("n_requests")).first()[0]
    out["gmm"] = {"k": a.k, "silhouette": silhouette, "automated_component": auto,
                  "clusters": {str(k): {**v, "share_of_hosts": v["hosts"] / n_train,
                                        "share_of_traffic": v["requests"] / tot_req}
                               for k, v in prof.items()}}
    # validation on the held-out robots.txt hosts
    if held.count():
        hp = gmm.transform(held).withColumnRenamed("prediction", "cluster")
        caught = hp.where(F.col("cluster") == auto).count()
        base = prof[auto]["hosts"] / n_train       # random-selection hit rate
        rate = caught / held.count()
        out["robots_validation"] = {"held_out_hosts": held.count(), "captured": caught,
                                    "lift_over_random": rate / base if base else None}
    print("GMM silhouette", silhouette, "automated component", auto)

    # ---- comparison: K-Means k = 2 and k = 8 ---------------------------------
    out["kmeans"] = {}
    for k in (2, 8):
        km = KMeans(k=k, seed=42, featuresCol="scaled").fit(train)
        kp = km.transform(train).withColumnRenamed("prediction", "cluster")
        kprof = profile_of(kp)
        kauto = automated_component(kprof)
        rh = km.transform(held).withColumnRenamed("prediction", "cluster") if held.count() else None
        rec = (rh.where(F.col("cluster") == kauto).count() / held.count()) if rh is not None else None
        base = kprof[kauto]["hosts"] / n_train
        out["kmeans"][str(k)] = {"flagged_cluster_hosts": kprof[kauto]["hosts"],
                                 "robots_recall": rec,
                                 "lift": (rec / base) if (rec is not None and base) else None}
        print("KMeans", k, out["kmeans"][str(k)])

    save_json(out, a.out)
    spark.stop()


if __name__ == "__main__":
    main()
