"""The eight-model bake-off, then save the chosen classifier (the MLP).

All models are trained on the same stratified 80/20 split and scored on the
same held-out test set. Metric: macro-F1 (a rare class cannot be ignored for
free), plus accuracy, per-class recall with support, and latency in
milliseconds per 1,000 requests.

Usage: python jobs/classify_fm.py [--curated data/curated/attacks]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import models  # noqa: E402
from common import IDX, get_spark, latency_ms_per_1k, macro_f1, save_json, to_vector  # noqa: E402

TREE_ONLY = {"Random Forest (480 trees)", "Random Forest (160 trees)",
             "Cascade Forest (Deep Forest)", "Decision Tree"}


def candidates(skip_trees=False):
    c = [("Multilayer Perceptron 42-32-16-5", "Neural network", models.mlp),
         ("Factorization Machines (One-vs-Rest)", "Latent factor",
          models.factorization_machines),
         ("Logistic Regression", "Linear", models.logistic_regression),
         ("Naive Bayes", "Probabilistic", models.naive_bayes)]
    if not skip_trees:  # measured as the comparison, never deployed
        c += [("Random Forest (480 trees)", "Tree - excluded", lambda: models.random_forest(480)),
              ("Random Forest (160 trees)", "Tree - excluded", lambda: models.random_forest(160)),
              ("Cascade Forest (Deep Forest)", "Tree - excluded", models.CascadeForest),
              ("Decision Tree", "Tree - excluded", models.decision_tree)]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="data/curated/attacks")
    ap.add_argument("--model-out", default="models/mlp")
    ap.add_argument("--results", default="output/classification_results.json")
    ap.add_argument("--skip-trees", action="store_true")
    a = ap.parse_args()

    spark = get_spark("classify_bakeoff")
    df = to_vector(spark.read.parquet(a.curated)).select("payload", "label", "raw")
    # the split is stratified: command injection has only 89 rows in total,
    # so a plain random split could leave it almost absent from the test set
    train = df.sampleBy("label", {float(i): 0.8 for i in IDX.values()}, seed=7).cache()
    test = df.subtract(train).cache()
    print("train rows:", train.count(), " test rows:", test.count())

    results, chosen = [], None
    for name, family, build in candidates(a.skip_trees):
        t0 = time.time()
        est = build()
        model = est.fit(train)
        m = macro_f1(model.transform(test))
        m.update(model=name, family=family,
                 ms_per_1k=latency_ms_per_1k(model, test),
                 fit_seconds=time.time() - t0)
        results.append(m)
        print(f"{name:40s} macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
              f"{m['ms_per_1k']:.1f} ms/1k")
        if name.startswith("Multilayer"):
            chosen = model
    if chosen is not None:
        chosen.write().overwrite().save(a.model_out)
        print("saved MLP pipeline to", a.model_out)

    # Feature-group experiment: do we need attack vocabulary at all?
    # Same tree baseline on all 42 features vs the 29 structural ones.
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import VectorSlicer
    ablation = []
    for label, idx in (("All features (42)", list(range(42))),
                       ("Structural only (29)", list(range(29)))):
        sl = VectorSlicer(inputCol="raw", outputCol="sub", indices=idx)
        rf = RandomForestClassifier(labelCol="label", featuresCol="sub",
                                    numTrees=160, maxDepth=12, seed=42)
        mdl = Pipeline(stages=[sl, rf]).fit(train)
        r = macro_f1(mdl.transform(test))
        ablation.append({"feature_set": label, "count": len(idx),
                         "macro_f1": r["macro_f1"],
                         "ms_per_1k": latency_ms_per_1k(mdl, test)})
        print(ablation[-1])

    save_json({"models": results, "feature_ablation": ablation}, a.results)
    spark.stop()


if __name__ == "__main__":
    main()
