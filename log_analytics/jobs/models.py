"""Model builders shared by the bake-off and the generalisation test.

Scaling rule: trees ignore feature scale, gradient-based models do not.
Every deployed model is gradient-based, so features are standardised
first (zero mean, unit variance). Naive Bayes needs non-negative input,
so it gets min-max scaling instead.
"""
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier, FMClassifier, LogisticRegression,
    MultilayerPerceptronClassifier, NaiveBayes, OneVsRest,
    RandomForestClassifier)
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

N_FEATURES = 42
N_CLASSES = 5


def _std():
    return StandardScaler(inputCol="raw", outputCol="features",
                          withMean=True, withStd=True)


def mlp():
    return Pipeline(stages=[_std(), MultilayerPerceptronClassifier(
        labelCol="label", featuresCol="features",
        layers=[N_FEATURES, 32, 16, N_CLASSES],  # inputs -> hidden -> hidden -> classes
        maxIter=160, seed=42)])


def factorization_machines():
    fm = FMClassifier(labelCol="label", featuresCol="features",
                      factorSize=8, maxIter=100, seed=42)
    return Pipeline(stages=[_std(), OneVsRest(classifier=fm, labelCol="label",
                                              featuresCol="features")])


def logistic_regression():
    return Pipeline(stages=[_std(), LogisticRegression(
        labelCol="label", featuresCol="features", maxIter=200,
        family="multinomial")])


def naive_bayes():
    mm = MinMaxScaler(inputCol="raw", outputCol="features")
    return Pipeline(stages=[mm, NaiveBayes(labelCol="label",
                                           featuresCol="features",
                                           modelType="multinomial")])


def _raw_tree(est):
    # trees are scale-invariant: feed the raw vector straight in
    est.setFeaturesCol("raw")
    return Pipeline(stages=[est])


def random_forest(trees):
    return _raw_tree(RandomForestClassifier(labelCol="label", numTrees=trees,
                                            maxDepth=12, seed=42))


def decision_tree():
    return _raw_tree(DecisionTreeClassifier(labelCol="label", maxDepth=12,
                                            seed=42))


# --------------------------------------------------------------------------
# Cascade Forest (Deep Forest): each layer is several forests whose class
# probabilities are appended to the raw features and passed to the next
# layer. Out-of-fold probabilities are used while training so a layer never
# sees predictions made on rows its own forests were fitted on.
# --------------------------------------------------------------------------
class CascadeForest:
    def __init__(self, layers=2, forests_per_layer=2, trees=80, folds=3):
        self.layers, self.k, self.trees, self.folds = (
            layers, forests_per_layer, trees, folds)
        self.fitted = []  # per layer: list of fitted forest models

    def _forests(self, layer):
        return [RandomForestClassifier(
            labelCol="label", featuresCol="cf_in", numTrees=self.trees,
            maxDepth=12, seed=100 * layer + i,
            featureSubsetStrategy="sqrt" if i % 2 == 0 else "log2")
            for i in range(self.k)]

    def _add_probs(self, df, models):
        """Run every forest over df; return df with prob0..probK columns."""
        for j, m in enumerate(models):
            df = (m.transform(df).withColumnRenamed("probability", f"prob{j}")
                  .drop("rawPrediction", "prediction"))
        return df

    def _next_input(self, df):
        out = VectorAssembler(
            inputCols=["cf_in"] + [f"prob{j}" for j in range(self.k)],
            outputCol="cf_next").transform(df)
        drop = ["cf_in"] + [f"prob{j}" for j in range(self.k)]
        return out.drop(*drop).withColumnRenamed("cf_next", "cf_in")

    def fit(self, df):
        df = df.withColumn("cf_in", F.col("raw"))
        for layer in range(self.layers):
            # forests kept for inference are fitted on all of the training data
            self.fitted.append([e.fit(df) for e in self._forests(layer)])
            # the next layer is trained on out-of-fold probabilities
            parts = df.randomSplit([1.0] * self.folds, seed=11 + layer)
            held_out = []
            for f in range(self.folds):
                train = None
                for g, part in enumerate(parts):
                    if g != f:
                        train = part if train is None else train.union(part)
                models = [e.fit(train) for e in self._forests(layer)]
                held_out.append(self._add_probs(parts[f], models))
            df = held_out[0]
            for part in held_out[1:]:
                df = df.union(part)
            df = self._next_input(df).cache()
        return self

    def transform(self, df):
        df = df.withColumn("cf_in", F.col("raw"))
        for layer, models in enumerate(self.fitted):
            df = self._add_probs(df, models)
            if layer < len(self.fitted) - 1:
                df = self._next_input(df)
        # final class = argmax of the averaged probabilities of the last layer
        arrs = [vector_to_array(F.col(f"prob{j}")) for j in range(self.k)]
        avg = F.array(*[sum(a[i] for a in arrs) / self.k
                        for i in range(N_CLASSES)])
        return (df.withColumn("avg_prob", avg)
                .withColumn("prediction", F.expr(
                    "cast(array_position(avg_prob, array_max(avg_prob)) - 1 as double)")))
