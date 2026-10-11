"""Spark Structured Streaming scorer: 5-second micro-batches, checkpointed.

Watches a directory for new text files (one raw request per line), featurises
each line, scores it with the saved MLP and writes Parquet + JSON.

Usage: python jobs/stream_score.py --source data/stream_in
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import LABELS, featurise, get_spark, to_vector  # noqa: E402
from pyspark.ml import PipelineModel  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/stream_in")
    ap.add_argument("--model", default="models/mlp")
    ap.add_argument("--out", default="output/stream")
    ap.add_argument("--checkpoint", default="output/_checkpoints/stream")
    ap.add_argument("--trigger", default="5 seconds")
    a = ap.parse_args()

    spark = get_spark("stream_score", shuffle_partitions=4)
    model = PipelineModel.load(a.model)
    names = F.array(*[F.lit(l) for l in LABELS])

    stream = (spark.readStream.format("text").option("maxFilesPerTrigger", 1)
              .load(a.source).withColumnRenamed("value", "payload"))
    scored = model.transform(to_vector(featurise(stream)))
    scored = scored.select(
        F.current_timestamp().alias("scored_at"), "payload",
        names[F.col("prediction").cast("int")].alias("predicted_class"))

    q1 = (scored.writeStream.format("parquet").outputMode("append")
          .option("path", a.out + "/parquet")
          .option("checkpointLocation", a.checkpoint + "/parquet")
          .trigger(processingTime=a.trigger).start())
    q2 = (scored.writeStream.format("json").outputMode("append")
          .option("path", a.out + "/json")
          .option("checkpointLocation", a.checkpoint + "/json")
          .trigger(processingTime=a.trigger).start())
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
