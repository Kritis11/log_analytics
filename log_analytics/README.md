# Log Analytics System for a Web Server

Classification and clustering of web-server traffic with Apache Spark (PySpark / MLlib).

* **Part 1 - Classification:** every request is reduced to 42 explainable features and classified into
  `norm`, `sqli`, `xss`, `path-traversal` or `cmdi` by a Multilayer Perceptron (42-32-16-5).
  Eight models are compared on the same stratified 80/20 split; the model is then tested on an
  independent corpus (Fwaf) to check that it has not simply learnt the attack tools' signatures.
* **Part 2 - Clustering:** NASA-HTTP logs are sessionised, nine behavioural features are built per host,
  and a Gaussian Mixture Model (k = 3) isolates automated traffic without any labels.

## Layout
```
jobs/
  features.py            42 features (29 structural + 13 keyword), pure Python
  common.py              Spark session, UDF featurisation, macro-F1, latency helpers
  models.py              MLP, Factorization Machines (OvR), LR, NB, RF, Decision Tree, Cascade Forest
  attack_etl.py          labelled corpus -> Parquet with features attached
  classify_fm.py         eight-model bake-off + feature-group experiment; saves the MLP
  mlp_generalisation.py  independent-corpus test (Fwaf) and the fix
  stream_score.py        Structured Streaming scorer, 5-second micro-batches
  etl.py                 NASA-HTTP logs -> date-partitioned Parquet
  reports.py             eleven Spark SQL report tables
  bot_detection.py       sessionisation, 9 host features, GMM, K-Means comparison, robots.txt validation
  forecast_baseline.py   Holt-Winters vs seasonal-naive baseline
scripts/                 dataset download helpers
```

## Setup
Requires Java 11/17 and Python 3.9-3.11.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run - classification
```bash
make attack-data     # download HTTP Params Dataset + Fwaf into data/
make attack-etl      # parse and featurise -> data/curated/attacks
make classify        # bake-off (-> output/classification_results.json), then generalisation test
make stream          # score files dropped in data/stream_in/ (needs the saved model in models/mlp)
```

## Run - clustering
```bash
make data            # download NASA-HTTP logs into data/nasa/
make skeleton        # ingest, parse, run the SQL reports
make bots            # fit the GMM (-> output/dashboard.json) and the forecast baseline
```

Every job takes `--help`-style arguments for input/output paths (see the top of each file).
Jobs run in Spark local mode by default; to use a cluster, launch them with `spark-submit --master <url> --py-files jobs/features.py,jobs/common.py,jobs/models.py jobs/<job>.py`.

## Data
| Dataset | Use |
|---|---|
| HTTP Params Dataset (Morzeux) - benign values from HTTP CSIC 2010, attacks from sqlmap, xssya, Vega, FuzzDB | training and in-corpus test |
| Fwaf corpus (`goodqueries.txt`, `badqueries.txt`) | independent generalisation test |
| NASA-HTTP logs (Jul-Aug 1995) | clustering |

Datasets are not stored in this repository (`data/` is git-ignored); use the download scripts.

## Results
Run the jobs to regenerate the numbers; they are written to `output/*.json`.
Fill the final figures from your own run here:

| Item | Value |
|---|---|
| MLP macro-F1 / accuracy / ms per 1,000 requests | |
| GMM silhouette (k = 3) | |
| Hosts in automated cluster / robots.txt lift | |

## Known limitations
* Attack payloads in the training corpus were produced by security tools, not captured in the wild;
  this is why the in-corpus score overstates real-world performance (see `mlp_generalisation.py`).
* Command injection has very few examples, so its recall is not reliable.
* The `/robots.txt` validation is a weak label: it gives a lower bound on recall, not a true recall.
* The NCSA Common Log Format has no response-time field, so response weight is measured in bytes.
