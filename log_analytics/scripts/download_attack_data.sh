#!/usr/bin/env bash
# Downloads the two labelled attack corpora into data/.
# Check the URLs still resolve before relying on them.
set -euo pipefail
mkdir -p data
# HTTP Params Dataset (Morzeux): benign values from HTTP CSIC 2010, attacks from sqlmap/xssya/Vega/FuzzDB
curl -L -o data/payload_full.csv \
  https://raw.githubusercontent.com/Morzeux/HttpParamsDataset/master/payload_full.csv
# Fwaf corpus (independent test set)
curl -L -o data/goodqueries.txt \
  https://raw.githubusercontent.com/faizann24/Fwaf-Machine-Learning-driven-Web-Application-Firewall/master/goodqueries.txt
curl -L -o data/badqueries.txt \
  https://raw.githubusercontent.com/faizann24/Fwaf-Machine-Learning-driven-Web-Application-Firewall/master/badqueries.txt
