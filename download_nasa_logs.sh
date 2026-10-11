#!/usr/bin/env bash
# NASA-HTTP logs (July and August 1995). The original host (ita.ee.lbl.gov)
# is sometimes down; if so, download the two files from the Kaggle mirror
# "NASA HTTP logs" and place them, unzipped, in data/nasa/.
set -euo pipefail
mkdir -p data/nasa
for m in Jul95 Aug95; do
  curl -L -o data/nasa/NASA_access_log_$m.gz \
    ftp://ita.ee.lbl.gov/traces/NASA_access_log_$m.gz || true
  [ -s data/nasa/NASA_access_log_$m.gz ] && gunzip -f data/nasa/NASA_access_log_$m.gz
done
ls -lh data/nasa
