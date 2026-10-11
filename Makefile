PY ?= python

# ---- Part 1: classification ----
attack-data:  ; bash scripts/download_attack_data.sh
attack-etl:   ; $(PY) jobs/attack_etl.py
classify:     ; $(PY) jobs/classify_fm.py && $(PY) jobs/mlp_generalisation.py
stream:       ; $(PY) jobs/stream_score.py --source data/stream_in

# ---- Part 2: clustering ----
data:         ; bash scripts/download_nasa_logs.sh
skeleton:     ; $(PY) jobs/etl.py && $(PY) jobs/reports.py
bots:         ; $(PY) jobs/bot_detection.py && $(PY) jobs/forecast_baseline.py

.PHONY: attack-data attack-etl classify stream data skeleton bots
