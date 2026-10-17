# anomaly-detection: fix list

Audit of `main` at `7fd6b3f` (2026-10-17). Every claim below was checked by reading the code, running the test suite, or running the full pipeline on a clean copy of the repo. Line numbers refer to that commit.

## What I ran and what came back

- `python -m pytest`: 141 passed, 1089 warnings (almost all pandas `PerformanceWarning` from `features.py`).
- `python run_pipeline.py` on a **fresh clone**: crashes at step 1 (`OSError: Cannot save file into a non-existent directory: 'outputs'`). After creating `outputs/` by hand it runs in about 2 minutes.
- Test-set results (seed 295, selected threshold 0.40): precision 0.965, recall 0.941, F1 0.953, ROC-AUC 0.986, PR-AUC 0.979. Test anomaly rate is 25%.
- Per-type test recall: stuck_sensor 1.00, drop 1.00, impossible_value 1.00, oscillation 0.93, drift 0.90, spike 0.81 (spike has only 31 test rows).
- All 82 test false positives occur within 30 steps after an anomaly ended on the same machine (80 of them within 10 steps). Zero false positives on clean normal data.
- Alerts: 2351 row alerts, 114 events. 32 of the 114 events (28%) are false positives, so **event-level precision is 0.72**, not the 0.965 row-level number.
- 129 sensor readings in the test window break a hard safety limit (`SAFETY_THRESHOLDS`) and produce **no alert at all**, because the model scored those rows 0.
- `POST /predict` with `{"features": {}}` returns `prediction: 1, anomaly_score: 0.83`. An empty request is classified as an anomaly.

The headline problem: the alerting layer reads the ground-truth labels. The dashboard's sensor and anomaly-type charts are the generator's answers, not the model's output. Fix that before anything else, because it invalidates what the project claims to demonstrate.

---

## P0: wrong results or broken out of the box

### P0-1. Alerts use ground-truth labels to pick the sensor and anomaly type
- **Where:** `alerts.py:229` (`sensor = row['target_sensor']`), `alerts.py:248` and `:281` (`anomaly_type: row['anomaly_type']`), `alert_events.py:67` and `:191` (events grouped by `anomaly_type`), `db_queries.py:95`.
- **What's wrong:** `target_sensor` and `anomaly_type` are generator labels. The model only outputs one machine-level score per row. Every alert is attributed to the sensor the generator corrupted, and every event is labeled with the true anomaly type. In production you would not have either.
- **Evidence:** all 82 false positives land on sensor `"none"` with the "target sensor was not recognized" reason; every true positive is attributed to exactly the right sensor.
- **Why it matters:** the "Sensor Distribution" and "Anomaly Type Distribution" charts, event grouping, and event drill-down are all label leakage. Anyone reviewing the project who reads `alerts.py` will catch this.
- **Fix:** make the model produce what the alert needs. Simplest route: the per-sensor labels `{sensor}_anomaly` already exist in the data, so train one classifier per sensor (or a `MultiOutputClassifier`) and alert on whichever sensor scores above threshold. If you want anomaly type on the dashboard, add a separate multi-class classifier trained on `anomaly_type` and show its prediction, labeled as predicted. Keep `target_sensor`/`anomaly_type` only in an evaluation column (e.g. `true_sensor`, `true_anomaly_type`) that the alert logic never reads. Then group events by `(machine_id, predicted_sensor, step gap)`.

### P0-2. Hard safety-limit violations are never alerted unless the model also fires
- **Where:** `alerts.py:330-332` (only `prediction == 1` rows are considered), `alerts.py:257` (only the target sensor is checked).
- **What's wrong:** `check_violation` runs only on rows the model flagged, and only on one sensor. A reading above `critical_high` on a row the model scored 0 produces nothing. `ALERT_TYPE_THRESHOLD` (`alerts.py:43`) is defined and never used.
- **Evidence:** 129 sensor readings in the test window exceed a warning or critical limit on rows with `prediction == 0`; 10 more are on non-target sensors of flagged rows. None of them alert.
- **Fix:** run the rule check on every row and every sensor independently of the model. Emit a `threshold_violation` alert when only the rule fires, `model_anomaly` when only the model fires, `model_and_threshold` when both. Rules are the safety net; they must not depend on the model.

### P0-3. False positives get higher severity than real low-confidence detections
- **Where:** `alerts.py:244` (unknown sensor hardcoded to `SEVERITY_WARNING`), `alerts.py:200-203` (`assign_model_severity` returns `INFO` below 0.70).
- **Evidence:** all 82 false-positive alerts are WARNING; 335 true-positive alerts are INFO. 32 FP events are WARNING.
- **Fix:** goes away once P0-1 is fixed (there is no "unknown sensor" path). Until then, the fallback should be the lowest severity, not WARNING.

### P0-4. Fresh clone cannot run the pipeline
- **Where:** `generator.py:308` (`save_sensor_data` calls `to_csv` with no `mkdir`), also `features.py:297`, `alerts.py:348`, `alert_events.py` `save_alert_events`.
- **Evidence:** `python generator.py` on a clean checkout raises `OSError` because `outputs/` is gitignored.
- **Fix:** `Path(output_path).parent.mkdir(parents=True, exist_ok=True)` in every save helper (`model.save_dataframe` already does this; reuse it). Add a CI job that runs the pipeline end to end on a small dataset (see P1-9) so this cannot regress.

### P0-5. Running `pytest` overwrites your real model artifacts with a toy model
- **Where:** `model.py:256-287` writes to the global `ARTIFACT_DIR` regardless of the paths passed in; `tests/test_model.py:168-190` calls `run_model_pipeline` and even reads `artifacts/model_metadata.json` from the repo.
- **Evidence:** after `pytest`, `artifacts/feature_columns.json` contains `["temperature", "temperature_delta"]`. If you trained the real model first, it is gone, and `/predict` silently serves the 2-feature toy.
- **Fix:** add an `artifact_dir` parameter to `run_model_pipeline` and pass `tmp_path` in the test. Same pattern for every module that writes to a hardcoded path.

### P0-6. `build_features` ignores its own sort
- **Where:** `features.py:20`: `df.sort_values(...).reset_index(drop=True)` result is discarded.
- **Evidence:** feeding the same data shuffled produces different features: 484 of 485 `temperature_delta` values differ.
- **Why it matters:** it only works today because the generator happens to emit rows in step order. Any other input (DB export, real data, `/predict` batch path later) gets garbage features with no error.
- **Fix:** `df = df.sort_values(['machine_id', 'step']).reset_index(drop=True)`. Add a test that shuffles input and asserts identical output.

### P0-7. `/predict` accepts garbage and calls it an anomaly
- **Where:** `model_serving.py:20-27`, `schemas.py` `PredictionRequest`.
- **What's wrong:** missing features are filled with 0, unknown keys are silently dropped, and there is no size limit. The README (`README.md:460`) documents the zero-fill as intended behavior.
- **Evidence:** `{}` and `{"bogus": 1e9}` both return `prediction: 1, score: 0.83`.
- **Fix:** return 422 if any expected feature is missing or any key is unknown (list them in the error). Reject NaN/inf. Update the README.

---

## P1: methodology problems that make the reported metrics misleading

### P1-1. The labels are wrong right after every anomaly ends
- **Where:** `generator.py:285-300`. When `remaining_duration` hits 0 the label flips to normal, but the value is still wherever the spike/drop/drift/impossible_value left it. `step_normal` mean-reverts at 10% per step, so the signal takes roughly 20 to 30 steps to recover while being labeled normal. Clipped `impossible_value` (e.g. voltage 0 becomes 55 on the next step) is the worst case.
- **Evidence:** 100% of test false positives fall in that recovery window.
- **Why it matters:** your false-positive count measures generator artifacts, not model behavior. It also teaches the model that "obviously abnormal value" is sometimes normal.
- **Fix:** pick one and document it. (a) Reset the sensor to a normal value when the anomaly ends. (b) Extend the label through a recovery period until the value is back within the normal band. (c) Keep the labels but exclude a recovery window from scoring and report it separately. (b) is the most realistic.

### P1-2. Only row-level metrics are reported, and they overstate the alerting quality
- **Where:** `model.py:76-100`, `evaluate.py`.
- **What's wrong:** rows inside one 100-step drift are 100 highly correlated samples. Operators care about events: was the anomaly caught, how fast, how many false events per machine-day.
- **Evidence:** row precision 0.965 vs event precision 0.72.
- **Fix:** add event-level recall (event detected if at least one alert overlaps it), event precision (alert events overlapping no true event), detection latency (steps from true start to first alert, median and p90 by type), and false events per 1000 machine-steps. Report those in the README as the primary numbers. Drop `accuracy`; at a 25% base rate it says nothing.

### P1-3. The purge gap is justified wrongly and does not cover the real leakage
- **Where:** `config.py:13` (`PURGE_GAP_STEPS = 50`), `README.md:250`.
- **What's wrong:** the README says the gap stops rolling windows crossing split boundaries. The features are backward-looking only, so a validation row using training-period history is not leakage (you would have that history in production). The actual max lookback is 78 steps anyway (train starts at step 79 after `dropna`), so 50 would not cover it even if it mattered. The real risk is a single anomaly event appearing in two splits, and drift/stuck_sensor run up to 100 steps.
- **Evidence:** 2 anomaly events span the entire second purge gap, so the same event is in both validation and test; 13 events touch a purge region.
- **Fix:** set the gap to at least the max anomaly duration (100), or drop any event that crosses a boundary from both sides. Rewrite the README sentence to describe what the gap actually prevents.

### P1-4. Some anomaly types are trivially detectable by construction
- **Where:** `generator.py` `stuck_sensor` (returns the exact same float every step, so `delta == 0.0` exactly), `impossible_value` (negative pressure/flow or exactly 0 V).
- **Evidence:** recall 1.00 for stuck_sensor, drop, impossible_value.
- **Fix:** add small quantization noise to stuck values (real stuck sensors still jitter at the ADC level) and report per-type results so a reader sees the hard types (spike, drift, oscillation) separately from the easy ones. Headline F1 is inflated by the easy types.

### P1-5. Generator leaks the label into non-target sensors
- **Where:** `generator.py:260-264`. During any anomaly, every non-target sensor gets extra noise that only exists while `mode == 'ANOMALY'`.
- **Why it matters:** the model can partly learn "machine is in anomaly mode" from sensors that are supposedly normal. Small effect, but it is a pure simulation artifact.
- **Fix:** delete those lines, or apply the same extra noise in normal mode.

### P1-6. Results come from one seed and one split
- **Where:** `generator.py` `DEFAULT_FIXED_SEED = 295`, `model.py` single train/val/test.
- **Fix:** run the full pipeline over 5 to 10 generator seeds and report mean and spread of the event-level metrics. Without that, threshold 0.40 vs 0.35 and the ablation differences are within noise and you cannot claim either.

### P1-7. No baseline
- **What's wrong:** there is nothing to show the Random Forest with 228 engineered features beats something simple.
- **Fix:** add at least (a) a per-sensor rolling z-score rule, and (b) an unsupervised model trained on normal data only (IsolationForest). Report them next to the RF. Also be explicit in the README that this is supervised classification on simulated labels; real industrial anomaly data is rarely labeled, which is the main reason people use unsupervised methods.

### P1-8. Threshold grid is coarse and arbitrary
- **Where:** `model.py:32` (0.30 to 0.70 in 0.05 steps).
- **Fix:** sweep the full precision-recall curve from `metrics.precision_recall_curve` on validation and pick by an explicit rule (max F1, or max recall at precision >= X). Better: select the threshold on the event-level metric from P1-2, since alerts are what get used.

### P1-9. `/predict` serves a model that no client can realistically call
- **What's wrong:** it expects 228 engineered features computed from up to 78 steps of history. A caller would have to reimplement `features.py` exactly. That is train/serve skew waiting to happen.
- **Fix:** add an endpoint that takes a window of raw readings (at least 79 steps for one machine), runs `build_features` server-side, and scores the last row. Keep the feature endpoint only if you have a reason.

---

## P2: correctness bugs in storage, API, dashboard, and deployment

### P2-1. `sensor_readings.is_anomaly` is always NULL
- **Where:** `db.py` column `is_anomaly`; the raw CSV column is `any_anomaly`. `load_dataframe_to_table` only inserts matching names, so the label is dropped. `timestamp` is also always NULL because the generator never produces one.
- **Evidence:** 50000 rows, 0 non-null `is_anomaly`, 0 non-null `timestamp`.
- **Fix:** rename the column in one place (DB or generator) and add a loader test asserting non-null. Drop `timestamp` or generate it.

### P2-2. `pipeline_runs` records hardcoded metadata, not what actually ran
- **Where:** `load_to_db.py:31-33` (`fixed_seed=295`, `model_threshold=0.35`, `max_step_gap=3`).
- **Evidence:** the run row says 0.35; the model selected 0.40.
- **Fix:** read the threshold from `artifacts/model_metadata.json`, the seed from the generator output (write it to a small run manifest), and `MAX_STEP_GAP` from `alert_events.py`.

### P2-3. A load is not atomic
- **Where:** `load_to_db.py:129-160`, `load_dataframe_to_table` commits per table; `load_csv_if_exists` skips missing files with a print.
- **What's wrong:** a missing or failed CSV leaves a run with partial data, and the API serves it as a complete run. `with sqlite3.connect(...)` also does not close the connection; it only manages the transaction.
- **Fix:** one transaction for the whole run; fail (and roll back) if any required file is missing. Close the connection explicitly or use `contextlib.closing`.

### P2-4. Missing database turns every endpoint into a 500
- **Where:** `db.py:11`. `sqlite3.connect` silently creates an empty file if the path does not exist, then every query fails with "no such table".
- **Evidence:** `GET /runs` with no DB returns 500.
- **Fix:** open read-only from the API (`sqlite3.connect(f"file:{path}?mode=ro", uri=True)`), and return 503 with a clear message from `/health` and other routes if the DB or tables are missing. `/health` should actually check the DB and model artifacts.

### P2-5. `/predict` reloads the model from disk on every request and leaks internals in errors
- **Where:** `model_serving.py:9` (`joblib.load` inside `predict_feature_row`), `api.py:273-274` (`detail=str(error)`).
- **Evidence:** ~0.24 s per request; error responses expose file paths like `artifacts/model.joblib`.
- **Fix:** load once at startup (FastAPI `lifespan`) and keep it on `app.state`. Log the exception server-side and return a generic message. Store the scikit-learn version in `model_metadata.json` and refuse to load on mismatch, since joblib pickles are not portable across sklearn versions.

### P2-6. Filters accept any string and silently return nothing
- **Where:** `api.py:113` (`severity: str | None`), sensor and anomaly_type filters. `Literal` is imported and unused.
- **Evidence:** `?severity=critical` returns `[]` instead of 422.
- **Fix:** use `Literal["INFO", "WARNING", "CRITICAL"]` and the sensor/anomaly enums. Return a total count (header or wrapper object) with paginated lists.

### P2-7. Row alerts are linked to events by re-matching filters, not by ID
- **Where:** `db_queries.py:69-100` (`get_row_alerts_for_event`).
- **Fix:** write `event_id` onto each row alert in `group_alert_events` and store it in `row_alerts`. Then the lookup is a join on `event_id`, and it stops depending on `anomaly_type` (which P0-1 removes anyway).

### P2-8. Docker images are not self-contained; the hosted deploy config is not in the repo
- **Where:** `.dockerignore` excludes `*.db`, `outputs/`, `artifacts/`; `docker-compose.yml` bind-mounts `.:/app`, which replaces everything the image copied.
- **What's wrong:** locally it works only because of the bind mount. Built and run anywhere else (your Render deploy, per the last three commits), the API has no DB and no model, so every data route 500s and `/predict` 500s. Nothing in the repo describes how the hosted version gets its data.
- **Fix:** pick one: build the data and model inside the image (a build stage that runs the pipeline on the fixed seed), or have the container run the pipeline on startup if the DB is missing. Commit the Render config (`render.yaml`) so the deployment is reproducible. Remove the `.:/app` bind mount from the default compose file (keep it in a `docker-compose.override.yml` for dev). Add a compose `healthcheck` on `/health` and `depends_on: condition: service_healthy`, which also makes the dashboard's polling loop unnecessary locally. Run as a non-root user.

### P2-9. Dashboard polling and defaults
- **Where:** `dashboard.py:12-27` and `:183-188`.
- **What's wrong:** the comment says ~3 minutes; worst case is 36 x (10 s timeout + 5 s sleep) = 9 minutes of a frozen page. The run picker defaults to run 1, so after reloading data users see the oldest run.
- **Fix:** default to `/runs/latest` (or a selectbox populated from `/runs`). Cap total wait by wall clock, show progress text while waiting, and wrap fetches in `st.cache_data` with a short TTL.

### P2-10. `MODEL_THRESHOLD` config is dead and misleading
- **Where:** `config.py:8`, `docker-compose.yml` sets `MODEL_THRESHOLD=0.35`. Nothing reads it; the served threshold comes from `model_metadata.json`.
- **Fix:** delete it, or make it an explicit override that `model_serving` honors and logs.

---

## P3: tests, code quality, dependencies, docs

### Tests
- **T1. `model_serving.py` has zero tests.** `/predict` is only tested with `predict_feature_row` monkeypatched out (`tests/test_api.py:1695-1705`). The README (`README.md:523`) says tests cover "model inference endpoint"; that is not true. Add a test that trains a tiny model into `tmp_path`, loads it, and checks missing/extra feature handling (after P0-7).
- **T2. Feature tests check column names, not values.** `tests/test_features.py` would not catch P0-6 or a wrong rolling window. Add hand-computed expectations on a tiny series (e.g. delta, rolling mean, z-score, slope on a known ramp) and the shuffle-invariance test.
- **T3. No end-to-end smoke test.** Add one test (or CI step) that runs generator (2 machines, ~400 steps) through features, model, alerts, events, `load_to_db`, then hits the API with `TestClient`. This would have caught P0-4 and P2-1.
- **T4. Leakage regression tests.** Assert that alert construction never reads `target_sensor`/`anomaly_type` (after P0-1), and that no split boundary cuts through an anomaly event (after P1-3).
- **T5. `dashboard.py` is untested.** Move the fetch/transform code into functions in a separate module and test them; keep Streamlit calls thin. Today all of the page runs at import time.
- **T6. 1089 warnings.** Fix the `features.py` fragmentation (build feature columns in a dict and `pd.concat(axis=1)` once) and then run pytest with `-W error::FutureWarning` so pandas deprecations fail CI instead of breaking you on the next major version.

### Code quality
- **Q1. Redundant features.** `{sensor}_centered_signal_t` equals `{sensor}_dev` (`features.py:72` vs `:274`); `roll_avg_delta` equals `10_step_diff / 10`; `abs_5_step_diff`/`abs_10_step_diff` are `.abs()` of existing columns; `dir_imbalance_*` is computed and then permanently dropped in `model.py:34`. Remove them. Duplicates split Random Forest importance across copies and make the ablation and importance output misleading.
- **Q2. `slope` via `rolling().apply(np.polyfit)`** (`features.py:14`, `:113-125`) is called ~58k times per 5000 rows. Replace with the closed form (rolling covariance of x and y over rolling variance of x), which is vectorized.
- **Q3. Dead code and leftovers.** `CRITICAL_SCORE_THRESHOLD`, `ALERT_TYPE_THRESHOLD`, `SEVERITY_PRIORITY` in `alerts.py` are unused; `eval_cols` in `alerts.py:32` is unused; `get_alert_events_for_run` and `get_predictions_for_run` in `db_queries.py` are unused by the API; `current_osc_recall`/`voltage_osc_recall` in `model.py:79-80` are debugging leftovers. Replace them with a generic per-type, per-sensor recall table.
- **Q4. `WARNING_SCORE_THRESHOLD = 0.35`** (`alerts.py:157`) is a copy of the old model threshold. With the selected threshold at 0.40 the "low score" template can never fire. Derive score bands from the model threshold in `model_metadata.json`.
- **Q5. Constants and paths are duplicated across modules.** The sensor list is defined in `generator.py`, `features.py`, and `alerts.py`; output paths are hardcoded strings in `generator.py`, `features.py`, `alerts.py`, `alert_events.py`, `evaluate.py`, `load_to_db.py` even though `config.OUTPUT_DIR` exists. Move both into `config.py`.
- **Q6. `inspect_generator.py` is broken.** It imports `matplotlib` (not in requirements), reads `sensor_data_raw.csv` from the repo root (wrong path), and runs at import time. Fix or delete.
- **Q7. Flat layout.** 17 top-level modules. Moving them into a package (`anomaly_detection/`) with a `pyproject.toml` and a CLI entry point would remove the `pythonpath = .` hack in `pytest.ini`. Low priority, but it reads better to anyone reviewing the repo.
- **Q8. Timestamps.** `model.py:284` uses naive `datetime.now()`. Use `datetime.now(timezone.utc)`.
- **Q9. Row-by-row loops.** `alerts.build_alerts` and `alert_events.group_alert_events` use `iterrows`. Fine at 2k alerts, but a vectorized version (boolean masks, `groupby` with gap-based `cumsum`) is shorter and easier to test.

### Dependencies and CI
- **D1. Nothing is pinned** (`requirements.txt`). The model artifact is a pickle tied to the scikit-learn version, and the code relies on pandas groupby/rolling index behavior that has changed between majors. Pin exact versions (or use a lock file via `pip-tools`/`uv`), and record the sklearn version in the artifact metadata (P2-5).
- **D2. Runtime and dev deps are mixed.** `pytest` and `httpx` ship in both Docker images; the API image installs Streamlit and the dashboard image installs scikit-learn. Split into `requirements.txt` (API), `requirements-dashboard.txt`, `requirements-dev.txt`.
- **D3. Version mismatch.** `.python-version`, CI, and Docker use 3.11. CI never tests anything newer. The suite does pass on 3.13 with current pandas 3.0 and scikit-learn 1.9 (I ran it), so adding 3.12 and 3.13 to a CI matrix is cheap.
- **D4. CI only runs pytest.** Add `ruff` (lint + format check), the end-to-end smoke run from T3, and `pip` caching in `setup-python`. `on: [push, pull_request]` runs twice for every PR push from a branch in the same repo; restrict `push` to `main`.

### Docs
- **R1. The README reports no results.** For a portfolio ML project this is the most important missing piece. Add a results table (event-level metrics from P1-2, per-type recall, baseline comparison from P1-7, spread across seeds from P1-6) and a short "limitations" section: simulated data, supervised labels, known label noise.
- **R2. Fix the inaccurate claims:** the purge gap explanation (`README.md:250`), "missing features are filled with 0" as a feature (`README.md:460`), and "tests cover model inference endpoint" (`README.md:523`).
- **R3. Fresh-clone instructions.** After P0-4, verify the README steps on a clean clone. Document the deployed version (URL, how it gets data) or remove references to it.
- **R4. Explain the design choices** that a reviewer will ask about: why a supervised RF, why these thresholds in `SAFETY_THRESHOLDS`, why `MAX_STEP_GAP = 3`, how severity is assigned.

---

## Suggested order

1. P0-4, P0-5, P0-6 (small and mechanical).
2. P0-1 with P0-2 and P0-3 together: per-sensor model, rule checks on every row, severity rework. This is the largest change and touches `model.py`, `alerts.py`, `alert_events.py`, the DB schema, and tests.
3. P1-1 and P1-3 (fix the labels and the split), then P1-2 (event metrics), then rerun and record numbers.
4. P1-6, P1-7 (seeds and baselines), then R1 (write the results into the README).
5. P0-7, P1-9, P2-x (serving, storage, deploy).
6. P3 cleanup alongside whatever file you are already editing.

Reproduce the numbers in this audit: clone, `mkdir outputs`, `python run_pipeline.py`, `python load_to_db.py`, then compare `outputs/alert_events.csv` `real_value` counts and `outputs/predictions.csv` against the figures at the top.
