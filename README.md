# TrafficFlowBench 2026 — team Eyad Qasim

Code that reproduces our selected final submission to the IEEE Big Data Cup 2026 TrafficFlowBench competition, and our report.

| | |
|---|---|
| Team | Eyad Qasim (one member: Eyad Qasim Raheem, independent) |
| Kaggle | `eyadqasim` |
| Selected submission | `sub_e50.csv`, submitted 2026-11-06 02:29 UTC |
| Scores | public 0.88236, private **0.87520** (rank 88 of 801) |
| Report | `report/report.pdf` |

## Run it

```bash
python run_all.py --data /path/to/competition/input --assets /path/to/trafficflowbench-086517-public-assets --out ./run
```

* `--data` is any folder that contains the competition's `kaggle_public` directory (it is searched recursively).
* `--assets` is the public Kaggle dataset `anubisyy/trafficflowbench-086517-public-assets` (see External data below).
* On Kaggle, attach the competition data and that dataset and run `python run_all.py` with no flags: both are found under `/kaggle/input`, and the output goes to `/kaggle/working/run`.

`./run.sh` does the same with a dependency check first (`./run.sh help` lists its options). The result is `run/s07_submission/submission.csv`. If a copy of our submitted `sub_e50.csv` is anywhere under the inputs, stage 07 also compares the two cell by cell and prints the differences.

**Hardware and runtime.** One NVIDIA T4 (16 GB), 4 CPU cores, 30 GB RAM, as in a Kaggle GPU notebook: about 7 hours in total, of which about 4.5 hours is the four Task 1 U-Net seeds on the GPU. No internet access is needed.

**Environment** (Kaggle Python image, November 2026): Python 3.13, numpy 2.1.3, pandas 2.3.3, scipy 1.16.3, pyarrow 23.0.1, LightGBM 4.6.0, CatBoost 1.2.10, PyTorch 2.11 (CUDA build).

## Stages

`run_all.py` runs the stages in this order; each writes to its own folder under `--out` and reads the folders written before it.

| Stage | Script | Task | What it does | Output |
|---|---|---|---|---|
| 01 | `stages/s01_prep.py` | 2 | dense time x link arrays per corridor and split, published window tables | `prep/` |
| 06 | `stages/s06_odme.py` | 4 | exact non-negative ridge solve per corridor and split (lambda 0.05) | `odme_rows.csv` |
| 04 | `stages/s04_onset.py` | 2 | onset models trained on pseudo-months; March pseudo-month check; online-rule check | `onset_probs.csv` |
| 05 | `stages/s05_ongoing.py` | 2 | ongoing models trained on pseudo-months; March pseudo-month check; online-rule check | `ongoing_probs.csv` |
| 02 | `stages/s02_state_unet.py` | 1, 3 | state U-Net, run once per seed: 2026, 7, 11, 23 | `unet3_preds.parquet` per seed |
| 03 | `stages/s03_state_merge.py` | 1, 3 | mean of the seeds and the density step | `state_rows.csv` |
| 07 | `stages/s07_submission.py` | all | joins everything in `submission_key.csv` order with the decoding rules of the selected file | `submission.csv` |

`python run_all.py --only s04_onset` runs one stage; `--from-stage s02_state_unet` resumes.

**Task 2 on a new month.** The Task 2 stages read each split's first day and number of days from the timestamps in the data, so a newly generated month in the same file layout needs no code change. `python run_all.py --task2-only` runs only stages 01, 04, 05 and 07 (about 2 hours on 4 CPU cores, no GPU); its `submission.csv` has every row of `submission_key.csv`, with the queue predictions filled in and zeros in the state and ODME columns.

**Random seeds.** Task 1 U-Nets: 2026, 7, 11, 23 (`U_SEED`, set by `run_all.py`). Onset: LightGBM 20261004 and bag seeds 11, 22, 33, CatBoost 20261005, numpy generator 5. Ongoing: LightGBM 20261105.

## Verification

We reran this package from scratch on Kaggle, with only the competition data, the public assets dataset and a copy of `sub_e50.csv` (for the comparison) attached.

* **Task 2 only** (`--task2-only`, CPU notebook: 4 cores, 30 GB RAM, no GPU): 118 minutes (stage 01 13 min, 04 35 min, 05 68 min, 07 1 min). The split dates read from the data were train 2030-06-01 (273 days), validation 2031-03-01 (31 days) and private 2031-04-01 (30 days). Both online-rule checks passed on all 80 windows: the largest difference was 2.4e-7 in the onset features and 0 in the ongoing features and probabilities. The queue predictions are identical to those of `sub_e50.csv`: the same 273 onset cells and the same 10,119 ongoing cells are set to 1, and no others.

## The online rule (Task 2)

For a window with origin T only data stamped at or before T is used: the 60-minute history, the released row at T, and month-to-date evidence computed from prefix sums that read only earlier rows of the same split (plus the origins of earlier windows from `window_index.csv`). All Task 2 models are fitted on the train split only; thresholds and blend weights were chosen on the train split and the March (public) leaderboard. Stages 04 and 05 repeat the organizers' check on every scored window: every row after the origin is set to missing in all arrays (speed, flow, occupancy, ramps, eligibility), the window's features are recomputed, and the stage stops if any value differs by more than floating-point rounding (relative tolerance 1e-5). The results, with the largest difference per window, are written to `online_rule_check_onset.csv` and `online_rule_check_ongoing.csv`. Set `--no-online-check` to skip it. Tasks 1, 3 and 4 are offline and use whole months, as the rules allow.

## External data and code

No external data is used.

* Public dataset `anubisyy/trafficflowbench-086517-public-assets` (author anubisyy, Apache License 2.0): the onset checkpoints `onset_lgb.txt` and `onset.cbm` (their probability is blended into the first onset window of each corridor and month) and `ongoing_lgb.txt` (used only as a comparison in the stage 05 check).
* Feature code of the same public pipeline, as written out by its companion notebook `anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517` (folder `tfb_source/`), copied unchanged into `third_party/public_pipeline/` (`config.py`, `queue_features.py`, `queue_ongoing.py`, `base_panel.py`; Apache License 2.0, see the `NOTICE` there).
* Libraries listed under Environment, all open source.

Our own code is released under the Apache License 2.0 (`LICENSE`).
