"""Single entry point: reproduce the selected TrafficFlowBench submission (team Eyad Qasim, file sub_e50.csv) from the released data.

    python run_all.py --data /path/to/competition/input --assets /path/to/trafficflowbench-086517-public-assets --out ./run

--data must contain the competition's kaggle_public directory (anywhere below it); --assets is the public dataset
anubisyy/trafficflowbench-086517-public-assets (onset/ongoing checkpoints of the 0.86517 public pipeline, Apache-2.0).
On Kaggle both are found under /kaggle/input and the flags can be left out. Stages run in order; each writes to its own
folder under --out and reads the earlier folders, the data and the assets. The final file is <out>/s07_submission/submission.csv.
Runtime on a Kaggle T4 machine (4 CPU, 1 GPU, 30 GB RAM): about 6 h (5.9 h measured), of which 3.9 h is the four Task 1 U-Net seeds on the GPU.
"""
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["s01_prep", "s06_odme", "s04_onset", "s05_ongoing", "s02_state_unet", "s03_state_merge", "s07_submission"]   # run order: CPU stages first, then the GPU seeds
SEEDS = [2026, 7, 11, 23]          # the four U-Net seeds of the submission


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/kaggle/input", help="folder that contains kaggle_public (searched recursively)")
    ap.add_argument("--assets", default="", help="folder of the public assets dataset (default: searched under --data)")
    ap.add_argument("--out", default="/kaggle/working/run", help="output folder")
    ap.add_argument("--from-stage", default="s01_prep", choices=STAGES, help="skip the stages before this one (their folders must exist)")
    ap.add_argument("--only", default="", choices=[""] + STAGES, help="run a single stage")
    ap.add_argument("--no-online-check", action="store_true", help="skip the rows-after-T check inside stages 04 and 05")
    ap.add_argument("--task2-only", action="store_true", help="Task 2 only (stages 01, 04, 05, 07), e.g. for a newly generated month in the same layout: "
                    "submission.csv then carries the queue predictions and zeros in the other tasks' columns")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    roots = [a.data] + ([a.assets] if a.assets else []) + [os.path.join(HERE, "third_party")]
    todo = [a.only] if a.only else STAGES[STAGES.index(a.from_stage):]
    if a.task2_only:
        todo = [s for s in todo if s in ("s01_prep", "s04_onset", "s05_ongoing", "s07_submission")]
    for st in todo:
        runs = [(f"{st}_s{sd}", {"U_SEED": str(sd)}) for sd in SEEDS] if st == "s02_state_unet" else [(st, {})]
        for name, extra in runs:
            out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)
            prior = [os.path.join(a.out, d) for d in sorted(os.listdir(a.out)) if d < name and os.path.isdir(os.path.join(a.out, d))]
            env = dict(os.environ, TFB_INPUT=os.pathsep.join(roots + prior), TFB_OUT=out, TFB_ONLINE_CHECK="0" if a.no_online_check else "1",
                       TFB_TASK2_ONLY="1" if a.task2_only else "0", PYTHONUNBUFFERED="1", **extra)
            t0 = time.time(); print(f"=== {name} -> {out}", flush=True)
            with open(os.path.join(out, "log.txt"), "w") as lg:
                p = subprocess.Popen([sys.executable, os.path.join(HERE, "stages", st + ".py")], env=env, cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in p.stdout:
                    lg.write(line); print(f"[{name}] {line}", end="", flush=True)
                rc = p.wait()
            print(f"=== {name} finished in {(time.time() - t0) / 60:.1f} min, exit {rc}", flush=True)
            if rc != 0:
                sys.exit(f"stage {name} failed (exit {rc}); see {out}/log.txt")
    print("submission:", os.path.join(a.out, "s07_submission", "submission.csv"))


if __name__ == "__main__":
    main()
