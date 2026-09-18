# Slurm scripts

Cluster: `frontend.fisica.cabib`. Detected setup (checked from this frontend):

- Partitions: `cpu` (default, 2 nodes × 56 cores), `knl_tacc` (144 nodes ×
  272 cores, Knights Landing — much bigger, use it for very large sweeps),
  `gpu` (not useful here, this code is pure CPU/numpy).
- No Slurm account required (`sacctmgr` shows no per-user associations, and
  the partitions allow all groups/accounts), so none of the scripts pass
  `--account`.
- Nodes power on demand (`sinfo` shows idle nodes as `idle~`, i.e. powered
  down). A freshly submitted job can sit in `CONFIGURING` for a few minutes
  while its node boots — that's normal, not a stuck job. `squeue -u $USER`
  will flip to `R` once the node is up.
- Conda env `alm_env` already exists
  (`/home/koltona/miniconda3/envs/alm_env`) with numpy/scipy/matplotlib, but
  **not pandas**. Only `plot_zeta_vs_n.py` needs pandas, and that's a
  plotting script you'd typically run on the frontend after merging results,
  not inside a job. If you do want to run it (or anything else needing
  pandas) via Slurm, run `slurm/setup_env.sh` once first.

## Files

| File | Purpose |
|---|---|
| `common.sh` | Sourced by every sbatch script: `cd`s to the repo root and activates `alm_env`. Edit here if the env name/location ever changes. |
| `setup_env.sh` | Run once from the frontend to make sure `alm_env` has numpy/scipy/matplotlib/pandas. Safe to re-run. |
| `submit_demo.sbatch` | Sanity-check job: runs `alm.py`'s built-in demo. Use this first to confirm the env/partition setup works. |
| `submit_single.sbatch` | Runs one `run_structure_factor.py` call; all args after the script path are forwarded to it. |
| `scan_array.sbatch` | Array-job replacement for `scan_sweep_bc.sh`: runs the full `(bc, n, L)` sweep as parallel array tasks (one combination per task) instead of one long serial loop. Each task writes its own CSV row into `slurm/results/`. |
| `merge_scan_results.sh` | Run after `scan_array.sbatch` finishes: concatenates the per-task CSVs in `slurm/results/` into a single `scan_results.csv` at the repo root, ready for `plot_zeta_vs_n.py`. |
| `logs/` | `.out`/`.err` files land here (`%x_%j` for single jobs, `%x_%A_%a` for array tasks). Gitignored except for `.gitkeep`. |
| `results/` | Per-task CSV rows from `scan_array.sbatch`, merged by `merge_scan_results.sh`. Gitignored except for `.gitkeep`. |

## Usage

```bash
# 0. one-time sanity check
sbatch slurm/submit_demo.sbatch
squeue -u $USER

# 1. a single run, any args you'd normally pass to run_structure_factor.py
sbatch slurm/submit_single.sbatch -L 65536 -n 2.0 -c 1.0 --samples 500 \
    --bc periodic --no-plot --no-show

# 2. full sweep, in parallel instead of scan_sweep_bc.sh's serial loop
sbatch slurm/scan_array.sbatch
squeue -u $USER                     # wait for all 60 array tasks to finish
slurm/merge_scan_results.sh         # -> scan_results.csv at repo root

# 3. back on the frontend (or wherever you have pandas):
python3 plot_zeta_vs_n.py --csv scan_results.csv --Lmin 8192 --no-show
```

`scan_array.sbatch`'s grid (`BCS`, `NS`, `LS`, `NSAMPLES`) mirrors
`scan_sweep_bc.sh` exactly. If you change the sweep, edit the arrays *and*
`#SBATCH --array=0-N` (N = `len(BCS)*len(NS)*len(LS) - 1`) together.

For a much larger sweep (bigger `L`, more samples, finer `n` grid), switch
`--partition=cpu` to `--partition=knl_tacc` in the sbatch scripts — it has
far more nodes available, at the cost of slower individual cores.
