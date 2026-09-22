# Slurm scripts

Cluster: `frontend.fisica.cabib`. Detected setup (checked from this frontend,
**except the `gpu`-partition / `cupy_env` details below, which have NOT been
verified on the actual cluster** — see the warning in
`scan_array_free.sbatch`):

- Partitions: `cpu` (default, 2 nodes × 56 cores), `knl_tacc` (144 nodes ×
  272 cores, Knights Landing — much bigger, use it for very large sweeps),
  `gpu`. Originally noted as "not useful here" back when the whole codebase
  was CPU-only numpy; that's no longer true for `bc=free` specifically (see
  `gpu_free.py` and `run_structure_factor.py --gpu`) — `bc=periodic` still
  has no GPU path (its per-sample `brentq` root search hasn't been ported)
  and stays on `cpu`/`knl_tacc`.
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
- `cupy_env` (needed for `--gpu` / `scan_array_free.sbatch`) only exists on
  a workstation so far, **not confirmed to exist on the cluster**. Before
  first use, on a `gpu`-partition node: check the CUDA driver version with
  `nvidia-smi`, then `conda create -n cupy_env python=3.10 && conda activate
  cupy_env && pip install cupy-cudaXXx` (XX matching that driver, e.g.
  `cupy-cuda12x`), mirroring what `slurm/setup_env.sh` does for `alm_env`.
  Also confirm the actual GRES name with `scontrol show partition gpu` —
  `scan_array_free.sbatch` currently assumes `--gres=gpu:1`.

## Files

| File | Purpose |
|---|---|
| `common.sh` | Sourced by every sbatch script: `cd`s to the repo root and activates `$ALM_CONDA_ENV` (default `alm_env`; GPU scripts set this to `cupy_env`). Edit here if an env name/location ever changes. |
| `setup_env.sh` | Run once from the frontend to make sure `alm_env` has numpy/scipy/matplotlib/pandas. Safe to re-run. |
| `submit_demo.sbatch` | Sanity-check job: runs `alm.py`'s built-in demo. Use this first to confirm the env/partition setup works. |
| `submit_single.sbatch` | Runs one `run_structure_factor.py` call; all args after the script path are forwarded to it. CPU-only as written (`--partition=cpu`, `alm_env`) — pass `--bc free --gpu` only if you also adapt its partition/gres and `ALM_CONDA_ENV` the way `scan_array_free.sbatch` does. |
| `scan_array_periodic.sbatch` | Array-job replacement for the `bc=periodic` half of `scan_sweep_bc.sh`: one `(n, L)` combination per task, on `cpu`. Each task writes its own CSV row into `slurm/results/`. |
| `scan_array_free.sbatch` | Same, for the `bc=free` half, on `gpu` (`--gres=gpu:1`, `cupy_env`, `run_structure_factor.py --gpu`). **Unverified on the cluster — read the warning at the top of the file before first submission.** |
| `merge_scan_results.sh` | Run after both array jobs finish: concatenates the per-task CSVs in `slurm/results/` into a single `scan_results.csv` at the repo root, ready for `plot_zeta_vs_n.py`. |
| `logs/` | `.out`/`.err` files land here (`%x_%j` for single jobs, `%x_%A_%a` for array tasks). Gitignored except for `.gitkeep`. |
| `results/` | Per-task CSV rows from the array jobs, merged by `merge_scan_results.sh`. Gitignored except for `.gitkeep`. |

## Usage

```bash
# 0. one-time sanity check
sbatch slurm/submit_demo.sbatch
squeue -u $USER

# 1. a single run, any args you'd normally pass to run_structure_factor.py
sbatch slurm/submit_single.sbatch -L 65536 -n 2.0 -c 1.0 --samples 500 \
    --bc periodic --no-plot --no-show

# 2. full sweep, in parallel instead of scan_sweep_bc.sh's serial loop --
#    periodic (CPU) and free (GPU) are separate jobs since they need
#    different partitions/envs
sbatch slurm/scan_array_periodic.sbatch
sbatch slurm/scan_array_free.sbatch
squeue -u $USER                     # wait for all tasks in both jobs to finish
slurm/merge_scan_results.sh         # -> scan_results.csv at repo root

# 3. back on the frontend (or wherever you have pandas):
python3 plot_zeta_vs_n.py --csv scan_results.csv --Lmin 8192 --no-show
```

Each script's grid (`NS`, `LS`, `NSAMPLES`) mirrors `scan_sweep_bc.sh`
exactly. If you change the sweep, edit the arrays *and* `#SBATCH
--array=0-N` (N = `len(NS)*len(LS) - 1`) together, in **both**
`scan_array_periodic.sbatch` and `scan_array_free.sbatch`, since they must
stay in sync (they cover the same `(n, L)` grid, just different `bc`).

For a much larger `bc=periodic` sweep (bigger `L`, more samples, finer `n`
grid), switch `--partition=cpu` to `--partition=knl_tacc` in
`scan_array_periodic.sbatch`/`submit_single.sbatch` — it has far more nodes
available, at the cost of slower individual cores. (`knl_tacc` has no GPUs,
so this doesn't apply to the `bc=free` job.)
