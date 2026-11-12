# VOMCP

VOMCP (POMCP with virtual observations) is an online POMDP planner that decides when it is worth spending a step on gathering information. It is a modification of POMCP (Silver & Veness, 2010) for problems where the agent has an explicit `observe` action: one that does not change the state but returns an additional observation.

Rather than putting `observe` in the search tree like any other action, VOMCP samples a few *virtual observations* from its current belief, plans separately from each of the beliefs they would lead to, and takes `observe` only if a statistical test says that doing so is significantly better than the best ordinary action.

This repository contains the planner, several baselines, and the two test domains used to compare them: a partially observable gridworld and Laser Tag.

## How it works

At each decision step VOMCP does the following:

1. **Sample virtual observations.** Draw `n_vobs` states from the current particle belief and generate the observation that `observe` would return in each. Identical observations are merged and their multiplicities recorded.
2. **Build a belief for each one.** For each distinct virtual observation, update the current belief by rejection sampling, as though `observe` had been taken and had returned that observation. Each of these beliefs becomes the root of a *virtual tree*.
3. **Split the search budget.** The rollout budget is divided equally between the main tree (rooted at the current belief) and the virtual trees. Optionally, a fixed fraction can be given to the virtual trees instead (`--split`).
4. **Search every tree with POMCP.** All trees search over the ordinary actions only; `observe` never appears inside a tree.
5. **Value `observe`.** Its value is the multiplicity-weighted average, over virtual trees, of the discounted value of the best action in each tree (plus the observation cost, if the domain has one).
6. **Test.** A one-sided Welch's t-test compares that value with the value of the best ordinary action in the main tree. The agent takes `observe` only if it is significantly better at the chosen threshold (`--pval`, default 0.20); otherwise it takes the best ordinary action.

After acting, the search tree is reused as in POMCP. If the agent observed and the real observation matches one of the virtual observations, the corresponding virtual tree becomes the new root.

## Planners included

| Name | What it does |
| --- | --- |
| `VOMCP` | The method described above. |
| `VOMCP_avg` | VOMCP trees, but `observe` is valued by averaging over all actions in the virtual trees, not just the best one. Gridworld only. |
| `VOMCP_unequal` | VOMCP with an unequal budget split between main and virtual trees; requires `--split`. Gridworld only. |
| `POMCP` | Standard POMCP over the ordinary actions; never takes `observe`. |
| `EA` | Extra action: `observe` is an ordinary action at every node of a single POMCP tree. In the gridworld, the root decision uses the same t-test. |
| `EAF` | Extra action first: `observe` is available at the root only; deeper nodes and rollouts use ordinary actions. |
| `EAF_modified` | Same tree as `EAF`, but `observe` is valued as the visit-weighted average of the best action's value in each observation branch beneath it. Called `eafm` in Laser Tag. |

## Repository layout

```
gridworld/
  main.py                      run experiments
  PO_gridworld.py              the environment and its maps
  planners/                    base_planner, pomcp, vomcp, ea, eaf, node
  main_complexity_analysis.py  tree-depth statistics along saved histories
  data_analysis/               scripts for collating runs and plotting
laser_tag/
  main.py                      run one experiment
  src/envs/                    Laser Tag environment
  src/reasoning/               pomcp, vomcp, ea, eaf, eafm
  process_file.py, plot_bar_graph.py   result aggregation and plotting
```

The two domains are independent implementations and do not share code.

## Requirements

- Python 3.9 or later
- Gridworld: `numpy` (version 1.x; the code uses `np.float_`, which NumPy 2.0 removed) and `scipy`. The analysis scripts also need `matplotlib`.
- Laser Tag: the packages in `laser_tag/requirements.txt`.

```bash
pip install "numpy<2" scipy matplotlib
pip install -r laser_tag/requirements.txt   # only needed for Laser Tag
```

## Gridworld

A FrozenLake-style grid. The agent starts in the top-left corner and must reach the goal while avoiding holes; reaching either ends the episode. There are four moves plus `observe`.

- A move fails with probability `p_fail`, leaving the agent where it is.
- Every step returns a noisy reading of the agent's position: each coordinate is correct with probability `p_obs` and otherwise falls uniformly within `obs_width` cells of the truth.
- `observe` keeps the agent in place and returns a fresh reading. It has no reward cost, but it uses a time step, so later rewards are discounted further.
- By default the goal gives +1000 and a hole gives -1. With `--survival_rew`, the goal gives +1,000,000 and each step onto a non-terminal cell gives +1.

Run from inside `gridworld/`:

```bash
cd gridworld
python main.py --mode VOMCP --env 6x6 --nruns 10 --seed 0 --save_folder results
```

The planners are pure Python and the default budget is 20,000 rollouts per step, so full-size runs are slow. For a quick check, use something like `--nruns 1 --n_rollouts 1000`.

Main options (`python main.py --help` lists all of them):

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode` | `VOMCP` | `VOMCP`, `VOMCP_avg`, `VOMCP_unequal`, `POMCP`, `EA`, `EAF`, `EAF_modified` |
| `--env` | `6x6` | `6x6`, `6x6_easy`, `6x6_hard`, `6x6_hard2`, `shortcut` |
| `--nruns` | 10 | Number of episodes |
| `--seed` | 0 | Random seed |
| `--n_rollouts` | 20000 | Total rollout budget per planning step, shared across all trees |
| `--n_samples` | `n_rollouts` | Rejection-sampling budget per belief update (or the particle count, with `--fixed_filter`) |
| `--n_vobs` | 4 | Number of virtual observations to sample (VOMCP variants) |
| `--pval` | 0.20 | Significance threshold for choosing `observe` |
| `--split` | none | Fraction of the budget given to the virtual trees; default is an equal share per tree |
| `--c` | 1000 | UCT exploration constant |
| `--gamma` | 0.95 | Discount factor |
| `--p_obs`, `--p_fail`, `--obs_width` | 0.75, 0.3, 1 | Environment noise settings |
| `--max_depth`, `--max_steps` | 200, 150 | Search depth limit and episode length cap |

**Output.** Each run writes to `--save_folder`:

- `<name>.npy`: seven arrays saved in sequence. The first two are indexed by grid cell, the rest by episode: a count of observations received, state visit counts, success flags, episode lengths, total rewards, discounted returns, and the number of `observe` actions taken. Read them with seven successive `np.load` calls on one open file.
- `<name>.pkl`: the action-observation history of each episode, plus search-tree statistics for the EA-family planners.
- `run_parameters/run_parameters_<seed>.pkl`: the settings used. Despite the extension, this is a plain text file.

The default `<name>` encodes the planner and its settings. Partial results are kept in `<name>_0.npy` and `<name>_0.pkl` while a run is in progress.

`data_analysis/collate_runs.py` merges the per-seed files; the other scripts in that folder produce the plots.

## Laser Tag

The Laser Tag benchmark from DESPOT (Somani et al., 2013): a robot on an 11 x 7 grid with obstacles must find and tag a moving opponent, sensing its surroundings with up to eight range-finding lasers. The actions are four moves, `Tag` and `Observe`. Moves and `Observe` each cost 1; a tag gives +10 if it succeeds and -10 if it misses. `--n_lasers` sets how many of the eight lasers return a reading on each step; the active ones are re-drawn at random every step.

The environment framework is adapted from AdLeap-MAS, and this part of the repository is under GPL-3.0 (`laser_tag/LICENSE`).

Run from inside `laser_tag/`:

```bash
cd laser_tag
python main.py --method vomcp --n_vo 4 --n_lasers 8 --n_rollouts 10000 --seed 0
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--method` | `vomcp` | `vomcp`, `pomcp`, `ea`, `eaf`, `eafm` |
| `--n_vo` | 1 | Number of virtual observations to sample (VOMCP) |
| `--n_lasers` | 8 | Number of active lasers per step |
| `--n_rollouts` | `n_samples` | Rollout budget per planning step |
| `--n_samples` | 10000 | Rejection-sampling budget per belief update (or the particle count, with `--fixed_filter`) |
| `--pval` | 0.20 | Significance threshold for choosing `observe` |
| `--seed` | -1 | Random seed; negative leaves the generators unseeded. Also used as the run number in the output filename. |
| `--render` | off | Show the pygame display |

Each run lasts up to 200 steps and writes one semicolon-separated CSV row per step: step number, reward, whether the agent observed, planning time, and rollout and simulation counts.

**Before running:** the output directory is hard-coded near the top of the main loop in `laser_tag/main.py` (the `path = ...` line). Change it to a location on your machine.

`process_file.py` aggregates the CSV files and `plot_bar_graph.py` plots the results; both contain paths that need adjusting.

## References

- D. Silver and J. Veness. Monte-Carlo Planning in Large POMDPs. *Advances in Neural Information Processing Systems 23*, 2010.
- A. Somani, N. Ye, D. Hsu and W. S. Lee. DESPOT: Online POMDP Planning with Regularization. *Advances in Neural Information Processing Systems 26*, 2013.

## License

This repository is distributed under the GNU General Public License v3.0. See `LICENSE`.
