# Commit map after the history rewrite

On 2026-10-24 the history was rewritten once to remove a work email address from
`docs/SETUP_REPORT.md` in old commits (no code, data or result changed; the final tree is
identical). Every commit hash changed, including the GitHub-signed initial commit, whose
signature the rewrite dropped. Results files written before the rewrite record old hashes;
this table maps them to the new ones. Generated from the two histories, paired in order and
checked to have the same author, author date and subject.

| Old | New | Subject | Recorded in results/ |
|---|---|---|---|
| `204f8f0` | `7e2b40d` | Initial commit |  |
| `f525844` | `084252c` | chore: scaffold |  |
| `48107e1` | `500c2b5` | chore: split CPU/GPU torch handling, pin chemprop 2.3.1 |  |
| `94b2d71` | `abcacf3` | chore: exclude rdkit-pypi, pin rdkit to chemprop-compatible version |  |
| `421e51c` | `a5679f5` | feat: scaffold-split construction, split audit and leakage tests (T2, T3) |  |
| `09c6200` | `8286f21` | docs: allow committing splits/ in CLAUDE.md |  |
| `0a6ec51` | `a8d801a` | docs: record sources and licence status of split data |  |
| `cf36105` | `cd289d0` | feat: Chemprop MVE model, ensemble and MC dropout training/prediction (T4) |  |
| `a2bac57` | `348055d` | feat: resumable run grid and train_all.py with smoke mode (T5) |  |
| `08fa85a` | `a12dd60` | feat: Kaggle notebook template and kaggle_ctl.py for CLI-driven runs (T6) |  |
| `33f03f0` | `7b4ffe4` | fix: locate extracted code dir in Kaggle notebook |  |
| `69a3e3f` | `4ba079c` | fix: PyTDC to optional data group, pip --no-deps on Kaggle, keep torchmetrics in export |  |
| `1545726` | `1bf1ad0` | fix: probe torch in a fresh process in the Kaggle notebook |  |
| `382498d` | `e674b00` | fix: keep smoke outputs out of cache/preds and results/; record Kaggle smoke run |  |
| `a0d3f8a` | `077406b` | feat: pilot/full modes with baked-in task and seed lists for the Kaggle notebook | yes |
| `cfe3417` | `84df48d` | results: Kaggle pilot compute logs (3 tasks, seeds 1-2, 60 epochs, 2xT4) | yes |
| `6a03809` | `dc7d1f2` | results: Kaggle full-grid compute logs (3 tasks x 5 seeds x 6 trainings, 2xT4) |  |
| `b0bfe92` | `1edd9f4` | results: pack full-grid predictions as results/preds.zip (45 parquet files, 4.5 MB) |  |
| `4771341` | `9e916e1` | feat: split, normalized and Mondrian conformal with leakage tests (T7) |  |
| `ec44d16` | `51cd59d` | feat: add Mondrian conformal with normalized score |  |
| `78b9b82` | `58c6ad8` | feat: test-set metrics (T8): accuracy, NLL, calibration, selective RMSE, conformal coverage by distance |  |
| `ec2e53c` | `568e245` | results: T8 metric tables for the full grid; document metric definitions |  |
| `551116d` | `fa0f9d3` | feat: seed aggregation, scaffold-cluster bootstrap CIs and paired comparisons (T9) |  |
| `8f69138` | `b3aae26` | results: T9 statistics tables; metrics regenerated at the T9 commit |  |
| `17a3008` | `ac4f001` | feat: CIs that also resample calibration groups and refit conformal quantiles (DR-005) |  |
| `54249f1` | `d85f9ba` | results: statistics with calibration-resampled CIs (DR-005) |  |
| `dec9dc5` | `00c843e` | feat: auto-generated report and figures (T10) |  |
| `fb9a383` | `b7d7532` | results: summary.md and figures generated at the T10 commit |  |
| `7a39284` | `64ab33b` | docs: README with question, setup, prior work, reproduction, results with CIs and limitations (T12) |  |
| `749983c` | `aae2806` | docs: eval protocol as executed, data licences, Chemprop v2 citation, final setup state |  |
| `83cc33b` | `dc18275` | docs: README workflow diagram and plain-language guide to reading the results |  |
| `6b09a91` | `1440ef9` | docs: remove work email address from setup report |  |
| `6c67e3f` | `c9eced1` | docs: qualify Caco-2 miscalibration claim, state width costs and the Mondrian caveat in the README |  |
| `8c3c483` | `e486516` | feat: post-hoc checks: size-matched valid control, Winkler score, median error per split (DR-006) |  |
| `e0c4ccd` | `ce1863c` | results: post-hoc checks (DR-006) and Winkler score; tables regenerated at the DR-006 commit |  |
| `639c04c` | `28dceda` | docs: post-hoc checks (DR-006) in the README: TL;DR, size-matched control, Winkler score, median errors, weighted conformal as untested remedy |  |
| `ac75394` | `9059043` | docs: AGENTS.md with the project rules for Codex |  |
| `e49298f` | `ee56a3f` | results: summary figures for calibration source and far-quartile coverage |  |
| `4432b8a` | `3f3b1eb` | feat: post-hoc calibration-set composition checks: largest scaffold groups in the split audit, naive_cal without large test-absent groups, valid - naive_cal by seed group (DR-007) | yes |
| `a9bcbb4` | `0d7863c` | results: split audit with largest scaffold groups; composition-controlled calibration and seed-group comparisons (DR-007), regenerated at the DR-007 commit |  |
| `2d31149` | `4a44d5c` | docs: calibration-set composition (DR-007) in the README; post-hoc items 4-5 in the eval protocol; Murcko grouping limitation |  |
| `cdb03bd` | `bd88291` | results: remove the two hand-made summary figures; every figure in results/ is built by make evaluate |  |
| `4685be1` | `76cd4a4` | docs: label the CI type on every TL;DR line |  |
