# splits/

Derived splits of three TDC ADMET benchmark tasks, one file per task and seed
(`{task}_seed{seed}.csv`; columns `smiles, y, split, scaffold_id, nn_dist`).
Built by `scripts/make_splits.py` from the TDC `admet_group` benchmark; this repo adds the
`naive_cal` hold-out, the scaffold ids and the nearest-neighbour distances.

## Sources and licences (as stated on https://tdcommons.ai/single_pred_tasks/adme/, checked 2026-10-11)

| Task | Original source | Licence stated by TDC |
|---|---|---|
| caco2_wang | Wang et al., J. Chem. Inf. Model. 56(4), 763-773 (2016), https://pubmed.ncbi.nlm.nih.gov/27018227/ | "Not Specified" |
| lipophilicity_astrazeneca | AstraZeneca, ChEMBL deposition, https://doi.org/10.6019/chembl3301361; via MoleculeNet (Wu et al., Chem. Sci. 9, 513-530, 2018) | "Not Specified" |
| solubility_aqsoldb | Sorkun, Khetan & Er, Sci. Data 6, 143 (2019), https://doi.org/10.1038/s41597-019-0151-1 | CC BY 4.0 |

TODO: the licences for caco2_wang and lipophilicity_astrazeneca are not confirmed from the original sources.
TDC: Huang et al., NeurIPS Datasets and Benchmarks 2021 (arXiv:2102.09548).
