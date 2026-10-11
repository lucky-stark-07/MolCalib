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

TDC: Huang et al., NeurIPS Datasets and Benchmarks 2021 (arXiv:2102.09548). TDC's MIT licence
covers its code, not the datasets.

## Licence of the data in this repository (checked 2026-10-12)

The MIT licence of this repository covers the code only. The split files here, and the
predictions in `results/preds.zip` (which contain the same SMILES and targets), are derived data
and keep the terms of their source:

- **solubility_aqsoldb**: CC BY 4.0 (as stated by TDC). Attribution: Sorkun et al. 2019 above.
- **lipophilicity_astrazeneca**: from ChEMBL document CHEMBL3301361. ChEMBL data is licensed
  CC BY-SA 3.0 Unported (https://chembl.gitbook.io/chembl-interface-documentation/about), so
  these derived files are shared under CC BY-SA 3.0 with attribution to ChEMBL and AstraZeneca.
  The ChEMBL document page does not state other terms for this deposition.
- **caco2_wang**: no licence found. TDC lists it as "Not Specified"; the data comes from Wang et
  al. 2016 (J. Chem. Inf. Model., DOI 10.1021/acs.jcim.5b00642). Redistribution rights are
  unconfirmed. These files can be regenerated without redistribution with `make splits`
  (deterministic), which downloads the data from TDC.
