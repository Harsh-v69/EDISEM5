# Knowledge graph schema

Single source of truth in code: `ayurveda_kg/kg.py` (`SCHEMA`). Tables are stored as parquet in `data/processed/kg/` (`nodes_<Type>.parquet`, `edges_<type>.parquet`) and loaded into a NetworkX `MultiDiGraph` with `kg.load_tables` + `kg.build_kg`. Coverage numbers are in the generated `docs/kg_report.md`.

## Node types

| Type | ID format | Key attributes | Source |
|---|---|---|---|
| Herb | `herb:<IMPPAT name>` | name, aliases (other names, e.g. *Tinospora cordifolia* -> *Tinospora sinensis*), common | `config/scope.yaml` |
| Compound | `cpd:<InChIKey>` (fallback `cpd:<IMPPAT id>` if no InChIKey) | name, inchikey, smiles, pubchem_cid, chembl_id, phy_ids (IMPPAT ids merged into it), n_merged, synonyms | IMPPAT detail pages |
| Target | `gene:<HGNC symbol>` | symbol, is_cyp | IMPPAT, DGIdb, ChEMBL, TDC (all use gene symbols) |
| Drug | `drug:<scope name>` | name, cls (drug class), chembl_id (parent molecule), aliases | `config/scope.yaml` + ChEMBL |

## Edge types

| Edge | From -> To | Attributes | Source | Meaning |
|---|---|---|---|---|
| `contains` | Herb -> Compound | parts, n_refs | IMPPAT plant pages | The herb is reported to contain the compound |
| `modulates` | Compound -> Target | sources | IMPPAT "human targets" (ChEMBL, STITCH, etc. as listed) | Compound is linked to the protein (experimental or predicted, see `sources`) |
| `predicted_cyp_inhibitor` | Compound -> Target(CYP) | source | SwissADME predictions shown on IMPPAT | **Prediction**, not a measurement |
| `predicted_pgp_substrate` | Compound -> Target(ABCB1) | source | SwissADME via IMPPAT | **Prediction** |
| `targets` | Drug -> Target | sources, interaction_types, score | DGIdb | Drug acts on the gene product |
| `substrate_of` | Drug -> Target(enzyme) | source | ChEMBL metabolism table + TDC Carbon-Mangels (CYP2C9/2D6/3A4) | Drug is metabolised by the enzyme |
| `non_substrate_of` | Drug -> Target(enzyme) | source | TDC Carbon-Mangels (label 0) | Reported **not** to be a substrate (real negatives) |
| `ddi` | Drug - Drug (stored once, unordered) | level (worst of Major/Moderate/Minor/Unknown), source | DDInter 2.0 | Known drug-drug interaction between scoped drugs |

## Label-leakage note (important for Phase 3)

The HDI silver label ("herb compound inhibits CYP X AND drug is a substrate of X") is computed from `predicted_cyp_inhibitor`, `modulates` (CYP targets), and `substrate_of`. These are **label-source edges**: a GNN trained on the same graph could rediscover the labelling rule instead of learning anything new. In Phase 3/4 these edges (and any path that reconstructs the label) must be masked from the training graph, and train/test splits must be by **herb**, not by pair. A test will assert the masking.

## Known limitations

- Predicted edges (SwissADME CYP inhibition, P-gp substrate, many IMPPAT targets) are in-silico predictions. Edge attributes keep the source so models and the paper can separate them from experimental evidence.
- Drugs with no `substrate_of` edge are mostly renally cleared (metformin, atenolol, digoxin, furosemide, etc.), which is expected. Genuine gaps: rivaroxaban, apixaban (CYP3A4 substrates) and theophylline (CYP1A2): not in ChEMBL metabolism or TDC. Fill from cited literature when building gold labels (Phase 3), not silently.
- The KG contains scoped herbs/drugs only (`config/scope.yaml`); it is not all of IMPPAT/DrugBank by design.
- IMPPAT licence is CC BY-NC-ND 4.0: derived IMPPAT tables must not be redistributed. The paper releases code and identifiers only.
