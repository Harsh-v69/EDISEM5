# Data access audit

Probed 2026-09-30T20:49:39+00:00. OK means the URL answered, not that bulk download or licensing is settled. See the manual notes section for what that means.

| Source | URL | Needed for | Status | HTTP | Content-Type | Note |
|---|---|---|---|---|---|---|
| IMPPAT 2.0 | https://cb.imsc.res.in/imppat/ | herb, compound, use, formulation | OK | 200 | text/html; charset=UTF-8 |  |
| PubChem PUG REST | https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/curcumin/property/InChIKey/JSON | compound identifiers | OK | 200 | application/json |  |
| ChEMBL API | https://www.ebi.ac.uk/chembl/api/data/status.json | compound-target bioactivity | OK | 200 | application/json |  |
| DGIdb downloads | https://dgidb.org/downloads | drug-gene interactions | OK | 200 | text/html |  |
| DDInter 2.0 | https://ddinter2.scbdd.com/ | drug-drug interactions, drug info | OK | 200 | text/html; charset=utf-8 |  |
| DrugBank academic | https://go.drugbank.com/academic_research/ | optional: drug-target, substrates | HTTP 403 | 403 | text/html; charset=UTF-8 |  |
| AyurParam (HF search) | https://huggingface.co/api/models?search=ayurparam | local Ayurveda LLM | OK | 200 | application/json; charset=utf-8 |  |

## Manual notes (what the probes do not tell you; written after Phase 1)

| Source | Bulk download? | Licence | What we did |
|---|---|---|---|
| IMPPAT 2.0 | Advertised batch files return **404**; per-page access works | **CC BY-NC-ND 4.0** (no derivatives, do not redistribute derived tables) | Polite crawl (1 req/s, cached) of the 20 scoped herbs: plant, compound detail and human-targets pages. Release code + IDs only; ask IMSc for permission/bulk files before publishing |
| PubChem | REST API | Open | Not needed yet (IMPPAT detail pages already give CID/InChIKey) |
| ChEMBL | REST API / full dump | CC BY-SA 3.0 | Molecule search for 35 drugs + the whole metabolism table (2,147 rows) |
| DGIdb | Yes (`dgidb.org/data/latest/*.tsv`) | Open | interactions, drugs, genes TSVs |
| DDInter 2.0 | Yes (8 CSVs by ATC class) | Public; cite | All 8 files |
| DrugBank | **No** (HTTP 403; academic licence application needed) | Academic licence | Skipped; optional plug-in if the project owner obtains a licence |
| AyurParam (HuggingFace) | Model search reachable | See model card | Checked only; used in Phase 6 |
| TDC CYP substrates (Harvard Dataverse) | Yes | Cite TDC + Carbon-Mangels & Hutter 2011 | CYP2C9/2D6/3A4 substrate labels, added because ChEMBL metabolism alone covered only ~14 of 35 drugs |
| FDA CYP substrate table | Page not machine-readable from a script | Public | Not used |
