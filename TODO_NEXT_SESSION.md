# Remaining Work — LeafletFA Reproducibility

## LeafletFA repo — clean, migration complete

- [x] Add AnnData format tests to `tests/test_basic.py`
- [x] Add AnnData format description to README and docs
- [x] Run quickstart notebook end-to-end — `cell_ontology_class` is the column name
- [x] Pin deps and promote scanpy/umap-learn to core — done in `pyproject.toml`
- [x] Fix COO sparse FutureWarning / cluster count indexing bug
- [x] Migrate package from `src/BetaDirichletFactor` to `leafletfa/`
- [x] Scaffold MkDocs Material docs site

### Remaining MkDocs pages

- [ ] **Fill `outputs/splicing-programs.md`** — what psi means, how to interpret loadings
- [ ] **Fill `outputs/cell-activities.md`** — assign_post / X_PHI, how to plot
- [ ] **Fill `outputs/differential.md`** — link to `differential_splicing` module
- [ ] **Fill `ecosystem.md`** — scanpy / scvi-tools / muon interop
- [ ] **Deploy docs to GitHub Pages** — add `.github/workflows/docs.yml`

### Other

- [ ] **PyPI release** — `python -m build && twine upload`
- [ ] **Docstrings on key public methods** — `from_anndata`, `train`, `get_all_variables` need
  Google-style Args/Returns blocks so mkdocstrings renders them properly

---

## leafletfa-paper repo — not started

> **Purpose:** Frozen reproducibility record. NOT a user-facing tool.

- [ ] **Rename repo on GitHub:** Settings → Rename → `leafletfa-paper`
- [ ] **Add figure index to README** — table mapping every paper figure to its script/notebook
- [ ] **Update analysis script imports** — replace HPC `sys.path` hacks with `from leafletfa import LeafletFA`
