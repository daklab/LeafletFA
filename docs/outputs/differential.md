# Differential Splicing

LeafletFA includes a `differential_splicing` module for testing whether splicing programs are differentially active between groups of cells, and for identifying junctions that drive those differences.

## Testing differential SP activity

```python
from leafletfa import differential_splicing

# Compare SP activities between two groups
results = differential_splicing.test_differential_activity(
    adata,
    groupby="cell_ontology_class",
    group1="T cell",
    group2="B cell",
)
```

## API reference

See the full API: [leafletfa.differential_splicing](../api/differential_splicing.md)
