## LeafletFA: Bayesian Factor Analysis for Single-Cell Splicing

LeafletFA is a scalable probabilistic Beta-Dirichlet factor model designed to decompose sparse single-cell splicing variation into interpretable, continuous Splicing Programs (SPs). Unlike traditional methods, LeafletFA discovers coordinated modules of splicing events (Alternative Transcript Structure Events - ATSEs) that reflect biological states, such as cellular aging or lineage specification, without requiring pre-defined cell type labels.

### Key Features
- Scalable Inference: Powered by Pyro and Stochastic Variational Inference (SVI) for atlas-scale datasets (200,000+ cells).
- Sparsity Robust: Specifically designed to handle the high dropout and sparse coverage inherent in single-cell splicing data.
- Biologically Interpretable: Learns a "splicing dictionary" where each factor represents a coordinated regulatory program.

### Compatibility
LeafletFA is optimized for full-length transcript sequencing (e.g., Smart-Seq2) which provides the internal junction coverage necessary for alternative splicing analysis.

### Ongoing Development
- [x] Implement Beta-Dirichlet factor model (LeafletFA).
- [x] Support for cross-species transfer learning.
- [x] GPU-accelerated mini-batch training.
- [ ] Comprehensive ReadTheDocs documentation.
