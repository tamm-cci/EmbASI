# EmbASI

EmbASI is a python-based wrapper for QM/QM embedding simulations using the ASI (Atomic Simulation Interface). The wrapper handles communication between QM code library calls through C-based callbacks, and imports/exports relevant matices such as density matrices and hamiltonians required for embedding schemes such as Projection-Based Embedding (PbE). Atomic information is communicated through Atomic Simulation Environment (ASE) atoms objects.

[Full EmbASI documentation](https://tamm-cci.github.io/EmbASI/index.html)

# Supported QM Pakage(s)

- FHI-aims

# Requires

- Python >=3.8
- ASE >=2.4.0
- asi4py >=2.4.1
- Shared-object library for relevant QM driver

# Extending the EmbASI to new Codebases

Extending the EmbASI workflow to your own codebase requires some modification of your codebase. These modification control the export and import of data structures, as supported by the ASI API.

Instructions for implementing the ASI API and EmbASI specific directives are provided at:
[Documentation and template files for implementing the ASI API](https://gitlab.com/pvst/asi/-/tree/main/src/dev_templates/fortran?ref_type=heads)
[Documentation for implementing EmbASI](https://tamm-cci.github.io/EmbASI/dev_guide.html)

