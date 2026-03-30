======================================
Developer Guide to Implementing EmbASI
======================================

Introduction
~~~~~~~~~~~~

EmbASI is designed as a minimal workflow for abstracting the tasks associated
with embedding to the Pythonic wrapper. The modification to the host codebase
should be small, but some familiarity with your codebase is desirable to
implement the required control flow mechanisms.

Where possible, we have provided abstract template routines in ``<EmbASI_ROOT>/templates/fortran``.


Atomic Simulation Interface (ASI) API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ASI is a C-based API which manages the transfer of data structures
to and from the QM driver. The vast majority of development work
required to implement EmbASI will involve implementing the C-based
callback infrastructure of ASI. Template routines and an installation
guide are included in the ASI API documentation. Other than stating
where certain matrix dimensions are stored in your codebase, the
callback routines should hopefully work out of the box with the
templates provided.

Implementing the EmbASI Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EmbASI workflow for projection-based embedding involves three
types of calculations:

1. Full self-consistent Kohn-Sham calculations.
2. Non-self-consistent reference energy calculations.
3. Self-consistent embedded calculations.


   
