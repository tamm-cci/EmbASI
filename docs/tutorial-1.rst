.. _tut1:

**Tutorial: Basic Usage of EmbASI for QM/QM Embedding**
=======================================================

**Authors:** Gabriel Bramley **Date:** 2025-11-20

--------------

What Does This Tutorial Cover?
------------------------------

This tutorial covers the basic installation and usage instructions for
performing a QM/QM embedding simulation with EmbASI to calculate the
dissociation energy with EmbASI.

For more information regarding Projection-based embedding, we recommend
the following papers: 1. `Manby et
al. <https://pubs.acs.org/doi/10.1021/ct300544e>`__ 2. `Lee et
al. <https://pubs.acs.org/doi/10.1021/acs.accounts.8b00672>`__

For specific information regarding the implementation of EmbASI, please
refer to: 1. `Bramley et
al. <https://chemrxiv.org/engage/chemrxiv/article-details/6915bc0cef936fb4a2efb408>`__

--------------

1. Installation Instructions
----------------------------

The latest release version of EmbASI may be installed simply through
pip:

.. code:: shell

   pip install embasi

Alternatively, the development version of EmbASI and requested branches
may be directly installed through the github repository:

.. code:: shell

   git clone https://github.com/tamm-cci/EmbASI
   cd EmbASI
   pip install .

As EmbASI is reliant on the Atomic Simulation Interface (ASI), FHI-aims
must be compiled as a shared object library. To do so, simply add the
following to your CMake script:

.. code:: shell

   set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

--------------

2. Known Limitations
--------------------

Before going into the usage of EmbASI, it is important to know the
present limitations of the software to avoid disappointment. Firstly,
forces and stresses have not been implemented. Secondly, the framework
presently does not (explicitly) support periodic calculations,
particularly for systems with multiple **k**-points. Although early
testing shows that single **k**-point calculations are feasible, we have
neither tested this functionality extensively, nor can we in good faith
present such an implementation as complete.

Development is underway to address both of the above limitations.

--------------

3. Basic Usage Instructions
---------------------------

The following example shows how one can perform a QM/QM embedding
simulation for a methanol dimer with Projection-based embedding. The -OH
moieties are described at the high-level of theory (PBE0) and the rest
of the dimer described at a lower-level of theory (PBE).

First, one must import the relevant modules for executing the EmbASI
workflow and importing the geometry for the structure of interest:

.. code:: python

   from embasi.embedding import ProjectionEmbedding
   from embasi.parallel_utils import root_print
   from ase.data.s22 import s22, s26, create_s22_system
   from ase.calculators.aims import Aims, AimsProfile

Then, one must also define the relevant environmental variables that
specify the location of the shared object library of FHI-aims and the
desired basis set:

.. code:: python

   os.environ['ASI_LIB_PATH'] = <AIMS_ROOT_DIR>/lib.aims.XXX.so
   os.environ['AIMS_SPECIES_DIR'] = <AIMS_ROOT_DIR>/species_defaults/defaults_2020/light

Alternatively, these environmental variables may be set in the bash
shell.

Next, input parameters for the calculation are specified using the
syntax for ASE Calculator objects. A calculator object is defined for
both the high and low levels of theory:

.. code:: python

   calc_ll = Aims(xc='PBE', profile=AimsProfile(command="asi-doesnt-need-command"),
       atomic_solver_xc="PBE",
       override_initial_charge_check=True,
     )

   calc_hl = Aims(xc='PBE0', profile=AimsProfile(command="asi-doesnt-need-command"),
       atomic_solver_xc="PBE",
       override_initial_charge_check=True,
     )

A brief word on the two non-standard key-word options for the atomic
solver and the initial charge checks. Although PBE and PBE0 use the same
underlying atomic solver for defining the single-atom reference
quantities, if one were to use two levels of theory that use different
default atomic solvers (*i.e.*, B3LYP and PBE), one would find
significant mismatch in the integrated charge densities between the high
and low level calculation steps. The initial charge checks are overriden
because subsystems that partition across covalent charge often result in
subsystem calculations with overall charge. As charge information is
provided through an imported density matrix, supplying this keyword
avoids an caused by failing to specify the initial charge configuration
in ``geometry.in``.

Then we set-up the geomtry for the methanol dimer by importing the
structural information from the S26 dataset of ASE:

.. code:: python

   methanol_dimer_idx = s26[22]
   methanol_dimer = create_s22_system(methanol_dimer_idx)

The QM/QM embedding simulation is set-up by instantiating the
``ProjectionEmbedding`` class with the desired input directives:

.. code:: python

   Projection = ProjectionEmbedding(methanol_dimer,
                                    embed_mask=[2,1,2,2,2,1,2,1,2,2,2,1],
                                    calc_base_ll=calc_ll,
                                    calc_base_hl=calc_hl,
                                    projection="huzinaga-sc",
                                    localisation="SPADE",
                                    run_dir="dimer",
                                    parallel=True,
                                    gc=True,)

The above arguments achieve the following:
   1. **methanol_dimer**: Supplies the structural and species information stored in the ASE atoms object.
   2. **embed_mask**: Specifies which atoms are considered at the high-level of theory (1) and the low-level of theory (2).
   3. **calc_base_ll**: The calculator object with input directives for the low-level of theory calculation.
   4. **calc_base_hl**: The calculator object with input directives for the high-level of theory calculation.
   5. **projection**: The projection operator used to orthogonalise the KS-eigenstates of the two, partitioned subsystems. The present option (``huzinaga-sc``) for `self-consistent Huzinaga projection <https://pubs.aip.org/aip/jcp/article/145/6/064107/440939>`__ is known to be a robust option that produces minimal errors. The ``level-shift`` projection operator is also performant, but can sometimes result in anomolously high errors.
   6. **localisation**: The algorithm used to localise and partition the charge of the supersystem into the subsystems specified in ``embed_mask``. ``SPADE`` provides a parameter-free localisation routine that runs natively in the EmbASI wrapper. Alternatively, ``qmcode`` uses an iterative Pipek-Mizey localisation scheme implemented in FHI-aims, with individual KS states assigned to the subsystem assigned to the high-level of theory based on their Hirshfeld population (>0.1 \|e\|). We recommend ``SPADE`` as parallelism is fully supported.
   7. **run_dir**: Specifies the directory used to store intermediate calculations during the embedding calculation.
   8. **parallel**: Specifies whether matrices in the EmbASI wrapper should be stored using BLACS distributed arrays and linear algebra routines performed with SCALAPACK. FHI-aims must be compiled with BLACS and SCALAPACK to support this option. **This keyword does not reflect on whether the QM driver itself runs in serial or parallel.**
   9. **gc**: An extra keyword that enables the deallocation of arrays and data objects that are no longer needed during the running of EmbASI. This reduces headline memory requirements.

Once the ``ProjectionEmbedding`` object has been created, the embedding
calculation is ran through the directive:

.. code:: python

   Projection.run()

The final embedded energy can then be stored in a desired variable:

.. code:: python

   meoh_dimer_pbe0inpbe_energy = Projection.DFT_AinB_total_energy

A similar step can be used to calculate the embedded total energy for
the metanol monomer and evaluated the dimer dissociation energy:

.. code:: python

   Projection = ProjectionEmbedding(methanol_dimer[:6],
                                    embed_mask=[2,1,2,2,2,1],
                                    calc_base_ll=calc_ll,
                                    calc_base_hl=calc_hl,
                                    projection="huzinaga-sc",
                                    localisation="SPADE",
                                    run_dir="dimer",
                                    parallel=True,
                                    gc=True,)

   Projection.run()
   meoh_monomer_pbe0inpbe_energy = Projection.DFT_AinB_total_energy

   dimer_dissociation_energy = meoh_dimer_pbe0inpbe_energy - (2 * meoh_monomer_pbe0inpbe_energy)

And we’re done! For advanced usage, please move onto `Part 2 <XXX>`__

