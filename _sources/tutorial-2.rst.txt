.. _tut2:

**Tutorial: Advanced Usage of EmbASI for QM/QM Embedding**
==========================================================

**Authors:** Gabriel Bramley **Date:** 2025-11-20

--------------

What Does This Tutorial Cover?
------------------------------

Following on from Part 1 (:ref:`tut1`), the following tutorial provides
advanced usage directives for running calculations with post-HF methods
and keywords for reducing the cost enabled by basis truncation.

--------------

1. Post-HF Methods
------------------

`Part 1 <XXX>`__ of the tutorial showed how to perform DFT-in-DFT
embedding for a simple test system. However, projection-based embedding
can also be used to perform Wavefunction-in-DFT (WF-in-DFT) embedding.
The post-HF methods in FHI-aims (including MP2 and RPA) may be used for
a high-level of theory by supplying an argument to ``post_scf`` for the
``ProjectionEmbedding`` object

.. code:: python

   Projection = ProjectionEmbedding(methanol_dimer,
                                    embed_mask=[2,1,2,2,2,1,2,1,2,2,2,1],
                                    post_scf="RPA",
                                    calc_base_ll=calc_ll,
                                    calc_base_hl=calc_hl,
                                    projection="huzinaga-sc",
                                    localisation="SPADE",
                                    run_dir="dimer",
                                    parallel=True,
                                    gc=True,)

Usage of post-HF methods in EmbASI has the same provisos as a standard
FHI-aims calculation (See: `GW, BSE, and RPA
tutorial <https://fhi-aims-club.gitlab.io/tutorials/rpa-and-gw-for-molecules-and-solids/>`__).
For example, one can change the type of resolution of identity used in
the beyond-DFT calculation by adding the ``RI_method`` keyword in the
high-level calculator object:

.. code:: python

   calc_hl = Aims(xc='PBE', profile=AimsProfile(command="asi-doesnt-need-command"),
       atomic_solver_xc="PBE",
       override_initial_charge_check=True,
       RI_method="LVL"
     )

As with a standard post-HF calculation with FHI-aims based on the RI
framework, special choices for the basis set must be used to obtain
well-converged energies with respect to basis size. For these
calculations, we recommend using the ``NAO-VCC-XZ`` library of basis
sets, and applying a basis set `extrapolation
strategy <https://fhi-aims-club.gitlab.io/tutorials/rpa-and-gw-for-molecules-and-solids/basisSets/#fn:5>`__
to approach the complete basis set limit. The two-point extrapolation
scheme of `Zhang et
al. <https://iopscience.iop.org/article/10.1088/1367-2630/15/12/123033>`__
is especially useful in this context:

.. math::


   \begin{aligned}
   E(\infty) = \frac{E(4)(4+d)^{\alpha} - E(5)(5+d)^{\alpha} }{(4 + d)^{\alpha} - (5 + d)^{\alpha}},
   \end{aligned}

which provides extrapolation to the CBS energy :math:`E(\infty)` for the
NAO-VCC-4Z and NAO-VCC-5Z basis sets, where :math:`\alpha=4.51` and
:math:`d=2.65`.

--------------

2. Basis Truncation
-------------------

In all the examples presented, the high-level calculation is performed
with the entire supersystem basis. Although the occupied subspace has
been reduced, the basis functions of the environment are still included
in the simulation. This means that summations of densities and
potentials are still performed over these basis centers, and the
corresponding virtual orbitals associated with these basis functions are
included in any post-HF calculation. Failing to truncate these basis
functions therefore limits the time and memory improvements enabled by
QM/QM embedding.

However, basis centers that do not contribute significantly to the
high-level subsystem may be excluded by supplying the argument,
``truncate_basis_thresh`` to the ``ProjectionEmbedding`` object:

.. code:: python

   Projection = ProjectionEmbedding(methanol_dimer,
                                    embed_mask=[2,1,2,2,2,1,2,1,2,2,2,1],
                                    post_scf="RPA",
                                    calc_base_ll=calc_ll,
                                    calc_base_hl=calc_hl,
                                    projection="huzinaga-sc",
                                    localisation="SPADE",
                                    run_dir="dimer",
                                    truncate_basis_thresh=0.0001,
                                    parallel=True,
                                    gc=True,)

The threshold stated excludes basis centers in the environment subsystem
that contribute less than the specified charge to the high-level
subsystem. Although basis truncation enables substantial speed-ups for
the high-level subsystem calculation, it can also introduce a
`significant error to the embedding total energy unless properly
parameterised <https://chemrxiv.org/engage/chemrxiv/article-details/6915bc0cef936fb4a2efb408>`__.
We recommend scanning through a set of truncation tresholds
(:math:`10^{-2}-10^{-6} \|e\|`) to determine a value that gives the best
compromise between accuracy and speed-up. Generally, relative values
such as adsorption energies can result in reduced errors from truncation
due to cancellation of errors.

