====================
Installation
====================

Requirements
____________________

* A shared-object library for a supported QM driver (`FHI-aims <https://fhi-aims.org>`_)
* `asi4py >= 1.3.0 <https://pvst.gitlab.io/asi/>`_
* `ASE <= 2.22.1 <https://wiki.fysik.dtu.dk/ase/>`_

Embasi Installation
__________________________


Installation from PyPI
~~~~~~~~~~~~~~~~~~~~~~

.. highlight:: bash

EmbASI may be installed from the PyPI respository, which will automatically download the source code, install the required packages, and install EmbASI in the target used for pip installation::

  $ pip install --upgrade embasi

Installation from Source
~~~~~~~~~~~~~~~~~~~~~~~~

.. highlight:: bash

The source code for EmbASI may be downloaded from the github repository on a local machine, and pip installation executed in the top directory of the EmbASI package::

  $ git clone git@github.com:logsdail/EmbASI.git
  $ cd embasi
  $ pip install .


QM Driver Installation
______________________

FHI-aims
~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   FHI-aims is a software package licensed by `MS1P <https://ms1p.org/index.php>`_ for academic and commercial settings. If you do not hold a FHI-aims license, please refer to the `licensing page <https://fhi-aims.org/get-the-code-menu/licensing-models>`_ for further information and instructions on acquiring a licence.

.. highlight:: bash

Download the development version of FHI-aims from the supported Gitlab repository::

  $ git clone git@aims-git.rz-berlin.mpg.de:aims/FHIaims.git

Instructions for compiling FHI-aims may be found in either the `FHI-aims manual <https://fhi-aims.org/uploads/documents/FHI-aims.221103_1.pdf>`_, with additional architecture/compiler specific instructions at `aims-club <https://fhi-aims-club.gitlab.io/tutorials/Basics-of-Running-FHI-aims/preparations/>`_. No modifications to the installation of FHI-aims are required except for a CMake directive to install a shared object library as opposed to a standard executible (which is a requirement of `ASI <https://pvst.gitlab.io/asi/>`_ in general). This is achieved by adding the following line to the initial_cmake.cache file::

  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
