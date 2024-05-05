import sys, os
from glob import glob
import numpy as np
from mpi4py import MPI
from ase.build import molecule
from ase.io import read
from ase.calculators.aims import Aims
from asi4py.asecalc import ASI_ASE_calculator
from asi_embedding.asi_embedding.atoms_embedding import AtomsEmbedding


libaims_path = os.environ["LIBAIMS_PATH"]
print(libaims_path)
lib_names = glob(libaims_path + "/libaims.*so")
print(lib_names)
assert len(lib_names) == 1, lib_names # only one lib should be there
ASI_LIB_PATH = lib_names[0]
print(ASI_LIB_PATH)

atoms = read('input/geometry.in')
calc = Aims(xc='PBE', 
    occupation_type="gaussian 0.01",
    mixer="pulay",
    n_max_pulay=10,
    charge_mix_param=0.5,
    sc_accuracy_rho=1E-05,
    sc_accuracy_eev=1E-03,
    sc_accuracy_etot=1E-06,
    sc_accuracy_forces=1E-04,
    sc_iter_limit=100,
    output_level="MD_light",
    compute_forces=True,
    postprocess_anyway = True,
    density_update_method='density_matrix', # for DM export
  )
imports = ["DM"]
exports = ["dm","hamiltonian","overlap"]
Embedding = AtomsEmbedding(atoms,calc,imports=imports,exports=exports)
Embedding.calc.keep_overlap=True
Embedding.calc.keep_hamiltonian=True
Embedding.calc.keep_density_matrix=True

Embedding.callbacks["DM"].data_array = {(1,1):np.loadtxt('DM_9.txt').T}

E = Embedding.get_potential_energy()
forces = Embedding.get_forces()
if MPI.COMM_WORLD.Get_rank() == 0:
  print(f'Potential energy: {E:.5f}')
  print(f'Forces:')
  np.savetxt(sys.stdout, forces, fmt="%10.6f")
print(Embedding.callbacks["dm"].data_array)
print(Embedding.callbacks["overlap"].data_array)
print(Embedding.callbacks["hamiltonian"].data_array)
if ((1,1) in Embedding.callbacks["dm"][1,1]):
  print(f'Nel: {np.sum(Embedding.callbacks["dm"][1,1] * Embedding.callbacks["overlap"][1,1])}')
  print(f'HD: {np.sum(Embedding.callbacks["dm"][1,1] * Embedding.callbacks["hamiltonian"][1,1])}')
