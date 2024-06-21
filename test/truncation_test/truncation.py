import os
from ASI_embedding.embedding import ProjectionEmbedding
from ase.calculators.aims import Aims
from ase.build import molecule
from ase.visualize import view
from ase.data.s22 import s22, s26, create_s22_system
import numpy as np
import csv

os.environ['ASI_LIB_PATH'] = "/home/gabram/Software/FHIaims/builds/mkl_oneapi_embedding/libaims.240501.scalapack.mpi.so"

calc_ll = Aims(xc='PBE',
    relativistic='atomic_zora scalar',
    occupation_type="gaussian 0.01",
    mixer="pulay",
    n_max_pulay=10,
    KS_method="parallel",
    RI_method="LVL",
    charge_mix_param=0.5,
    sc_accuracy_rho=1E-05,
    sc_accuracy_eev=1E-03,
    sc_accuracy_etot=1E-06,
#    sc_accuracy_forces=1E-04,
    sc_iter_limit=100,
    collect_eigenvectors=True,
#    compute_forces=True,
    postprocess_anyway = True,
    density_update_method='density_matrix', # for DM export
    output=["eigenvectors", "hamiltonian_matrix","matrices_elsi","json_log"],
    compensate_multipole_errors=True,
    atomic_solver_xc="PBE",
    compute_kinetic=True,
    energy_density="harris_foulkes",
    )

calc_hl = Aims(xc='SCAN',
    relativistic='atomic_zora scalar',
    occupation_type="gaussian 0.01",
    mixer="pulay",
    KS_method="parallel",
    RI_method="LVL",
    n_max_pulay=10,
    charge_mix_param=0.5,
    sc_accuracy_rho=1E-05,
    sc_accuracy_eev=1E-03,
    sc_accuracy_etot=1E-06,
#    sc_accuracy_forces=1E-04,
    sc_iter_limit=100,
    collect_eigenvectors=True,
#    compute_forces=True,
    postprocess_anyway = True,
    density_update_method='density_matrix', # for DM export
    output=["eigenvectors", "hamiltonian_matrix","matrices_elsi","json_log"],
    compensate_multipole_errors=True,
    atomic_solver_xc="PBE",
    compute_kinetic=True,
    energy_density="harris_foulkes",
  )

def run_binding_energy_test(hl_calc=calc_hl, ll_calc=calc_ll, hl_xc="pbe", ll_xc="pbe"):

    import time

    s22_atom = s26[22]

    dimer_list = []
    data_dict_list = []

    hl_calc.parameters["xc"] = hl_xc
    ll_calc.parameters["xc"] = ll_xc

    for dist_offset in np.arange(-0.5,1.5,0.11):
        H2O = create_s22_system(s22_atom)
        H_bond_vec = H2O.positions[11] - H2O.positions[1]
        H_bond_vec_norm = H_bond_vec/np.linalg.norm(H_bond_vec)

        H2O.positions[6:] = H2O.positions[6:] + ( dist_offset * H_bond_vec_norm )

        dimer_list.append(H2O)

        dist = np.linalg.norm(H_bond_vec) + dist_offset

        Projection = ProjectionEmbedding(H2O, embed_mask=[2,1,2,2,2,1,2,1,2,2,2,1],calc_base_ll=ll_calc, calc_base_hl=hl_calc, frag_charge=-2, mu_val=1.e+6)
        #, truncate_basis=True)
        start = time.time()
        Projection.run()
        end = time.time()
        
        data_dict = {}

        data_dict["LL XC"] = calc_ll.parameters["xc"]
        data_dict["HL XC"] = calc_hl.parameters["xc"]
        data_dict["OH-H Distance"] = dist
        data_dict["AB LL Energy"] = Projection.AB_LL_PP.ev_corr_total_energy
        data_dict["A HL Energy"] = Projection.A_HL_PP.ev_corr_total_energy
        data_dict["A LL Energy"] = Projection.A_LL.ev_corr_total_energy
        #data_dict["AB LL Energy"] = Projection.AB_LL_PP.total_energy
        #data_dict["A HL Energy"] = Projection.A_HL_PP.total_energy
        #data_dict["A LL Energy"] = Projection.A_LL.total_energy
        data_dict["PB Correction"] = Projection.PB_corr
        data_dict["Total DFT Energy (A-in-B)"] = Projection.DFT_AinB_total_energy
        data_dict["Time"] = end - start

        data_dict_list.append(data_dict)

    return data_dict_list

def run_single_fragment(hl_calc=calc_hl, ll_calc=calc_ll, hl_xc="pbe",ll_xc="pbe"):

    import time

    s22_atom = s26[22]

    dimer_list = []
    data_dict_list = []

    hl_calc.parameters["xc"] = hl_xc
    ll_calc.parameters["xc"] = ll_xc

    H2O = create_s22_system(s22_atom)

    dist = 0.

    dimer_list.append(H2O)

    Projection = ProjectionEmbedding(H2O[:6], embed_mask=[2,1,2,2,2,1],
                            calc_base_ll=ll_calc, calc_base_hl=hl_calc, frag_charge=-1, mu_val=1.e+6)
                            #, truncate_basis=True)
    start = time.time()
    Projection.run()
    end = time.time()

    data_dict = {}
    data_dict["LL XC"] = calc_ll.parameters["xc"]
    data_dict["HL XC"] = calc_hl.parameters["xc"]
    data_dict["OH-H Distance"] = dist
    data_dict["AB LL Energy"] = Projection.AB_LL_PP.ev_corr_total_energy
    data_dict["A HL Energy"] = Projection.A_HL_PP.ev_corr_total_energy
    data_dict["A LL Energy"] = Projection.A_LL.ev_corr_total_energy
    #data_dict["AB LL Energy"] = Projection.AB_LL_PP.total_energy
    #data_dict["A HL Energy"] = Projection.A_HL_PP.total_energy
    #data_dict["A LL Energy"] = Projection.A_LL.total_energy
    data_dict["PB Correction"] = Projection.PB_corr
    data_dict["Total DFT Energy (A-in-B)"] = Projection.DFT_AinB_total_energy
    data_dict_list.append(data_dict)
    data_dict["Time"] = end - start


    #Projection.garbage_collect()

    return data_dict_list

data_dict_list = []
data_dict_list += run_binding_energy_test(hl_calc=calc_hl, ll_calc=calc_ll,hl_xc="PBE0", ll_xc="PBE")
data_dict_list += run_single_fragment(hl_calc=calc_hl, ll_calc=calc_ll, hl_xc="PBE0", ll_xc="PBE")

field_names = ["LL XC", "HL XC", "OH-H Distance", "AB LL Energy", "A HL Energy", "A LL Energy", "PB Correction", "Total DFT Energy (A-in-B)", "Time"]

with open('Values.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = field_names) 
    writer.writeheader() 
    writer.writerows(data_dict_list) 
