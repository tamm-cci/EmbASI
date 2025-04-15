from maze.zeolite import Zeolite
from maze.io_zeolite import save_zeolites, read_zeolites
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

bea_zeolite = Zeolite.make('BEA')
plot_atoms(bea_zeolite)

site = 185
cluster_indices = bea_zeolite.cluster_maker.get_cluster_indices(bea_zeolite, site)
print(cluster_indices)

cluster, od = bea_zeolite.get_cluster(185)
capped_cluster = cluster.cap_atoms()
cap_idx = capped_cluster.index_mapper.get_index(capped_cluster.parent_zeotype.name, capped_cluster.name, 185)

capped_cluster[cap_idx].symbol = "Sn"
h_cap_name = capped_cluster.additions['h_caps'][0]
plot_atoms(capped_cluster)

save_zeolites("/home/gabrielbramley/Software/EmbASI/examples/FHIaims/BEA_embed", [capped_cluster, capped_cluster.parent_zeotype], zip=False)
