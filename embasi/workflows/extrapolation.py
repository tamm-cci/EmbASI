import os, typing
from embasi.embedding import ProjectionEmbedding

#os.environ["ASI_LIB_PATH"] = "/home/dchen/Software/FHIaims/_build_lib/libaims.250711.scalapack.mpi.so"

class Extrapolation:
    """

    Provides an estimated value based on the assumption that the
    existing trend would continue. Allows us to predict energy at
    the CBS Limit by using results from smaller basis sets, while
    maintaining a high degree of accuracy as well as a more efficient
    method which requires less computing power as well as less time.

    Parameters
    ----------
    file1: str
        Name of file which contains the first basis set
    file2: str
        Name of file which contains the second basis set
    path: str
        Name of directory which contain the basis set
    atom: ASE Atom
        Object of an atom, which contains information about the atom
    embed_mask: list[int] or int
        Assigns either the first in atoms to region 1, or an index of
        int values 1 and 2 to each embedding layer. WARNING: The atoms
        object will be reordered such that embedding layer 1 appear
        first
    calc_ll: ASE FileIOCalculator
        Calculator object for layer 1
    calc_hl: ASE FileIOCalculator
        Calculator object for layer 2
    asi_path: str
        Name of directory where ASI (Atomic Simulation Interface) is installed
    projection1_param: dict
        Additional parameters for the first projection
    projection2_param: dict
        Additional parameters for the second projection
    d: float
        Value used for the formula E(∞)
    alpha: float
        Value used for the formula E(∞)

    """
    def __init__(self, file1, file2, path, atom, embed_mask, calc_ll, calc_hl, asi_path, projection1_param = {} , projection2_param = {} , d=2.85, alpha = 4.49):
        self.asi_path:str = asi_path
        os.environ["ASI_LIB_PATH"] = self.asi_path
        self.file1:str= file1
        self.file2:str = file2
        self.path:str = path
        self.atom:object = atom
        self.embed_mask: typing.List[int] = embed_mask
        self.calc_ll = calc_ll
        self.calc_hl = calc_hl
        self.mu_val: float = 1.e+6
        self.options: typing.List[str] = [file1,file2]
        self.results = []
        self.projection1_param = projection1_param
        self.projection2_param = projection2_param
        self.d = d
        self.alpha = alpha

    def checkInParam(self, item: str, default: str, cycle: int) -> str:
        if cycle == 0:
            app_Dict = self.projection1_param.items()
        else:
            app_Dict = self.projection2_param.items()

        for key, val in app_Dict:
            if key == item:
                return val

        return default

    @property
    def extrapolate(self) -> float:
        try:
            conv1 = int(self.file1)
            conv2 = int(self.file2)

            if conv1 > conv2:
                pass
            else:
                raise ValueError(
                    "File1 should be bigger than File2."
                )
        except:
            raise TypeError(
                "File1: int\nFile2: int"
            )

        for index, (item, projection_parameters) in enumerate(zip(self.options, [self.projection1_param, self.projection2_param])):
            os.environ["AIMS_SPECIES_DIR"] = f"{self.path}{item}Z"

            projection = ProjectionEmbedding(
                self.atom,
                embed_mask=self.embed_mask,
                calc_base_hl=self.calc_hl,
                calc_base_ll=self.calc_ll,
                mu_val=self.mu_val,
                total_energy_corr= self.checkInParam("total_energy_corr","1storder", index),
                localisation = self.checkInParam("localisation","SPADE", index),
                projection = self.checkInParam("projection","level-shift", index)
            )

            for key, val in projection_parameters.items():
                setattr(projection, key, val)

            projection.run()

            energy = projection.DFT_AinB_total_energy

            self.results.append(energy)

        return (self.results[0]*(int(self.file1)+self.d)**self.alpha - self.results[1]*(int(self.file2)+self.d)**self.alpha)/((int(self.file1) + self.d)**self.alpha-(int(self.file2)+self.d)**self.alpha)
