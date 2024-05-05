from ase import Atoms

class AtomsEmbedding(Atoms):

    # Should feed in an existing ASE calculator
    # Objects not wrapped properly inside each other. Hi ho.
    def __init__(self,atoms=None,calc=None,exports=[],imports=[],
                  cell=None, pbc=None, info=None, celldisp=None): # Dummy arguments

        import copy
        import os
        from scalapack4py import ScaLAPACK4py
        from mpi4py import MPI
        from asi4py.asecalc import ASI_ASE_calculator
        from ctypes import cast, py_object, CDLL, RTLD_GLOBAL

        if atoms is None:
            # GAB: This might seem strange. But if AtomsEmbedding.__class__ is invoked,
            # the incorrect routines for initialising an Atoms object are called.
            # For anything other than initialisation, this routine should allow the
            # Atoms object to initialise and the rest of the ASE code to go on its
            # merry way. :)

            Atoms.__init__(self,cell=cell,pbc=pbc,info=info,celldisp=celldisp)
            return

        else:
            # For now, let's just feed an atoms object in and use the logic of
            # Atoms.copy to generate the relevant attributes

            Atoms.__init__(self, cell=atoms.cell, pbc=atoms.pbc, info=atoms.info,
                            celldisp=atoms._celldisp)

            self.arrays = {}
            for name, a in atoms.arrays.items():
                self.arrays[name] = a.copy()
            self.constraints = copy.deepcopy(self.constraints)

        if self.symbols is None:
            print (f"ASI Embedding: AtomsEmbedding object empty. Read in geometry. \nAborting.")
            MPI.COMM_WORLD.Abort(1)

        self.registers_cb = {}
        self.registers_init = {}
        self.callbacks = {}

        # Global LIB path needs to be set elsewhere
        # self call by itself looks wrong
        # Creates AtomsEmbedding.calc and AtomsEmbedding.calc.asi
        # Modify output to separate inner and outer regions for improved user-side troubleshooting

        #TODO: Find better way of passing environmental variables.
        ASI_LIB_PATH = os.environ['ASI_LIB_PATH']
        asilib = CDLL(ASI_LIB_PATH, mode=RTLD_GLOBAL)
        self.sl = ScaLAPACK4py(asilib)

        self._initcalc = calc
        self.calc = ASI_ASE_calculator(ASI_LIB_PATH, self.init_ase_calculator, MPI.COMM_WORLD, atoms, work_dir="./", logfile="aims.out")

        self.check_exports_and_imports(exports,imports)
        self.generate_callbacks_export(exports)
        self.generate_callbacks_import(imports)

    def check_exports_and_imports(self,exports,imports):

        from .CallbackGenerators import CallbackGenerator

        # Checks the proposed imports and exports actually exist
        # Callbacks and inits likely have to be hardcoded by virtue of the
        # way Fortran deals with callbacks
        # While generating callbacks, prevents any invalid callbacks/init
        # statements being called externally.

        # Pseudocode
        # Get names of all register functions
        # Whittle down to existing arrays
        # Check proposed exports and imports against list
        # Crash if export and import options don't exist


        # Method for nabbing names of register functions
        register_cb_funcs = [func for func in dir(self.calc.asi) if "register" in func and "callback" in func]
        register_cb_names = [func.split('_')[1] for func in dir(self.calc.asi) if "register" in func and "callback" in func]

        print(register_cb_names, register_cb_funcs)

        register_init_funcs = [func for func in dir(self.calc.asi) if "register" in func and "init" in func]
        register_init_names = [func.split('_')[1] for func in dir(self.calc.asi) if "register" in func and "init" in func]

        print(register_init_names, register_init_funcs)

        # Checks whether requested callback for init has an existing register
        for exprt in exports:
            if exprt not in register_cb_names:
                print(f"Method {exprt} is not available for export.")
                # Kill calc

        for imprt in imports:
            if imprt not in register_init_names:
                print(f"Method {imprt} is not available for import.")
                # Kill calc

        for name, func in zip(register_cb_names, register_cb_funcs):
            self.registers_cb[name] = self.calc.asi.__getattribute__(func)
            self.callbacks[name] = CallbackGenerator(self.sl)

        for name, func in zip(register_init_names, register_init_funcs):
            self.registers_init[name] = self.calc.asi.__getattribute__(func)
            self.callbacks[name] = CallbackGenerator(self.sl)

    # TODO: Make method to invoke/deactivate registers at will.
    # TODO: Also need method to indicate which registers are currently active.
    def generate_callbacks_export(self, exports):

        # Actually assigns callback functions for a given AtomsEmbedd object

        print("Initialising export registers")
        for exprt in exports:
            print(f"Setting register for {exprt} callback...")
            self.registers_cb[exprt](self.callbacks[exprt].export_callback, self.calc.asi)

    def generate_callbacks_import(self, imports):

        # Actually assigns callback functions for a given AtomsEmbedd object

        print("Initialising import registers")
        for imprt in imports:
            print(f"Setting register for {imprt} initialisation...")
            self.registers_init[imprt](self.callbacks[imprt].import_callback, self.calc.asi)

    def init_ase_calculator(self, asi):
        # Information needs to come from the highest level to here somehow.
        # Tempted to shunt this elsewhere - very deep in the code atm.
        # Also needs to be manipulated by embedding routines to ensure
        # correct information is passed downwards,
        # Hardcode for testing.
        self.calc = self._initcalc
        self.calc.write_input(self)

    def copy(self):

        """Return a copy of Atoms Embed."""
        import copy

        print(f"Copying object...")
        atomsembed = self.__class__(cell=self.cell, pbc=self.pbc, info=self.info,
                               celldisp=self._celldisp.copy())

        atomsembed.arrays = {}
        for name, a in self.arrays.items():
            atomsembed.arrays[name] = a.copy()
        atomsembed.constraints = copy.deepcopy(self.constraints)

        atomsembed.calc = self.calc

        atomsembed.registers_cb = self.registers_cb.copy()
        atomsembed.registers_init = self.registers_init.copy()
        atomsembed.callbacks = self.callbacks.copy()

        return atomsembed
