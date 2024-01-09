"""
Methods for generating generic callback functions with their own data
"""

#TODO: MAKE SURE IMPORT STATEMENTS LINE UP

class CallbackGenerator():
  
  # GAB: Generates commonly needed callbacks for a given array
  # and provides an array to dump the relevant property.

  def  __init__(self):

    self.data_array=None
    # Get shape of data
    self.shape=None

  def export_callback(self, aux, iK, iS, descr, data):

    from ctypes import cast, py_object, CDLL, RTLD_GLOBAL
    import numpy as np
    
    print(f"Callback invoked!")
    asi = cast(aux, py_object).value

    try:
        if descr:
            descr = sl.wrap_blacs_desc(descr)
            if descr.is_distributed:
                #parprint("distributed case not implemented")
                return # TODO distributed case not implemented
            else:
                pass
        else:
            pass
        # single process case:
        #print (f"dm_calc invoked {asi.scf_cnt}")
        data = np.ctypeslib.as_array(data, shape=(asi.n_basis,asi.n_basis))
        self.data_array = data
        print(f"Callback Invoked!")
    except Exception as eee:
        print ("Something happened in dm_calc", eee)

  def import_callback(self, aux, iK, iS, descr, data):
    
    asi = cast(aux, py_object).value
    try:
        is_distributed=False
        if descr:
            descr = sl.wrap_blacs_desc(descr)
            if descr.is_distributed:
                is_distributed = True
                is_root = (descr.myrow==0 and descr.mycol==0)
    
        locshape = (descr.locrow, descr.loccol) if is_distributed else (asi.n_basis,asi.n_basis)

        data = np.ctypeslib.as_array(data, shape=locshape).T
    
        #if not is_distributed or is_root: TODO
        if is_distributed and not is_root:
          self.data_array = None

        if is_distributed:
            sl.scatter(self.data_array, descr, data)
        else:
            data[:, :] = self.data_array

        #parprint ("dm predict done")
    except Exception as eee:
        print ("EmbeddingASI: Something happened in dm_init", eee)


    # Setter functions for data_array go here

    # Getter functions for data_array go here

    # Data manipulation functions go here.
