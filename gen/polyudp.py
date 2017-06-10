import ctypes
import numpy as np
import pygmo

class PolyUDP:
    def __init__(self):
        from numpy.ctypeslib import ndpointer
        lib = ctypes.cdll.LoadLibrary('./lib.so')
        self.xs = int(lib.xs())
        self.f = lib.f
        self.f.restype = ctypes.c_double
        self.f.argtypes = [ndpointer(ctypes.c_double,shape=(self.xs,),flags='C_CONTIGUOUS')]
        self.jac = lib.jac
        self.jac.argtypes = [ndpointer(ctypes.c_double,shape=(self.xs,),flags='C_CONTIGUOUS'),
                ndpointer(ctypes.c_double,shape=(self.xs,),flags='C_CONTIGUOUS')]
        self.hess = lib.hess
        self.hess.argtypes = [ndpointer(ctypes.c_double,shape=(self.xs,),flags='C_CONTIGUOUS'),
                ndpointer(ctypes.c_double,shape=(self.xs*(self.xs+1)/2,),flags='C_CONTIGUOUS')]

    def fitness(self,x):
        return self.f(x)

    def get_bounds(self,):
        return (-2*np.ones(self.xs),2*np.ones(self.xs))

    def gradient(self,x):
        gr = np.zeros(self.xs)
        self.jac(x,gr)
        return gr

    def hessians(self,x):
        he = np.zeros(self.xs * (self.xs+1)/2)
        self.hess(x,he)
        return he
