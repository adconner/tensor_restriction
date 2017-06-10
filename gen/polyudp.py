import ctypes
import numpy as np
import pygmo
from numpy.ctypeslib import ndpointer

class PolyUDP:
    def __init__(self):
        lib = ctypes.cdll.LoadLibrary('./lib.so')
        self.xs = int(lib.xs())

    def fitness(self,x):
        f = ctypes.cdll.LoadLibrary('./lib.so').f
        f.restype = ctypes.c_double
        f.argtypes = [ndpointer(ctypes.c_double,shape=(self.xs,),flags='C_CONTIGUOUS')]
        return (f(x),)

    def gradient(self,x):
        jac = ctypes.cdll.LoadLibrary('./lib.so').jac
        jac.argtypes = [ndpointer(ctypes.c_double,shape=(self.xs,),flags='C_CONTIGUOUS'),
                ndpointer(ctypes.c_double,shape=(self.xs,),flags='C_CONTIGUOUS')]
        gr = np.zeros(self.xs)
        jac(x,gr)
        return gr

    def hessians(self,x):
        hess = ctypes.cdll.LoadLibrary('./lib.so').hess
        hess.argtypes = [ndpointer(ctypes.c_double,shape=(self.xs,),flags='C_CONTIGUOUS'),
                ndpointer(ctypes.c_double,shape=(self.xs*(self.xs+1)/2,),flags='C_CONTIGUOUS')]
        he = np.zeros(self.xs * (self.xs+1)/2)
        hess(x,he)
        return he

    def get_bounds(self,):
        return (-2*np.ones(self.xs),2*np.ones(self.xs))
