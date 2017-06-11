import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import ctypes
import numpy as np
import pygmo as pg
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
                ndpointer(ctypes.c_double,shape=(self.xs*(self.xs+1)//2,),flags='C_CONTIGUOUS')]
        he = np.zeros(self.xs * (self.xs+1)//2)
        hess(x,he)
        return (he,)

    def get_bounds(self,):
        return (-2*np.ones(self.xs),2*np.ones(self.xs))

    def has_gradient(self):
        try:
            ctypes.cdll.LoadLibrary('./lib.so').jac
            return True
        except:
            return False

    def has_hessians(self):
        try:
            ctypes.cdll.LoadLibrary('./lib.so').hess
            return True
        except:
            return False

class DoAll:
    def __init__(self,uda):
        self.alg=pg.algorithm(uda)
    def evolve(self,pop):
        res = pg.population(pop.problem)
        for i,p in enumerate(zip(pop.get_x(),pop.get_f())):
            tpop = pg.population(pop.problem)
            tpop.push_back(*p)
            tpop=self.alg.evolve(tpop)
            res.push_back(tpop.champion_x,tpop.champion_f)
        return res

class Comp:
    def __init__(self,uda1,uda2):
        self.alg1=pg.algorithm(uda1)
        self.alg2=pg.algorithm(uda2)
    def evolve(self,pop):
        res = self.alg2.evolve(pop)
        res = self.alg1.evolve(res)
        return res
