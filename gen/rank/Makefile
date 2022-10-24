# CFLAGS=-g -fsanitize=address
CFLAGS=-g -O3 -ffast-math -march=native
# CFLAGS=-g
# CFLAGS += -DCX
CXXFLAGS=-std=c++20 -I/usr/include/eigen3 $(shell pkg-config --cflags clp) $(CFLAGS)
LDFLAGS=-llapacke $(shell pkg-config --libs clp) -lceres -lcxsparse -lcholmod -lumfpack -lamd -lcamd -lcolamd -lccolamd -lbtf -lsuitesparseconfig -lmetis -llapack -lopenblas -lglog -lomp -pthread  $(CXXFLAGS)

targets = tensor_rank tensor_rank_try1
ofiles = prob.o tensor.o ../util.o ../discrete.o ../initial.o ../l2reg.o ../gauss_newton.o ../opts.o ../cholmod.o ../restrict/restrict.o ../restrict/als.o

all: $(targets)

tensor_rank: tensor_rank.o $(ofiles)
	$(CXX) $^ -o $@ $(LDFLAGS)
tensor_rank_try1: tensor_rank_try1.o $(ofiles)
	$(CXX) $^ -o $@ $(LDFLAGS)


../restrict/prob.h:# $(wildcard *.cpp)
	echo '#include "../rank/prob.h"' > $@
	
../restrict/tensor.h:# $(wildcard *.cpp)
	echo '#include "../rank/tensor.h"' > $@

../prob.h:# $(wildcard *.cpp)
	echo '#include "restrict/prob.h"' > $@
	
prob.o: prob.cpp
tensor.o: tensor.cpp
../tight.o: ../tight.cpp ../prob.h ../util.h ../opts.h ../discrete.h ../initial.h
../util.o: ../util.cpp ../prob.h ../util.h ../opts.h
../discrete.o: ../discrete.h ../util.h ../opts.h ../prob.h ../gauss_newton.h
../initial.o: ../initial.h ../opts.h ../prob.h
../l2reg.o: ../l2reg.h ../opts.h ../prob.h
../gauss_newton.o: ../l2reg.h ../opts.h ../prob.h
../opts.o: ../opts.h
../cholmod.o: ../prob.h
../restrict/restrict.o: ../restrict/restrict.cpp ../restrict/prob.h ../restrict/tensor.h
../restrict/als.o: ../restrict/tensor.h ../restrict/als.h

.PHONY: clean
clean:
	rm -f *.o ../*.o ../restrict/*.o
