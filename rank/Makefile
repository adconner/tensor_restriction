# CFLAGS=-g -fsanitize=address
CFLAGS=-g -O3 -ffast-math -march=native
# CFLAGS=-g
# CFLAGS += -DCX
CXXFLAGS=-std=c++20 $(shell pkg-config --cflags eigen3 lapacke) $(CFLAGS)
LDFLAGS=-lceres $(shell pkg-config --libs lapacke)
CC=clang
CXX=clang++

targets = tensor_rank tensor_rank_try1
ofiles = prob.o tensor.o ../util.o ../discrete.o ../initial.o ../l2reg.o ../opts.o ../restrict/restrict.o ../restrict/als.o

all: $(targets)

tensor_rank: tensor_rank.o $(ofiles)
	$(CXX) $^ -o $@ $(LDFLAGS)
tensor_rank_try1: tensor_rank_try1.o $(ofiles)
	$(CXX) $^ -o $@ $(LDFLAGS)


.PHONY: ../restrict/prob.h ../restrict/tensor.h ../prob.h
../restrict/prob.h:
	echo '#include "../rank/prob.h"' > $@
	
../restrict/tensor.h:
	echo '#include "../rank/tensor.h"' > $@

../prob.h:
	echo '#include "restrict/prob.h"' > $@
	
prob.o: prob.cpp
tensor.o: tensor.cpp
../tight.o: ../tight.cpp ../util.h ../opts.h ../discrete.h ../initial.h Makefile | ../prob.h
../util.o: ../util.cpp ../util.h ../opts.h Makefile | ../prob.h
../discrete.o: ../discrete.h ../util.h ../opts.h Makefile | ../prob.h
../initial.o: ../initial.h ../opts.h Makefile | ../prob.h
../l2reg.o: ../l2reg.h ../opts.h Makefile | ../prob.h
../gauss_newton.o: ../l2reg.h ../opts.h Makefile | ../prob.h
../opts.o: ../opts.h Makefile
../cholmod.o: Makefile | ../prob.h
../restrict/restrict.o: ../restrict/restrict.cpp ../restrict/prob.h ../restrict/tensor.h Makefile
../restrict/als.o: ../restrict/tensor.h ../restrict/als.h Makefile

.PHONY: clean
clean:
	rm -f *.o ../*.o ../restrict/*.o ../prob.h ../restrict/tensor.h ../restrict/prob.h
