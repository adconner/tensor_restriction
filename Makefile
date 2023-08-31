# CFLAGS=-g -fsanitize=address
CFLAGS=-g -O3 -ffast-math -march=native
# CFLAGS=-g
# CXXFLAGS=-std=c++20 -I/usr/include/eigen3 $(shell pkg-config --cflags lapacke clp) $(CFLAGS)
# LDFLAGS=-llapacke $(shell pkg-config --libs lapacke clp) -lceres -lcxsparse -lcholmod -lumfpack -lamd -lcamd -lcolamd -lccolamd -lbtf -lsuitesparseconfig -lmetis -llapack -lopenblas -lglog -lomp -pthread  $(CXXFLAGS)
CXXFLAGS=-std=c++20 $(shell pkg-config --cflags eigen3 lapacke) $(CFLAGS)
LDFLAGS=-lceres $(shell pkg-config --libs lapacke)
CC=clang
CXX=clang++

# .SUFFIXES:

ofiles=initial.o l2reg.o util.o opts.o discrete.o
# ofiles=initial.o l2reg.o util.o opts.o discrete.o cholmod.o
# ofiles+=$(sort $(patsubst %.c,%.o,$(wildcard gen/*.c)))
# ofiles+=$(sort $(patsubst %.cpp,%.o,$(wildcard gen/*.cpp)))
ofiles+=$(sort $(patsubst %.cpp,%.o,$(wildcard restrict/*.cpp)))

main: main.o $(ofiles)
	$(CXX) $^ -o $@ $(LDFLAGS)

tight: tight.o $(ofiles)
	$(CXX) $^ -o $@ $(LDFLAGS)

main.o: main.cpp prob.h util.h opts.h discrete.h initial.h cholmod.h
tight.o: tight.cpp prob.h util.h opts.h discrete.h initial.h
util.o: util.cpp prob.h util.h opts.h
discrete.o: discrete.h util.h opts.h prob.h gauss_newton.h
initial.o: initial.h opts.h prob.h
l2reg.o: l2reg.h opts.h prob.h
gauss_newton.o: l2reg.h opts.h prob.h
opts.o: opts.h
cholmod.o: prob.h
restrict/restrict.o: restrict/restrict.cpp restrict/prob.h restrict/tensor.h
restrict/als.o: restrict/tensor.h restrict/als.h

# %.o: %.cpp
# 	$(CXX) $(CXXFLAGS) $< -c -o $@

# clean:
# 	rm -f $(patsubst %.cpp,%.o,$(wildcard *.cpp)) $(target)
