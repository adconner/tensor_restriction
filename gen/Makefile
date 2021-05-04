# CFLAGS=-g -fsanitize=address
CFLAGS=-g -O3 -ffast-math -march=native
# CFLAGS=-g
CXXFLAGS=-std=c++14 -I/usr/include/eigen3 $(shell pkg-config --cflags clp) $(CFLAGS)
LDFLAGS=-llapacke $(shell pkg-config --libs clp) -lceres -lcxsparse -lcholmod -lumfpack -lamd -lcamd -lcolamd -lccolamd -lbtf -lsuitesparseconfig -lmetis -llapack -lopenblas -lglog -lomp -pthread  $(CXXFLAGS)
CC=clang
CXX=clang++

# .SUFFIXES:

ofiles=initial.o l2reg.o util.o opts.o discrete.o
# ofiles+=$(sort $(patsubst %.c,%.o,$(wildcard gen/*.c)))
# ofiles+=$(sort $(patsubst %.cpp,%.o,$(wildcard gen/*.cpp)))
ofiles+=$(sort $(patsubst %.cpp,%.o,$(wildcard restrict/*.cpp)))

main: main.o $(ofiles)
	$(CXX) $^ -o $@ $(LDFLAGS)

tight: tight.o $(ofiles)
	$(CXX) $^ -o $@ $(LDFLAGS)

main.o: main.cpp prob.h util.h opts.h discrete.h initial.h
tight.o: tight.cpp prob.h util.h opts.h discrete.h initial.h
util.o: util.cpp prob.h util.h opts.h
discrete.o: discrete.h util.h opts.h prob.h gauss_newton.h
initial.o: initial.h opts.h prob.h
l2reg.o: l2reg.h opts.h prob.h
gauss_newton.o: l2reg.h opts.h prob.h
opts.o: opts.h
restrict/restrict.o: restrict/restrict.cpp restrict/prob.h restrict/tensor.h

# %.o: %.cpp
# 	$(CXX) $(CXXFLAGS) $< -c -o $@

# clean:
# 	rm -f $(patsubst %.cpp,%.o,$(wildcard *.cpp)) $(target)
