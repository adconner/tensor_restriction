# CFLAGS=-g -fsanitize=address
CFLAGS=-O3 -ffast-math -march=native
# CFLAGS=-g
CXXFLAGS=-std=c++14 -I/usr/include/eigen3 $(CFLAGS)
LDFLAGS=-lceres -lcxsparse -lcholmod -lumfpack -lamd -lcamd -lcolamd -lccolamd -lbtf -lsuitesparseconfig -lmetis -llapack -lopenblas -lglog -lomp -pthread
CC=clang
CXX=clang++

# .SUFFIXES:

ofiles=main.o util.o discrete.o prob.o initial.o
ofiles+=$(sort $(patsubst %.c,%.o,$(wildcard ccode/prob*.c)))
targets=main

all: $(targets)

main: $(ofiles)
	$(CXX) $^ -o $@ $(LDFLAGS)

main.o: main.cpp prob.h util.h opts.h discrete.h initial.h
util.o: util.cpp prob.h util.h opts.h
discrete.o: discrete.h util.h opts.h prob.h
initial.o: initial.h opts.h

# %.o: %.cpp
# 	$(CXX) $(CXXFLAGS) $< -c -o $@

# clean:
# 	rm -f $(patsubst %.cpp,%.o,$(wildcard *.cpp)) $(target)
