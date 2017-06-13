CFLAGS=-O3 -ffast-math -march=native
CXXFLAGS=-std=c++11 -I/usr/include/eigen3 $(CFLAGS)
LDFLAGS=-lceres -lcxsparse -lcholmod -lumfpack -lamd -lcamd -lcolamd -lccolamd -lbtf -lsuitesparseconfig -lmetis -llapack -lopenblas -lglog -lomp -pthread
#LDFLAGS+=-static
CC=clang
CXX=clang++

cxxfiles=$(wildcard *.cpp)

target=main

all: $(target)

$(target): $(patsubst %.cpp,%.o,$(cxxfiles))
	$(CXX) $^ -o $@ $(LDFLAGS) 

main: main.cpp prob.h

clean:
	rm -f $(patsubst %.c,%.o,$(cfiles)) $(patsubst %.cpp,%.o,$(cxxfiles)) $(target)
