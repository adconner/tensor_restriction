CFLAGS=-O3 -ffast-math -march=native
CXXFLAGS=-std=c++14 -I/usr/include/eigen3 $(CFLAGS)
LDFLAGS=-lceres -lcxsparse -lcholmod -lumfpack -lamd -lcamd -lcolamd -lccolamd -lbtf -lsuitesparseconfig -lmetis -llapack -lopenblas -lglog -lomp -pthread
#LDFLAGS+=-static
CC=clang
CXX=clang++

cxxfiles=$(wildcard *.cpp)

target=main

all: $(target)

$(target): $(patsubst %.cpp,%.o,$(cxxfiles))
	$(CXX) $^ -o $@ $(LDFLAGS) 

main.o: main.cpp prob.h

clean:
	rm -f $(patsubst %.cpp,%.o,$(cxxfiles)) $(target)
