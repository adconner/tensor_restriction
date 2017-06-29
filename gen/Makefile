# CFLAGS=-g -fsanitize=address
# CFLAGS=-O3 -ffast-math -march=native -g
# CFLAGS=-g
CXXFLAGS=-std=c++14 -I/usr/include/eigen3 $(CFLAGS)
LDFLAGS=-lceres -lcxsparse -lcholmod -lumfpack -lamd -lcamd -lcolamd -lccolamd -lbtf -lsuitesparseconfig -lmetis -llapack -lopenblas -lglog -lomp -pthread
#LDFLAGS+=-static
CC=clang
CXX=clang++

.SUFFIXES:

ofiles=prob.o
targets=main check

all: $(targets)

%: %.o $(ofiles)
	$(CXX) $^ -o $@ $(LDFLAGS) 

main.o: main.cpp prob.h
check.o: check.cpp prob.h

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

# clean:
# 	rm -f $(patsubst %.cpp,%.o,$(wildcard *.cpp)) $(target)
