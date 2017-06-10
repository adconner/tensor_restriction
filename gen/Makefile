CFLAGS=-O3 -ffast-math -march=native 
#CFLAGS+=-DNDEBUG
CXXFLAGS=-std=c++11 -I/usr/include/eigen3 -I/usr/include/coin
CXXFLAGS+=$(CFLAGS)
LDFLAGS=-lnlopt -lipopt -pthread
#LDFLAGS+=-static
CC=clang
CXX=clang++

cfiles=f.c jac.c hess.c
cxxfiles=main.cpp

target=main

all: $(target)

$(target): $(patsubst %.c,%.o,$(cfiles)) $(patsubst %.cpp,%.o,$(cxxfiles))
	$(CXX) $(LDFLAGS) $^ -o $@

main.o: main.cpp polyudp.cpp

clean:
	rm -f $(patsubst %.c,%.o,$(cfiles)) $(patsubst %.cpp,%.o,$(cxxfiles)) $(target)
