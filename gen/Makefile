CFLAGS=-O3 -ffast-math -march=native -O0
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
sotarget=lib.so

cpp: $(target)

py: $(sotarget)

$(target): $(patsubst %.c,%.o,$(cfiles)) $(patsubst %.cpp,%.o,$(cxxfiles))
	$(CXX) $(LDFLAGS) $^ -o $@

$(sotarget): $(patsubst %.c,%.o,$(cfiles))
	$(CC) -shared $^ -o $@

main.o: main.cpp polyudp.cpp

clean:
	rm -f $(patsubst %.c,%.o,$(cfiles)) $(patsubst %.cpp,%.o,$(cxxfiles)) $(target) $(sotarget)
