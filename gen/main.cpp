#include <pagmo/pagmo.hpp>
using namespace pagmo;
#include "polyudp.cpp"

int main() {
  /* problem prob{schwefel(30)}; */
  problem prob{PolyUDP()};
  algorithm algo{sade(100)};
  archipelago archi{16,algo,prob,20};
  archi.evolve(10);
  archi.wait();
  for (const auto &isl : archi) {
    print(isl.get_population().champion_f(), "\n");
  }
}
