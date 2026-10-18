#include "Trajectory.hpp"
#include <vector>

void writeXyzFrame(std::FILE* file, const System& system, double time) {
    const std::size_t n = system.size();
    const double l = system.getBoxLength();

    // Storage index of each id: the inverse of the permutation the neighbour list sorts by
    std::vector<std::size_t> indexOf(n);
    const auto ids = system.ids();
    for (std::size_t i = 0; i < n; ++i) {
        indexOf[ids[i]] = i;
    }

    std::fprintf(file, "%zu\n", n);
    std::fprintf(file,
                 "Lattice=\"%.10g 0 0 0 %.10g 0 0 0 %.10g\" Properties=species:S:1:pos:R:3:velo:R:3:id:I:1 "
                 "Time=%.6f pbc=\"T T T\"\n",
                 l, l, l, time);
    for (std::size_t id = 0; id < n; ++id) {
        const auto r = system.positionOf(indexOf[id]);
        const auto v = system.velocityOf(indexOf[id]);
        std::fprintf(file, "Ar %.8f %.8f %.8f %.8f %.8f %.8f %zu\n", r[0], r[1], r[2], v[0], v[1], v[2], id);
    }
}
