#ifndef INC_HEATDIS_HEATDIS_HPP
#define INC_HEATDIS_HEATDIS_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include <resilient_spaces/resilient_spaces.hpp>

#define ITER_OUT 50
#define WORKTAG 5
#define REDUCED 1

namespace heatdis {
    void initData(std::size_t nbLines, std::size_t M, std::size_t rank,
        Kokkos::View<double*> h);

    double doWork(std::size_t numprocs, std::size_t rank, std::size_t M,
        std::size_t nbLines, Kokkos::View<double*> g, Kokkos::View<double*> h);
}    // namespace heatdis

#endif    // INC_HEATDIS_HEATDIS_HPP
