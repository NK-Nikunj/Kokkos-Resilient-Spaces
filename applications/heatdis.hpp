#ifndef INC_HEATDIS_HEATDIS_HPP
#define INC_HEATDIS_HEATDIS_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#define ITER_OUT 50
#define WORKTAG 5
#define REDUCED 1

namespace heatdis {
    void initData(int nbLines, int M, int rank, Kokkos::View<double*> h);

    double doWork(int numprocs, int rank, int M, int nbLines,
        Kokkos::View<double*> g, Kokkos::View<double*> h);
}    // namespace heatdis

#endif    // INC_HEATDIS_HEATDIS_HPP
