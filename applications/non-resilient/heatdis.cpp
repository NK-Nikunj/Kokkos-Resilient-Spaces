#include "heatdis.hpp"

namespace heatdis {
    void initData(std::size_t nbLines, std::size_t M, std::size_t rank,
        Kokkos::View<double*> h)
    {
        using range_policy = Kokkos::RangePolicy<>;
        using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

        /* set all of the data to 0 */
        Kokkos::parallel_for(
            "init_h", mdrange_policy({0u, 0u}, {nbLines, M}),
            KOKKOS_LAMBDA(
                std::size_t i, std::size_t j) { h((i * M) + j) = 0; });

        /* initialize some values on rank 0 to 100 */
        if (rank == 0)
        {
            int j = ceil(M * 0.9);
            Kokkos::parallel_for(
                "init_rank0", range_policy((std::size_t)(M * 0.1), j),
                KOKKOS_LAMBDA(std::size_t i) { h(i) = 100; });
        }
    }

    double doWork(std::size_t numprocs, std::size_t rank, std::size_t M,
        std::size_t nbLines, Kokkos::View<double*> g, Kokkos::View<double*> h)
    {
        MPI_Request req1[2], req2[2];
        MPI_Status status1[2], status2[2];
        // double localerror;
        // localerror = 0;
        Kokkos::View<double*> localerror("localerror", 1);
        Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>
            localerror_host("localerror_host", 1);
        localerror_host[0] = 0.;
        Kokkos::deep_copy(localerror, localerror_host);

        // using range_policy = Kokkos::RangePolicy<
        //     Kokkos::resilience::ResilientReplicate<Kokkos::Cuda>>;
        // using mdrange_policy = Kokkos::MDRangePolicy<
        //     Kokkos::resilience::ResilientReplicate<Kokkos::Cuda>,
        //     Kokkos::Rank<2>>;

        using range_policy = Kokkos::RangePolicy<>;
        using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

        Kokkos::parallel_for(
            "copy_g", mdrange_policy({0u, 0u}, {nbLines, M}),
            KOKKOS_LAMBDA(std::size_t i, std::size_t j) {
                h((i * M) + j) = g((i * M) + j);

                // return h((i * M) + j);
            });

        double* g_raw = g.data();
        double* h_raw = h.data();

        /* Send and receive data from the left -- all ranks besides 0 because there are no ranks to its left */
        if (rank > 0)
        {
            MPI_Isend(g_raw + M, M, MPI_DOUBLE, rank - 1, WORKTAG,
                MPI_COMM_WORLD, &req1[0]);
            MPI_Irecv(h_raw, M, MPI_DOUBLE, rank - 1, WORKTAG, MPI_COMM_WORLD,
                &req1[1]);
        }

        /* Send and receive data from the right -- all ranks besides numprocs - 1
     * because there are no ranks to its right */
        if (rank < numprocs - 1)
        {
            MPI_Isend(g_raw + ((nbLines - 2) * M), M, MPI_DOUBLE, rank + 1,
                WORKTAG, MPI_COMM_WORLD, &req2[0]);
            MPI_Irecv(h_raw + ((nbLines - 1) * M), M, MPI_DOUBLE, rank + 1,
                WORKTAG, MPI_COMM_WORLD, &req2[1]);
        }

        /* this should probably include ALL ranks 
     * (currently excludes leftmost and rightmost)
     */
        if (rank > 0)
        {
            MPI_Waitall(2, req1, status1);
        }
        if (rank < numprocs - 1)
        {
            MPI_Waitall(2, req2, status2);
        }

        /* perform the computation */
        Kokkos::parallel_for("compute",
            mdrange_policy({1u, 0u}, {nbLines - 1, M}),
            [localerror, g, h, M] __host__ __device__(
                std::size_t i, std::size_t j) {
                g((i * M) + j) = 0.25 *
                    (h(((i - 1) * M) + j) + h(((i + 1) * M) + j) +
                        h((i * M) + j - 1) + h((i * M) + j + 1));
                if (localerror[0] < fabs(g((i * M) + j) - h((i * M) + j)))
                {
                    localerror[0] = fabs(g((i * M) + j) - h((i * M) + j));
                }

                // return fabs(g((i * M) + j) - h((i * M) + j));
            });

        /* perform computation on right-most rank */
        if (rank == (numprocs - 1))
        {
            Kokkos::parallel_for("compute_right", range_policy(0, M),
                [g, nbLines, M] __host__ __device__(int j) {
                    g(((nbLines - 1) * M) + j) = g(((nbLines - 2) * M) + j);

                    // return g(((nbLines - 1) * M) + j);
                });
        }

        Kokkos::deep_copy(localerror_host, localerror);

        return localerror_host[0];
    }
}    // namespace heatdis
