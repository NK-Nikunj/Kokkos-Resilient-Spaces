#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <mpi.h>

#include "heatdis.hpp"

/* Added to test restart */
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

using namespace heatdis;

/*
    This sample application is based on the heat distribution code
    originally developed within the FTI project: github.com/leobago/fti
*/

int main(int argc, char* argv[])
{
    std::size_t rank, nbProcs, nbLines, M;
    double wtime, memSize, localerror, globalerror = 1;

    namespace bpo = boost::program_options;
    bpo::options_description desc("Heatdis");

    desc.add_options()(
        "size", bpo::value<std::size_t>()->default_value(100u), "Problem Size");
    desc.add_options()("nsteps", bpo::value<std::size_t>()->default_value(600u),
        "Number of timesteps");
    desc.add_options()("precision",
        bpo::value<double>()->default_value(0.00001), "Minimum precision");
    // desc.add_options()("config", bpo::value<std::string>());
    desc.add_options()(
        "scale", bpo::value<std::string>()->default_value("weak"));

    bpo::variables_map vm;

    // Setup commandline arguments
    bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
    bpo::notify(vm);

    std::size_t nsteps = vm["nsteps"].as<std::size_t>();
    const auto precision = vm["precision"].as<double>();

    int strong = 0, str_ret;

    std::string scale = vm["scale"].as<std::string>();
    if (scale == "strong")
        strong = 1;

    MPI_Init(&argc, &argv);
    int mpi_nbProcs, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_nbProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    nbProcs = mpi_nbProcs, rank = mpi_rank;

    std::size_t mem_size = vm["size"].as<std::size_t>();

    if (mem_size == 0)
    {
        printf("Wrong memory size! See usage\n");
        exit(3);
    }

    Kokkos::initialize(argc, argv);
    {
        if (!strong)
        {
            /* weak scaling */
            M = sqrt((double) (mem_size * 1024.0 * 1024.0 * nbProcs) /
                (2 * sizeof(double)));    // two matrices needed
            nbLines = (M / nbProcs) + 3;
        }
        else
        {
            /* strong scaling */
            M = sqrt((double) (mem_size * 1024.0 * 1024.0 * nbProcs) /
                (2 * sizeof(double) * nbProcs));    // two matrices needed
            nbLines = (M / nbProcs) + 3;
        }

        Kokkos::View<double*> h_view("h", M * nbLines);
        Kokkos::View<double*> g_view("g", M * nbLines);

        initData(nbLines, M, rank, g_view);

        memSize = M * nbLines * 2 * sizeof(double) / (1024 * 1024);

        // printf("nbProcs: %d\n", nbProcs);
        // printf("size of double: %d\n", sizeof(double));

        if (rank == 0)
            if (!strong)
            {
                printf("Local data size is %lu x %lu = %f MB (%lu).\n", M,
                    nbLines, memSize, mem_size);
            }
            else
            {
                printf("Local data size is %lu x %lu = %f MB (%lu).\n", M,
                    nbLines, memSize, mem_size / nbProcs);
            }
        if (rank == 0)
            printf("Target precision : %f \n", precision);
        if (rank == 0)
            printf("Maximum number of iterations : %lu \n", nsteps);

        wtime = MPI_Wtime();
        int i = 0;
        int v = i;
        printf("i: %d --- v: %d\n", i, v);
        if (i < 0)
        {
            i = 0;
        }

        while (i < nsteps)
        {
            localerror = doWork(nbProcs, rank, M, nbLines, g_view, h_view);

            if (((i % ITER_OUT) == 0) && (rank == 0))
            {
                printf("Step : %d, error = %f\n", i, globalerror);
            }
            if ((i % REDUCED) == 0)
            {
                MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
            }

            if (globalerror < precision)
            {
                printf("PRECISION ERROR\n");
                break;
            }
            i++;
        }
        if (rank == 0)
            printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);
    }
    Kokkos::finalize();

    MPI_Finalize();
    return 0;
}
