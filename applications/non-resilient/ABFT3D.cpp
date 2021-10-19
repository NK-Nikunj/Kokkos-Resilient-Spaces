#include <Kokkos_Core.hpp>
#include <iostream>

#include <boost/program_options.hpp>

#define PI 3.1415926535897932384

typedef int int_t;

//#include "Tile3D.h"
KOKKOS_FUNCTION double left_flux(double left, double self, double cfl)
{
    return -cfl * left + cfl * self;
}

KOKKOS_FUNCTION double right_flux(double self, double right, double cfl)
{
    return -cfl * right + cfl * self;
}

KOKKOS_FUNCTION double stencil6p(double self, double xm, double xp, double ym,
    double yp, double zm, double zp, double cfl)
{
    return (1.0 - 6.0 * cfl) * self + cfl * (xm + xp + ym + yp + zm + zp);
}

int main(int argc, char* argv[])
{
    double cfl = 0.1;    // must be less than 1/6

    namespace bpo = boost::program_options;
    bpo::options_description desc("ABFT3D");

    desc.add_options()(
        "xsize", bpo::value<int>()->default_value(98), "X Dimension");
    desc.add_options()(
        "ysize", bpo::value<int>()->default_value(98), "Y Dimension");
    desc.add_options()(
        "zsize", bpo::value<int>()->default_value(98), "Z Dimension");

    bpo::variables_map vm;

    // Setup commandline arguments
    bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
    bpo::notify(vm);

    int xsize = vm["xsize"].as<int>();
    int ysize = vm["ysize"].as<int>();
    int zsize = vm["zsize"].as<int>();

    bool do_check_sum = true;

    Kokkos::initialize(argc, argv);
    {
        using range_policy = Kokkos::RangePolicy<>;

        Kokkos::View<double***> stencil_0(
            "data1", xsize + 2, ysize + 2, zsize + 2);
        Kokkos::View<double***> stencil_1(
            "data2", xsize + 2, ysize + 2, zsize + 2);

        // Soft copy
        auto stencil_old = stencil_0;
        auto stencil_new = stencil_1;
        auto stencil_tmp = stencil_new;
        Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Atomic>> checksum(
            "checksum", 1);

        Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> hchecksum(
            "h_checksum", 1);
        hchecksum[0] = 0.;

        Kokkos::deep_copy(checksum, hchecksum);

        // Initialize stencil
        Kokkos::parallel_for(
            "init", range_policy(0, ysize + 2), KOKKOS_LAMBDA(int j) {
                for (int k = 0; k < zsize + 2; ++k)
                {
                    for (int i = 0; i < xsize + 2; ++i)
                    {
                        stencil_old(i, j, k) = std::sin(1.0 * PI * (double) i /
                            (double) (xsize + 1));    //*
                        stencil_new(i, j, k) = stencil_old(i, j, k);
                    }
                }
            });
        Kokkos::fence();

        Kokkos::Timer timer;

        //
        // How to catch the error ... order matters.  where does the error come from?
        // Physical location could matter (plus interaction with OS)
        //

        // Original checksum
        if (do_check_sum)
        {
            Kokkos::parallel_for(
                "init_checksum", range_policy(1, xsize + 1),
                KOKKOS_LAMBDA(int i) {
                    for (int j = 1; j <= ysize; ++j)
                    {
                        for (int k = 1; k <= zsize; ++k)
                        {
                            checksum[0] += stencil_old(i, j, k);
                        }
                    }
                });
            Kokkos::fence();
        }

        Kokkos::deep_copy(hchecksum, checksum);

        std::cout << "Initial Checksum " << std::scientific << hchecksum[0]
                  << std::endl;

        double chk_ = 0;
        for (int its = 0; its < 10; ++its)
        {
            if (do_check_sum)
            {
                // Deduct the checksum from the boundary
                Kokkos::parallel_for(
                    "first_loop", range_policy(1, zsize + 1),
                    KOKKOS_LAMBDA(int k) {
                        for (int j = 1; j <= ysize; ++j)
                        {
                            checksum[0] -= left_flux(stencil_old(0, j, k),
                                               stencil_old(1, j, k), cfl) +
                                right_flux(stencil_old(xsize, j, k),
                                    stencil_old(xsize + 1, j, k), cfl);
                        }
                    });
                Kokkos::fence();

                Kokkos::parallel_for(
                    "second_loop", range_policy(1, zsize + 1),
                    KOKKOS_LAMBDA(int k) {
                        for (int i = 1; i <= xsize; ++i)
                        {
                            checksum[0] -= left_flux(stencil_old(i, 0, k),
                                               stencil_old(i, 1, k), cfl) +
                                right_flux(stencil_old(i, ysize, k),
                                    stencil_old(i, ysize + 1, k), cfl);
                        }
                    });
                Kokkos::fence();

                Kokkos::parallel_for(
                    "third_loop", range_policy(1, ysize + 1),
                    KOKKOS_LAMBDA(int j) {
                        for (int i = 1; i <= xsize; ++i)
                        {
                            checksum[0] -= left_flux(stencil_old(i, j, 0),
                                               stencil_old(i, j, 1), cfl) +
                                right_flux(stencil_old(i, j, zsize),
                                    stencil_old(i, j, zsize + 1), cfl);
                        }
                    });
                Kokkos::fence();
            }

            // Apply stencil
            // Replace the loops by parallel for
            //
            Kokkos::parallel_reduce(
                "stencil_op", range_policy(1, xsize + 1),
                KOKKOS_LAMBDA(int i, double& chk_) {
                    for (int j = 1; j <= ysize; ++j)
                    {
                        for (int k = 1; k <= zsize; ++k)
                        {
                            //std::cout << "Old " << i << " " << j << " " << k << ": " <<  stencil_old(i,j,k) << std::endl;
                            stencil_new(i, j, k) = stencil6p(
                                stencil_old(i, j, k), stencil_old(i - 1, j, k),
                                stencil_old(i + 1, j, k),
                                stencil_old(i, j - 1, k),
                                stencil_old(i, j + 1, k),
                                stencil_old(i, j, k - 1),
                                stencil_old(i, j, k + 1), cfl);

                            chk_ += stencil_new(i, j, k);
                        }
                    }
                },
                chk_);
            Kokkos::fence();

            // chk_ = 0;

            if (do_check_sum)
            {
                Kokkos::deep_copy(hchecksum, checksum);

                // Testing the checksum
                auto error = std::abs((chk_ - hchecksum[0]));

                std::cout << "New Checksum " << std::scientific << chk_
                          << " and analytical checksum " << hchecksum[0]
                          << " with Error: " << error << std::endl;

                if (error > 1e-2)
                {
                    std::cout << "Failed " << error << std::endl;
                }

                // Alternate stencil assignment
                if (its % 2 == 0)
                {
                    stencil_old = stencil_1;
                    stencil_new = stencil_0;
                }
                else
                {
                    stencil_old = stencil_0;
                    stencil_new = stencil_1;
                }

                chk_ = 0.;
            }
        }

        double time_taken = timer.seconds();

        std::cout << "Executed in: " << time_taken << std::endl;
    }

    Kokkos::finalize();
    return 0;
}