//  Copyright (c) 2021 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <resilient_spaces/resilient_spaces.hpp>

#include <Kokkos_Core.hpp>

struct validator
{
    KOKKOS_FUNCTION bool operator()(int, int) const
    {
        return true;
    }
};

struct reduction_validator
{
    KOKKOS_FUNCTION bool operator()(double const& result) const
    {
        return (result == 100);
    }
};

struct operation
{
    KOKKOS_FUNCTION int operator()(int) const
    {
        return 42;
    }
};

struct reduction_op
{
    KOKKOS_FUNCTION void operator()(const int i, double& sum) const
    {
        sum += 1;
    }
};

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        validator validate{};
        reduction_validator red_val{};
        operation op{};
        reduction_op red_op{};

        // Host only variant
        {
            Kokkos::DefaultHostExecutionSpace inst{};

            // Replay Strategy
            Kokkos::resilience::ResilientReplay<
                Kokkos::DefaultHostExecutionSpace, validator>
                replay_inst(3, validate, inst);

            // Replicate strategies
            Kokkos::resilience::ResilientReplicate<
                Kokkos::DefaultHostExecutionSpace>
                replicate_inst(inst);
            Kokkos::resilience::ResilientReplicateValidate<
                Kokkos::DefaultHostExecutionSpace, validator>
                replicate_validate_inst(3, validate, inst);

            Kokkos::parallel_for(
                Kokkos::RangePolicy<Kokkos::resilience::ResilientReplay<
                    Kokkos::DefaultHostExecutionSpace, validator>>(
                    replay_inst, 0, 100),
                op);
            Kokkos::fence();

            Kokkos::parallel_for(
                Kokkos::RangePolicy<Kokkos::resilience::ResilientReplicate<
                    Kokkos::DefaultHostExecutionSpace>>(replicate_inst, 0, 100),
                op);
            Kokkos::fence();

            Kokkos::parallel_for(
                Kokkos::RangePolicy<
                    Kokkos::resilience::ResilientReplicateValidate<
                        Kokkos::DefaultHostExecutionSpace, validator>>(
                    replicate_validate_inst, 0, 100),
                op);
            Kokkos::fence();

            double sum;
            // Replay Strategy
            Kokkos::resilience::ResilientReplay<
                Kokkos::DefaultHostExecutionSpace, reduction_validator>
                replay_reduce_inst(3, red_val, inst);

            Kokkos::parallel_reduce(
                Kokkos::RangePolicy<Kokkos::resilience::ResilientReplay<
                    Kokkos::DefaultHostExecutionSpace, reduction_validator>>(
                    replay_reduce_inst, 0, 100),
                red_op,
                Kokkos::Sum<double, Kokkos::DefaultHostExecutionSpace>(sum));
            std::cout << "[Sum]: " << sum << std::endl;
        }

        // Device only variant
        {
            Kokkos::DefaultExecutionSpace inst{};

            // Replay Strategy
            Kokkos::resilience::ResilientReplay<Kokkos::DefaultExecutionSpace,
                validator>
                replay_inst(3, validate, inst);

            // Replicate strategies
            Kokkos::resilience::ResilientReplicate<
                Kokkos::DefaultExecutionSpace>
                replicate_inst(inst);
            Kokkos::resilience::ResilientReplicateValidate<
                Kokkos::DefaultExecutionSpace, validator>
                replicate_validate_inst(3, validate, inst);

            Kokkos::parallel_for(
                Kokkos::RangePolicy<Kokkos::resilience::ResilientReplay<
                    Kokkos::DefaultExecutionSpace, validator>>(
                    replay_inst, 0, 100),
                op);
            Kokkos::fence();

            Kokkos::parallel_for(
                Kokkos::RangePolicy<Kokkos::resilience::ResilientReplicate<
                    Kokkos::DefaultExecutionSpace>>(
                    replicate_validate_inst, 0, 100),
                op);
            Kokkos::fence();

            Kokkos::parallel_for(
                Kokkos::RangePolicy<
                    Kokkos::resilience::ResilientReplicateValidate<
                        Kokkos::DefaultExecutionSpace, validator>>(
                    replicate_validate_inst, 0, 100),
                op);
            Kokkos::fence();
        }
        std::cout << "Execution Complete" << std::endl;
    }

    Kokkos::finalize();

    return 0;
}