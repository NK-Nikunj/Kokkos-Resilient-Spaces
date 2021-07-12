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

struct operation
{
    KOKKOS_FUNCTION int operator()(int) const
    {
        return 42;
    }
};

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        validator validate{};
        operation op{};

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