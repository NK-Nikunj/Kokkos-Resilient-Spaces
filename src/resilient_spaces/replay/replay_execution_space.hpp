//  Copyright (c) 2021 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <Kokkos_Core.hpp>

namespace Kokkos { namespace resilience {

    template <typename ExecutionSpace, typename Validator>
    class ResilientReplay : public ExecutionSpace
    {
    public:
        // Typedefs for the ResilientReplay Execution Space
        using base_execution_space = ExecutionSpace;
        using validator_type = Validator;

        // Typedefs from ExecutionSpace
        using execution_space = ResilientReplay;
        using memory_space = typename ExecutionSpace::memory_space;
        using device_type = typename ExecutionSpace::device_type;
        using size_type = typename ExecutionSpace::size_type;
        using scratch_memory_space =
            typename ExecutionSpace::scratch_memory_space;

        template <typename... Args>
        ResilientReplay(std::uint64_t n, Validator const& validator,
            Args&&... args) noexcept
          : ExecutionSpace(args...)
          , validator_(validator)
          , replays_(n)
        {
        }

        Validator const& validator() const noexcept
        {
            return validator_;
        }

        std::uint64_t replays() const noexcept
        {
            return replays_;
        }

        KOKKOS_FUNCTION ResilientReplay(
            ResilientReplay&& other) noexcept = default;
        KOKKOS_FUNCTION ResilientReplay(ResilientReplay const& other) = default;

    private:
        const Validator validator_;
        const std::uint64_t replays_;
    };

}}    // namespace Kokkos::resilience

namespace Kokkos { namespace Tools { namespace Experimental {

    template <typename ExecutionSpace, typename Validator>
    struct DeviceTypeTraits<
        Kokkos::resilience::ResilientReplay<ExecutionSpace, Validator>>
    {
        static constexpr DeviceType id = DeviceTypeTraits<ExecutionSpace>::id;
    };

}}}    // namespace Kokkos::Tools::Experimental
