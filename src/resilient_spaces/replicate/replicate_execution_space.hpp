//  Copyright (c) 2021 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <Kokkos_Core.hpp>

namespace Kokkos { namespace resilience {

    template <typename ExecutionSpace, typename Validator>
    class ResilientReplicateValidate : public ExecutionSpace
    {
    public:
        // Typedefs for the ResilientReplicate Execution Space
        using base_execution_space = ExecutionSpace;
        using validator_type = Validator;

        using execution_space = ResilientReplicateValidate;
        using memory_space = typename ExecutionSpace::memory_space;
        using device_type = typename ExecutionSpace::device_type;
        using size_type = typename ExecutionSpace::size_type;
        using scratch_memory_space =
            typename ExecutionSpace::scratch_memory_space;

        template <typename... Args>
        ResilientReplicateValidate(std::uint64_t n, Validator const& validator,
            Args&&... args) noexcept
          : ExecutionSpace(args...)
          , validator_(validator)
          , replicates_(n)
        {
        }

        Validator const& validator() const noexcept
        {
            return validator_;
        }

        std::uint64_t replicates() const noexcept
        {
            return replicates_;
        }

        KOKKOS_FUNCTION ResilientReplicateValidate(
            ResilientReplicateValidate&& other) noexcept = default;
        KOKKOS_FUNCTION ResilientReplicateValidate(
            ResilientReplicateValidate const& other) = default;

    private:
        const Validator validator_;
        const std::uint64_t replicates_;
    };

    template <typename ExecutionSpace>
    class ResilientReplicate : public ExecutionSpace
    {
    public:
        // Typedefs for the ResilientReplicate Execution Space
        using base_execution_space = ExecutionSpace;
        using validator_type = void;

        using execution_space = ResilientReplicate;
        using memory_space = typename ExecutionSpace::memory_space;
        using device_type = typename ExecutionSpace::device_type;
        using size_type = typename ExecutionSpace::size_type;
        using scratch_memory_space =
            typename ExecutionSpace::scratch_memory_space;

        template <typename... Args>
        ResilientReplicate(Args&&... args) noexcept
          : ExecutionSpace(args...)
        {
        }

        KOKKOS_FUNCTION ResilientReplicate(
            ResilientReplicate&& other) noexcept = default;
        KOKKOS_FUNCTION ResilientReplicate(
            ResilientReplicate const& other) = default;
    };

}}    // namespace Kokkos::resilience

namespace Kokkos { namespace Tools { namespace Experimental {

    template <typename ExecutionSpace>
    struct DeviceTypeTraits<
        Kokkos::resilience::ResilientReplicate<ExecutionSpace>>
    {
        static constexpr DeviceType id = DeviceTypeTraits<ExecutionSpace>::id;
    };

    template <typename ExecutionSpace, typename Validator>
    struct DeviceTypeTraits<Kokkos::resilience::ResilientReplicateValidate<
        ExecutionSpace, Validator>>
    {
        static constexpr DeviceType id = DeviceTypeTraits<ExecutionSpace>::id;
    };

}}}    // namespace Kokkos::Tools::Experimental
