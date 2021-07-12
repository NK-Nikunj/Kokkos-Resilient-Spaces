//  Copyright (c) 2021 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

namespace Kokkos { namespace resilience { namespace traits {

    template <typename ExecutionSpace, typename... Traits>
    struct RangePolicyBase
    {
        using execution_space = ExecutionSpace;
        using base_execution_space =
            typename execution_space::base_execution_space;
        using RangePolicy =
            Kokkos::RangePolicy<base_execution_space, Traits...>;
        using validator = typename execution_space::validator_type;
    };

}}}    // namespace Kokkos::resilience::traits
