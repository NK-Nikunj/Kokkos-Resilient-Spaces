//  Copyright (c) 2021 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <resilient_spaces/replay/replay_execution_space.hpp>

#include <resilient_spaces/util/functor.hpp>
#include <resilient_spaces/util/traits.hpp>

namespace Kokkos { namespace Impl {

    template <typename FunctorType, typename... Traits>
    class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
        Kokkos::resilience::ResilientReplay<
            typename Kokkos::resilience::traits::RangePolicyBase<
                Traits...>::base_execution_space,
            typename Kokkos::resilience::traits::RangePolicyBase<
                Traits...>::validator>>
    {
    public:
        using Policy = Kokkos::RangePolicy<Traits...>;
        using BasePolicy = typename Kokkos::resilience::traits::RangePolicyBase<
            Traits...>::RangePolicy;
        using validator_type =
            typename Kokkos::resilience::traits::RangePolicyBase<
                Traits...>::validator;
        using base_execution_space =
            typename Kokkos::resilience::traits::RangePolicyBase<
                Traits...>::base_execution_space;
        using base_type = ParallelFor<
            Kokkos::resilience::util::ResilientReplayValidateFunctor<
                base_execution_space, FunctorType, validator_type>,
            typename Kokkos::resilience::traits::RangePolicyBase<
                Traits...>::RangePolicy,
            typename Kokkos::resilience::traits::RangePolicyBase<
                Traits...>::base_execution_space>;

        ParallelFor(FunctorType const& arg_functor, const Policy& arg_policy)
          : m_functor(arg_functor)
          , m_policy(arg_policy)
        {
        }

        void execute() const
        {
            Kokkos::resilience::util::ResilientReplayValidateFunctor<
                base_execution_space, FunctorType, validator_type>
                inst(m_functor, m_policy.space().validator(),
                    m_policy.space().replays());

            // Call the underlying ParallelFor
            base_type closure(inst, m_policy);
            closure.execute();

            if (inst.is_incorrect())
                throw std::runtime_error("Program ran out of replay options.");
        }

    private:
        const FunctorType m_functor;
        const Policy m_policy;
    };

}}    // namespace Kokkos::Impl
