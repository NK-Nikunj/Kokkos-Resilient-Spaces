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

    template <typename FunctorType, typename ReducerType, typename... Traits>
    class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>,
        ReducerType,
        Kokkos::resilience::ResilientReplay<
            typename Kokkos::resilience::traits::RangePolicyExtracter<
                Traits...>::base_execution_space,
            typename Kokkos::resilience::traits::RangePolicyExtracter<
                Traits...>::validator>>
    {
    public:
        using Policy = Kokkos::RangePolicy<Traits...>;
        using BasePolicy =
            typename Kokkos::resilience::traits::RangePolicyExtracter<
                Traits...>::RangePolicy;
        using validator_type =
            typename Kokkos::resilience::traits::RangePolicyExtracter<
                Traits...>::validator;
        using base_execution_space =
            typename Kokkos::resilience::traits::RangePolicyExtracter<
                Traits...>::base_execution_space;

        using base_type = ParallelReduce<FunctorType, BasePolicy, ReducerType,
            base_execution_space>;

        // Reducer specific typedefs
        using WorkTag = typename Policy::work_tag;
        using WorkRange = typename Policy::WorkRange;
        using Member = typename Policy::member_type;

        using Analysis = FunctorAnalysis<FunctorPatternInterface::REDUCE,
            BasePolicy, FunctorType>;

        using value_type = typename Analysis::value_type;
        using pointer_type = typename Analysis::pointer_type;
        using reference_type = typename Analysis::reference_type;

        ParallelReduce(FunctorType const& arg_functor, Policy const& arg_policy,
            const ReducerType& reducer)
          : m_functor(arg_functor)
          , m_policy(arg_policy)
          , m_reducer(reducer)
          , m_result_ptr(reducer.view().data())
        {
        }

        void execute() const
        {
            bool is_correct{false};
            auto initial_value = *m_result_ptr;
            for (std::size_t i = 0; i != m_policy.space().replays(); ++i)
            {
                base_type closure(m_functor, m_policy, m_reducer);
                closure.execute();

                bool result = m_policy.space().validator()(*m_result_ptr);

                if (result)
                {
                    is_correct = true;
                    break;
                }

                *m_result_ptr = initial_value;
            }

            if (!is_correct)
                throw std::runtime_error("Program ran out of replay options.");
        }

    private:
        const FunctorType m_functor;
        const Policy m_policy;
        const ReducerType m_reducer;
        const pointer_type m_result_ptr;
    };
}}    // namespace Kokkos::Impl