//  Copyright (c) 2021 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <resilient_spaces/replicate/replicate_execution_space.hpp>

#include <resilient_spaces/util/functor.hpp>
#include <resilient_spaces/util/traits.hpp>

namespace Kokkos { namespace Impl {

    template <typename FunctorType, typename... Traits>
    class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
        Kokkos::resilience::ResilientReplicateValidate<
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
        using base_type = ParallelFor<
            Kokkos::resilience::util::ResilientReplicateValidateFunctor<
                base_execution_space, FunctorType, validator_type>,
            typename Kokkos::resilience::traits::RangePolicyExtracter<
                Traits...>::RangePolicy,
            typename Kokkos::resilience::traits::RangePolicyExtracter<
                Traits...>::base_execution_space>;

        ParallelFor(FunctorType const& arg_functor, const Policy& arg_policy)
          : m_functor(arg_functor)
          , m_policy(arg_policy)
        {
        }

        void execute() const
        {
            Kokkos::resilience::util::ResilientReplicateValidateFunctor<
                base_execution_space, FunctorType, validator_type>
                inst(m_functor, m_policy.space().validator(),
                    m_policy.space().replicates());

            // Call the underlying ParallelFor
            base_type closure(inst, m_policy);
            closure.execute();

            if (inst.is_incorrect())
                throw std::runtime_error(
                    "All replicate returned incorrect result.");
        }

    private:
        const FunctorType m_functor;
        const Policy m_policy;
    };

    template <typename FunctorType, typename... Traits>
    class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
        Kokkos::resilience::ResilientReplicateValidate<
            typename Kokkos::resilience::traits::MDRangePolicyExtracter<
                Traits...>::base_execution_space,
            typename Kokkos::resilience::traits::MDRangePolicyExtracter<
                Traits...>::validator>>
    {
    public:
        using Policy = Kokkos::MDRangePolicy<Traits...>;
        using BasePolicy =
            typename Kokkos::resilience::traits::MDRangePolicyExtracter<
                Traits...>::MDRangePolicy;
        using validator_type =
            typename Kokkos::resilience::traits::MDRangePolicyExtracter<
                Traits...>::validator;
        using base_execution_space =
            typename Kokkos::resilience::traits::MDRangePolicyExtracter<
                Traits...>::base_execution_space;
        using base_type = ParallelFor<
            Kokkos::resilience::util::ResilientReplicateValidateFunctor<
                base_execution_space, FunctorType, validator_type>,
            typename Kokkos::resilience::traits::MDRangePolicyExtracter<
                Traits...>::MDRangePolicy,
            typename Kokkos::resilience::traits::MDRangePolicyExtracter<
                Traits...>::base_execution_space>;

        ParallelFor(FunctorType const& arg_functor, const Policy& arg_policy)
          : m_functor(arg_functor)
          , m_policy(arg_policy)
        {
        }

        void execute() const
        {
            Kokkos::resilience::util::ResilientReplicateValidateFunctor<
                base_execution_space, FunctorType, validator_type>
                inst(m_functor, m_policy.space().validator(),
                    m_policy.space().replicates());

            // Call the underlying ParallelFor
            base_type closure(inst, m_policy);
            closure.execute();

            if (inst.is_incorrect())
                throw std::runtime_error(
                    "All replicate returned incorrect result.");
        }

    private:
        const FunctorType m_functor;
        const Policy m_policy;
    };

    template <typename FunctorType, typename... Traits>
    class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
        Kokkos::resilience::ResilientReplicate<typename Kokkos::resilience::
                traits::RangePolicyExtracter<Traits...>::base_execution_space>>
    {
    public:
        using Policy = Kokkos::RangePolicy<Traits...>;
        using BasePolicy =
            typename Kokkos::resilience::traits::RangePolicyExtracter<
                Traits...>::RangePolicy;
        using base_execution_space =
            typename Kokkos::resilience::traits::RangePolicyExtracter<
                Traits...>::base_execution_space;

        using base_type =
            ParallelFor<Kokkos::resilience::util::ResilientReplicateFunctor<
                            base_execution_space, FunctorType>,
                BasePolicy, base_execution_space>;

        ParallelFor(FunctorType const& arg_functor, Policy const& arg_policy)
          : m_functor(arg_functor)
          , m_policy(arg_policy)
        {
        }

        void execute() const
        {
            Kokkos::resilience::util::ResilientReplicateFunctor<
                base_execution_space, FunctorType>
                inst(m_functor);

            // Call the underlying ParallelFor
            base_type closure(inst, m_policy);
            closure.execute();

            if (inst.is_incorrect())
                throw std::runtime_error(
                    "All replicates returned different results.");
        }

    private:
        const FunctorType m_functor;
        const BasePolicy m_policy;
    };

    template <typename FunctorType, typename... Traits>
    class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
        Kokkos::resilience::ResilientReplicate<
            typename Kokkos::resilience::traits::MDRangePolicyExtracter<
                Traits...>::base_execution_space>>
    {
    public:
        using Policy = Kokkos::MDRangePolicy<Traits...>;
        using BasePolicy =
            typename Kokkos::resilience::traits::MDRangePolicyExtracter<
                Traits...>::MDRangePolicy;
        using base_execution_space =
            typename Kokkos::resilience::traits::MDRangePolicyExtracter<
                Traits...>::base_execution_space;

        using base_type =
            ParallelFor<Kokkos::resilience::util::ResilientReplicateFunctor<
                            base_execution_space, FunctorType>,
                BasePolicy, base_execution_space>;

        ParallelFor(FunctorType const& arg_functor, Policy const& arg_policy)
          : m_functor(arg_functor)
          , m_policy(base_execution_space{arg_policy.space()},
                arg_policy.m_lower, arg_policy.m_upper)
        {
        }

        void execute() const
        {
            Kokkos::resilience::util::ResilientReplicateFunctor<
                base_execution_space, FunctorType>
                inst(m_functor);

            // Call the underlying ParallelFor
            base_type closure(inst, m_policy);
            closure.execute();

            if (inst.is_incorrect())
                throw std::runtime_error(
                    "All replicates returned different results.");
        }

    private:
        const FunctorType m_functor;
        const BasePolicy m_policy;
    };

}}    // namespace Kokkos::Impl
