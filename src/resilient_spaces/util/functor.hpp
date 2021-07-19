//  Copyright (c) 2021 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstdint>
#include <cstdlib>

namespace Kokkos { namespace resilience { namespace util {

    template <typename ExecutionSpace, typename Functor, typename Validator>
    class ResilientReplayValidateFunctor
    {
    public:
        KOKKOS_FUNCTION ResilientReplayValidateFunctor(
            Functor const& f, Validator const& v, std::uint64_t n)
          : functor(f)
          , validator(v)
          , replays(n)
          , incorrect_("result_correctness", 1)
        {
        }

        template <typename... ValueType>
        KOKKOS_FUNCTION void operator()(ValueType&&... i) const
        {
            for (std::uint64_t n = 0u; n != replays; ++n)
            {
                auto result = functor(i...);
                bool is_correct = validator(i..., result);

                if (is_correct)
                    break;

                if (n == replays - 1)
                    incorrect_[0] = true;
            }
        }

        bool is_incorrect() const
        {
            Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace>
                return_result("is_correct", 1);

            Kokkos::deep_copy(return_result, incorrect_);

            return return_result[0];
        }

    private:
        const Functor functor;
        const Validator validator;
        std::uint64_t replays;
        Kokkos::View<bool*, ExecutionSpace> incorrect_;
    };

    template <typename ExecutionSpace, typename Functor, typename Validator>
    class ResilientReplicateValidateFunctor
    {
    public:
        KOKKOS_FUNCTION ResilientReplicateValidateFunctor(
            Functor const& f, Validator const& v, std::uint64_t n)
          : functor(f)
          , validator(v)
          , replicates(n)
          , incorrect_("result_correctness", 1)
        {
        }

        template <typename... ValueType>
        KOKKOS_FUNCTION void operator()(ValueType&&... i) const
        {
            using return_type =
                typename std::invoke_result<Functor, ValueType...>::type;

            bool is_valid = false;
            return_type final_result{};

            for (std::uint64_t n = 0u; n != replicates; ++n)
            {
                auto result = functor(i...);
                bool is_correct = validator(i..., result);

                if (is_correct && !is_valid)
                {
                    final_result = result;
                    is_valid = true;
                }
            }

            if (!is_valid)
                incorrect_[0] = true;
        }

        bool is_incorrect() const
        {
            Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace>
                return_result("is_correct", 1);

            Kokkos::deep_copy(return_result, incorrect_);

            return return_result[0];
        }

    private:
        const Functor functor;
        const Validator validator;
        std::uint64_t replicates;
        Kokkos::View<bool*, ExecutionSpace> incorrect_;
    };

    template <typename ExecutionSpace, typename Functor>
    class ResilientReplicateFunctor
    {
    public:
        KOKKOS_FUNCTION ResilientReplicateFunctor(Functor const& f)
          : functor(f)
          , incorrect_("result_correctness", 1)
        {
        }

        template <typename... ValueType>
        KOKKOS_FUNCTION void operator()(ValueType... i) const
        {
            using return_type =
                typename std::invoke_result<Functor, ValueType...>::type;

            auto result_1 = functor(i...);
            auto result_2 = functor(i...);

            if (result_1 == result_2)
                return;

            return_type result_3 = functor(i...);

            if (result_1 == result_3)
                return;

            if (result_2 == result_3)
                return;

            incorrect_[0] = true;
        }

        bool is_incorrect() const
        {
            Kokkos::View<bool*, Kokkos::DefaultHostExecutionSpace>
                return_result("is_correct", 1);

            Kokkos::deep_copy(return_result, incorrect_);

            return return_result[0];
        }

    private:
        const Functor functor;
        Kokkos::View<bool*, ExecutionSpace> incorrect_;
    };

}}}    // namespace Kokkos::resilience::util
