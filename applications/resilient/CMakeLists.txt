# Copyright (c) 2021 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_executable(stencil_resilient stencil.cpp heatdis.cpp)
target_link_libraries(stencil_resilient PUBLIC Kokkos::kokkos MPI::MPI_CXX Boost::program_options)

add_executable(ABFT3D_resilient ABFT3D.cpp)
target_link_libraries(ABFT3D_resilient PUBLIC Kokkos::kokkos Boost::program_options)
