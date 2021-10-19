# Copyright (c) 2021 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.17)

project(ResilientSpaces CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

# Setting up dependencies
find_package(Kokkos REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

include_directories(src)

enable_testing()
add_subdirectory(tests)

option(KRS_WITH_APPLICATIONS "Enable building applications" OFF)
if(KRS_WITH_APPLICATIONS)
    find_package(MPI REQUIRED)
    find_package(Boost REQUIRED COMPONENTS program_options)

    add_subdirectory(applications)
endif()
