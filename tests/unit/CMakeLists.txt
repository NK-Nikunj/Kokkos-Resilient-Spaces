# Copyright (c) 2021 Nikunj Gupta

# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_custom_target(unit)

set(_tests
    resilient_spaces
)

foreach(_test ${_tests})
    set(_test_name ${_test}_test)
    add_executable(${_test_name} ${_test}.cpp)
    target_link_libraries(${_test_name} PUBLIC Kokkos::kokkos)
    add_dependencies(unit ${_test_name})
    add_test(NAME ${_test} COMMAND ${_test_name})
endforeach(_test ${_tests})
