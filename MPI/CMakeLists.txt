cmake_minimum_required(VERSION 3.1)

project(MPI_PIASK CXX C)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_C_STANDARD 17)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(MPI_PIASK main.cpp my_timers.c)
target_link_libraries(MPI_PIASK ${MPI_LIBRARIES})
target_include_directories(MPI_PIASK PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall ")
