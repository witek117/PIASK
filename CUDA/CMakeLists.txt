cmake_minimum_required(VERSION 3.16)

project(PIASK_CUDA CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 20)

add_executable(CUDA main.cu )

set_target_properties(
        CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

