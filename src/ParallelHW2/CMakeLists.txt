cmake_minimum_required(VERSION 3.9)
project(TestProject)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_CXX_COMPILER mpicxx )
add_executable(TestProject serial.cpp common.cpp)
target_link_libraries(TestProject ${MPI_LIBRARIES})