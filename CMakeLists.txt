cmake_minimum_required(VERSION 3.15)
project(terraeroder C CXX CUDA)

add_definitions(-D_USE_MATH_DEFINES)

set(THIRD_PARTY_DEPENDENCIES_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glad/include"
									"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw/include")
set(THIRD_PARTY_DEPENDENCIES_LIBRARIES glfw glad)

add_subdirectory(dependencies/glfw)
add_subdirectory(dependencies/glad)
add_subdirectory(src)

file(COPY resources DESTINATION ${CMAKE_BINARY_DIR})
