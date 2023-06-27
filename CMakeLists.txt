cmake_minimum_required(VERSION 3.20)
project(terraeroder C CXX)

find_package(CUDA)
if(CUDA_FOUND)
	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 14)
	set(USE_CUDA ON)
	set(CUDA_SEPARABLE_COMPILATION ON)
else()
	set(USE_CUDA OFF)
endif()

add_definitions(-D_USE_MATH_DEFINES)

set(THIRD_PARTY_DEPENDENCIES_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glad/include"
									"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw/include")
set(THIRD_PARTY_DEPENDENCIES_LIBRARIES glfw glad)

add_subdirectory(dependencies/glfw)
add_subdirectory(dependencies/glad)
add_subdirectory(src)

# file(COPY resources DESTINATION ${CMAKE_BINARY_DIR})
