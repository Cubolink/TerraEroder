
add_library(renderer

        renderer.h
        vertexarray.h
        vertexbufferlayout.h
        vertexbuffer.h
        indexbuffer.h
        material.h
        light.h
        shader.h
        texture.h
        shape.h
        glerrorhandler.h

        renderer.cpp
        vertexarray.cpp
        vertexbufferlayout.cpp
        vertexbuffer.cpp
        indexbuffer.cpp
        material.cpp
        light.cpp
        shader.cpp
        texture.cpp
        shape.cpp
        glerrorhandler.cpp
)

if (USE_CUDA)
    message(STATUS "Enabled CUDA support in the Renderer library")
    message(STATUS "CUDA-compatible facilities: CudaShape.h, CudaShape.cpp")
    target_sources(renderer
            PUBLIC
                CudaShape.h
            PRIVATE
                CudaShape.cpp)
endif()

set_property(TARGET renderer PROPERTY CXX_STANDARD 14)

target_link_libraries(renderer PRIVATE ${THIRD_PARTY_DEPENDENCIES_LIBRARIES} ${THIRD_PARTY_VENDOR_LIBRARIES})
target_include_directories(renderer PRIVATE
        ${THIRD_PARTY_DEPENDENCIES_INCLUDE_DIRECTORIES}
        ${THIRD_PARTY_VENDOR_INCLUDE_DIRECTORIES})
if (USE_CUDA)
target_include_directories(renderer PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
endif()
