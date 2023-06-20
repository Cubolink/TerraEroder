set(THIRD_PARTY_VENDOR_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
set(THIRD_PARTY_VENDOR_LIBRARIES stb_image)
add_subdirectory(vendor/stb_image)

set(FIRST_PARTY_INCLUDE_DIRECTORIES renderer)
set(FIRST_PARTY_LIBRARIES renderer)
add_subdirectory(renderer)

add_executable(terra_eroder main.cpp geometry.h shape_factory.h shape_factory.cpp)

set_property(TARGET terra_eroder PROPERTY CXX_STANDARD 14)
target_include_directories(terra_eroder PRIVATE
        ${THIRD_PARTY_DEPENDENCIES_INCLUDE_DIRECTORIES}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${THIRD_PARTY_VENDOR_INCLUDE_DIRECTORIES}
        ${FIRST_PARTY_INCLUDE_DIRECTORIES}
        )
target_link_libraries(terra_eroder PRIVATE
        ${THIRD_PARTY_DEPENDENCIES_LIBRARIES}
        ${THIRD_PARTY_VENDOR_LIBRARIES}
        ${FIRST_PARTY_LIBRARIES}
        )
