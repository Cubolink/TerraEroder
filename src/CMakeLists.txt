set(THIRD_PARTY_VENDOR_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
set(THIRD_PARTY_VENDOR_LIBRARIES stb_image imgui)
add_subdirectory(vendor/stb_image)
add_subdirectory(vendor/imgui)

set(FIRST_PARTY_INCLUDE_DIRECTORIES renderer)
set(FIRST_PARTY_LIBRARIES renderer)
add_subdirectory(renderer)

add_executable(terra_eroder main.cpp
        geometry.h
        obj_files.h obj_files.cpp
        camera.h camera.cpp
        controller.h controller.cpp
        display_controller.h display_controller.cpp
        color.h color.cpp
        shape_factory.h shape_factory.cpp)

set_target_properties(terra_eroder PROPERTIES
        CXX_STANDARD 14
        # CUDA_ARCHITECTURES 30 61 75
        )
set_property(TARGET terra_eroder PROPERTY CXX_STANDARD 14)
target_include_directories(terra_eroder PRIVATE
        ${THIRD_PARTY_DEPENDENCIES_INCLUDE_DIRECTORIES}
        # ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${THIRD_PARTY_VENDOR_INCLUDE_DIRECTORIES}
        ${FIRST_PARTY_INCLUDE_DIRECTORIES}
        )
target_link_libraries(terra_eroder PRIVATE
        ${THIRD_PARTY_DEPENDENCIES_LIBRARIES}
        ${THIRD_PARTY_VENDOR_LIBRARIES}
        ${FIRST_PARTY_LIBRARIES}
        )

add_custom_command(
        TARGET terra_eroder PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/resources ${CMAKE_BINARY_DIR}/resources
        COMMENT "Copying resources into the build folder"
)