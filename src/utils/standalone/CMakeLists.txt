find_package(Magnum)

set(standalone_SOURCES standalone.cpp)

#set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)
add_executable(standalone ${standalone_SOURCES})

target_include_directories(standalone
  PRIVATE
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
      ${CMAKE_CURRENT_LIST_DIR}/../../esp/gfx/cuda_helpers/
)

target_link_libraries(standalone
  PRIVATE
    assets
    gfx
    nav
    scene
    ${CUDART_LIBRARY}
)
