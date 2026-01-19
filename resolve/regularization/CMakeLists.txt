#[[

@brief Build Regularization Tools

@author Slaven Peles <peless@ornl.gov>
@author Jeffery Zhang <jefferyz@vt.edu>

]]

# C++ code
set(Regular_SRC 
    RegularizationSolver.cpp
)

# Header files to be installed
set(Regular_HEADER_INSTALL
    RegularizationSolver.hpp
)


# Build shared library ReSolve::regularization with Eigen dependency
add_library(resolve_regularization SHARED ${Regular_SRC})
target_link_libraries(resolve_regularization PRIVATE resolve_logger resolve_vector Eigen3::Eigen)



# Link to CUDA ReSolve backend if CUDA is enabled
if (RESOLVE_USE_CUDA)
  target_link_libraries(resolve_regularization PUBLIC resolve_backend_cuda)
endif()

# Link to HIP ReSolve backend if HIP is enabled
if (RESOLVE_USE_HIP)
  target_link_libraries(resolve_regularization PUBLIC resolve_backend_hip)
endif()

# Link to CPU backend if GPU support is not enabled
if (NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_regularization PUBLIC resolve_backend_cpu)
endif()

target_include_directories(resolve_regularization INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Install headers
install(FILES ${Regular_HEADER_INSTALL} DESTINATION include/resolve/regularization)
