# Set output directory relative to build root
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math -funroll-loops")

# Global build settings
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Collect sources
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp")

# Build shared library
add_library(fbank_extractor MODULE ${SOURCES})

# Set proper suffix based on platform
if(APPLE)
    set_target_properties(fbank_extractor PROPERTIES SUFFIX ".dylib")
elseif(WIN32)
    set_target_properties(fbank_extractor PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(fbank_extractor PROPERTIES SUFFIX ".so")
endif()

target_include_directories(
    fbank_extractor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/feature
)

# Only install in wheel builds (skip editable and sdist)
if(SKBUILD_STATE STREQUAL "wheel")
    install(TARGETS fbank_extractor DESTINATION senko)
endif()