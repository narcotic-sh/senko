cmake_minimum_required(VERSION 3.18)
project(senko LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Add the fbank_extractor subdirectory. This is where the C++ library is defined, built, and its installation is configured
add_subdirectory(senko/fbank_extractor)

# Only install in wheel builds (skip editable and sdist)
if(SKBUILD_STATE STREQUAL "wheel")
    install(TARGETS fbank_extractor DESTINATION senko)
    install(DIRECTORY models DESTINATION senko)
endif()