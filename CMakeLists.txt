cmake_minimum_required(VERSION 3.18)
project(senko LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# libfbank_extractor
add_subdirectory(senko/fbank_extractor)

# libvad_coreml
if(APPLE)
    add_subdirectory(senko/vad_coreml)
endif()

# Only install in wheel builds (skip editable and sdist)
if(SKBUILD_STATE STREQUAL "wheel")
    install(DIRECTORY models DESTINATION senko)
endif()