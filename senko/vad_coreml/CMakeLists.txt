# Make sure we're on macOS
if(NOT APPLE)
    return()
endif()

set(SWIFT_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/vad_lib.swift")
set(OUTPUT_LIB "${CMAKE_BINARY_DIR}/libvad_coreml.dylib")

add_custom_command(
    OUTPUT ${OUTPUT_LIB}
    COMMAND swiftc
        -emit-library
        -o ${OUTPUT_LIB}
        -O
        ${SWIFT_SOURCE}
    DEPENDS ${SWIFT_SOURCE}
    COMMENT "Compiling Swift library with swiftc"
    VERBATIM
)

# Create custom target that depends on the output
add_custom_target(libvad_coreml_target ALL
    DEPENDS ${OUTPUT_LIB}
)

# Only install in wheel builds (skip editable and sdist)
if(SKBUILD_STATE STREQUAL "wheel")
    install(FILES ${OUTPUT_LIB} DESTINATION senko)
endif()