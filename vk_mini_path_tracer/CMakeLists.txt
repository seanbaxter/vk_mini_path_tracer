# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 2.8)
set(PROJNAME "vk_mini_path_tracer")
project(${PROJNAME})

#####################################################################################
_add_project_definitions(${PROJNAME})

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)
set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS -shader)

#####################################################################################
# Executable
#
add_executable(${PROJNAME} ${SOURCE_FILES} ${GLSL_SOURCES})

#####################################################################################
# Source code group
#
source_group("Source Files" FILES ${SOURCE_FILES})

#####################################################################################
# Linkage
#
target_link_libraries(${PROJNAME} ${PLATFORM_LIBRARIES} shared_sources)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#
_copy_binaries_to_target( ${PROJNAME} )

install(FILES ${SPV_OUTPUT} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}/shaders")
install(FILES ${SPV_OUTPUT} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}/shaders")
install(DIRECTORY "../scenes" CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}")
install(DIRECTORY "../scenes" CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}")