SET(PROJECT_NAME DIFFUSION_PROBLEM)

#
# The version number.
#
SET(DIFFUSION_PROBLEM_VER_MAJOR 0)
SET(DIFFUSION_PROBLEM_VER_MINOR 1)


#
# Check for the existence of various optional folders:
#
if (EXISTS ${CMAKE_SOURCE_DIR}/doc/CMakeLists.txt)
  set (DIFFUSION_PROBLEM_HAVE_DOC_DIRECTORY TRUE)
endif ()

if (EXISTS ${CMAKE_SOURCE_DIR}/test/CMakeLists.txt)
  set (DIFFUSION_PROBLEM_HAVE_TEST_DIRECTORY TRUE)
endif ()


# Set the name of the project and target:
SET(TARGET "main")

SET(INCLUDE_DIR
		${CMAKE_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${INCLUDE_DIR})

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#  FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#  FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#  SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.

SET(SRC_DIR 
		${CMAKE_SOURCE_DIR}/source)

#SET(TARGET_SRC
#  ${SRC_DIR}/${TARGET}.cc
#  )
FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")


###############################################################################
###############################################################################
# Usually, you will not need to modify anything beyond this point...
###############################################################################
###############################################################################

MESSAGE(STATUS "This is CMake ${CMAKE_VERSION}")
MESSAGE(STATUS "")

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
FIND_PACKAGE(deal.II 9.4.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all dependencies fulfilled?
#
IF(NOT ((DEAL_II_WITH_PETSC AND NOT DEAL_II_PETSC_WITH_COMPLEX) OR DEAL_II_WITH_TRILINOS) OR NOT DEAL_II_WITH_P4EST) # keep in one line
  MESSAGE(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_PETSC = ON
    DEAL_II_PETSC_WITH_COMPLEX = OFF
    DEAL_II_WITH_P4EST = ON
or
    DEAL_II_WITH_TRILINOS = ON
    DEAL_II_WITH_P4EST = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options
    DEAL_II_WITH_PETSC = ${DEAL_II_WITH_PETSC}
    DEAL_II_PETSC_WITH_COMPLEX = ${DEAL_II_PETSC_WITH_COMPLEX}
    DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}
    DEAL_II_WITH_TRILINOS = ${DEAL_II_WITH_TRILINOS}
which conflict with the requirements.
One or both of the aforementioned combinations of prerequisites are not met by your installation, but at least one is required for this tutorial step."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )
  
  project(${PROJECT_NAME})
  ADD_EXECUTABLE(main  ${CMAKE_SOURCE_DIR}/source/DiffusionProblem.cc)

DEAL_II_SETUP_TARGET(main)


