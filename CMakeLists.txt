cmake_minimum_required(VERSION 3.1)
project(vo)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIR})

find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

find_package(DBoW3 REQUIRED)
include_directories(${DBoW3_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
    ${Pangolin_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${DBoW3_LIBS}
    ${CSPARSE_LIBRARY}
    ${CERES_LIBRARIES}
    ${OpenCV_LIBS})

include_directories(
${PROJECT_SOURCE_DIR}/include
)
add_subdirectory(src)
add_subdirectory(test)
