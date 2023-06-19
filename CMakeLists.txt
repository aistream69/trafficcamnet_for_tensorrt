cmake_minimum_required(VERSION 3.5)

project (TRAFFICCAMNET)
set(PROJECT_ROOT_PATH "${TRAFFICCAMNET_SOURCE_DIR}")
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_INSTALL_PREFIX "${PROJECT_ROOT_PATH}/build/release")
set(TENSORRT_ROOT "/path_to/tensorrt")
set(CUDA_INSTALL_DIR "/usr/local/cuda")
set(TOOLS_ROOT "/path_to_opencv/tools")
set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream")

set(target test)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -O2 -rdynamic -Wno-deprecated-declarations")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
  ${TENSORRT_ROOT}/samples/common
  ${DEEPSTREAM_ROOT}/sources/includes
  ${DEEPSTREAM_ROOT}/sources/gst-plugins/gst-nvdspostprocess/postprocesslib_impl
  ${CUDA_INSTALL_DIR}/targets/aarch64-linux/include
  ${TOOLS_ROOT}/opencv-4.5.5/build/release/include/opencv4
)

link_directories(
  ${PROJECT_ROOT_PATH}/work/cjson/lib
  ${CUDA_INSTALL_DIR}/targets/aarch64-linux/lib
  ${TOOLS_ROOT}/opencv-4.5.5/build/release/lib
)

add_executable(
  ${target} 
  main.cpp
  ${TENSORRT_ROOT}/samples/common/logger.cpp
)

set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(${target}
  -lnvinfer
  -lcudart
  -lopencv_highgui
  -lopencv_core
  -lopencv_imgproc
  -lopencv_imgcodecs
  -lopencv_dnn
  -lpthread
  -fPIE
  -Wl,-rpath,lib
)

install (TARGETS ${target} DESTINATION .)

