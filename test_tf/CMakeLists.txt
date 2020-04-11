cmake_minimum_required(VERSION 3.5)

project(hello_tf)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_VERBOSE_MAKEFILE ON)
    add_compile_options(-Wall -Wextra -pedantic-errors -Werror -Wno-sign-compare)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /WX)
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.11)
        add_compile_options(/permissive-)
    endif()
endif()

set(tensorflow_path /home/harry/tensorflow)

include_directories(${tensorflow_path})
include_directories(${tensorflow_path}/bazel-genfiles)
include_directories(${tensorflow_path}/bazel-bin/tensorflow)
include_directories(${tensorflow_path}/tensorfow/contrib/makefile/gen/proto)
include_directories(${tensorflow_path}/tensorflow/contrib/makefile/gen/protobuf-host/include)
include_directories(${tensorflow_path}/tensorflow/contrib/makefile/downloads/eigen)
include_directories(${tensorflow_path}/tensorflow/contrib/makefile/downloads/absl)
include_directories(${tensorflow_path}/tensorflow/contrib/makefile/downloads/nsync/public)

link_directories(${tensorflow_path}/bazel-bin/tensorflow)

include_directories(src/3rdparty/scope_guard/include)

add_executable(test_classifier src/test_classifier.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(test_classifier tensorflow_cc tensorflow_framework)

configure_file(models/frozen_model.pb ${CMAKE_CURRENT_BINARY_DIR}/frozen_model.pb COPYONLY)
