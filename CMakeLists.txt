cmake_minimum_required(VERSION 3.20)
project(llm-bench-cli VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable aggressive compiler optimizations for benchmarking
if(NOT MSVC)
    add_compile_options(-O3 -march=native)
endif()

include(FetchContent)

# Fetch llama.cpp v1.0.0 directly from GitHub
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(llama)

add_executable(llm-bench
    src/main.cpp
    src/engine.cpp
)

target_link_libraries(llm-bench PRIVATE llama ggml)
target_include_directories(llm-bench PRIVATE src)

