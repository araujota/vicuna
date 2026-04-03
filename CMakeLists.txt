cmake_minimum_required(VERSION 3.20)
project(vicuna C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Threads REQUIRED)

option(VICUNA_OPENSSL "Enable HTTPS support through cpp-httplib" ON)
if (VICUNA_OPENSSL)
    find_package(OpenSSL)
endif()

add_library(cpp-httplib STATIC vendor/cpp-httplib/httplib.cpp)
target_include_directories(cpp-httplib PUBLIC ${CMAKE_SOURCE_DIR}/vendor)
if (OpenSSL_FOUND)
    target_compile_definitions(cpp-httplib PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(cpp-httplib PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()

if (APPLE)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
    find_library(SECURITY_FRAMEWORK Security REQUIRED)
    target_link_libraries(cpp-httplib PUBLIC ${COREFOUNDATION_FRAMEWORK} ${SECURITY_FRAMEWORK})
endif()

add_subdirectory(tools/server)
