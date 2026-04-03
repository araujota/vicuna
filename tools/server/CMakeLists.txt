include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(TARGET llama-server)

set(TARGET_SRCS
    server.cpp
    server-common.cpp
    server-common.h
    server-cvec-generator.cpp
    server-cvec-generator.h
    server-decode-controller.cpp
    server-decode-controller.h
    server-deepseek.cpp
    server-deepseek.h
    server-emotive-runtime.cpp
    server-emotive-runtime.h
    server-http.cpp
    server-http.h
    server-local-llama-provider.cpp
    server-local-llama-provider.h
    server-runpod.cpp
    server-runpod.h
    server-runtime.cpp
    server-runtime.h
    server-runtime-control.cpp
    server-runtime-control.h
)

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE cpp-httplib Threads::Threads)
target_compile_definitions(${TARGET} PRIVATE "VICUNA_SERVER_BUILD_INFO=\"host-runtime\"")

if (OpenSSL_FOUND)
    target_link_libraries(${TARGET} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()
