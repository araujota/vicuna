include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# llama-server executable

set(TARGET llama-server)

set(TARGET_SRCS
    server.cpp
    server-common.cpp
    server-common.h
    server-deepseek.cpp
    server-deepseek.h
    server-emotive-runtime.cpp
    server-emotive-runtime.h
    server-embedding-backend.cpp
    server-embedding-backend.h
    server-http.cpp
    server-http.h
    server-runtime.cpp
    server-runtime.h
    ../../common/log.cpp
)
set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE cpp-httplib llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_definitions(${TARGET} PRIVATE "VICUNA_SERVER_BUILD_INFO=\"b${LLAMA_BUILD_NUMBER}-${LLAMA_BUILD_COMMIT}\"")

target_compile_features(${TARGET} PRIVATE cxx_std_17)
