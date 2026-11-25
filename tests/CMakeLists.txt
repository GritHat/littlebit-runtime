set(_py core/test_q1.py)
if(LITTLEBIT_LLM)
    add_executable(test_layer llm/test_layer.cpp)
    target_include_directories(test_layer PRIVATE ${llama_cpp_SOURCE_DIR}/src ${llama_cpp_SOURCE_DIR}/include)
    target_link_libraries(test_layer PRIVATE llama ggml)

    enable_language(C)
    add_executable(test_api llm/test_api.c)
    target_link_libraries(test_api PRIVATE littlebit-static)
    list(APPEND _py llm/test_runtime.py)
endif()
if(LITTLEBIT_NCM)
    list(APPEND _py ncm/test_ncm.py)
endif()
list(TRANSFORM _py PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    # needs numpy, pyyaml, tqdm and requests; gguf-py is taken from the llama.cpp sources
    add_test(NAME python COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider ${_py})
    set_tests_properties(python PROPERTIES ENVIRONMENT
        "LITTLEBIT_BIN=$<TARGET_FILE_DIR:littlebit-cli>;LLAMA_CPP_DIR=${llama_cpp_SOURCE_DIR}")
endif()
