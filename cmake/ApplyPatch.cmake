# Apply PATCH in the current directory unless it is already applied.
find_program(GIT git REQUIRED)
get_filename_component(_ceiling "${CMAKE_CURRENT_BINARY_DIR}/.." ABSOLUTE)
# Without a ceiling, `git apply` inside another repository's working tree silently skips every file.
set(_git ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${_ceiling} ${GIT} apply)
execute_process(COMMAND ${_git} --reverse --check "${PATCH}" RESULT_VARIABLE _applied OUTPUT_QUIET ERROR_QUIET)
if(_applied EQUAL 0)
    return()
endif()
execute_process(COMMAND ${_git} --whitespace=nowarn "${PATCH}" RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0 OR NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/src/llama-littlebit.cpp")
    message(FATAL_ERROR "failed to apply ${PATCH}")
endif()
