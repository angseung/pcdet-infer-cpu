function(Format TARGET_DEPENDS SRC_DIR)
  find_program(CLANG_FORMAT_PATH clang-format)
  
  if(CLANG_FORMAT_PATH)
    set(EXPRESSION h hpp hh c cc cxx cpp cu)
    list(TRANSFORM EXPRESSION PREPEND "${SRC_DIR}/*.")
    file(GLOB_RECURSE SOURCE_FILES FOLLOW_SYMLINKS
      LIST_DIRECTORIES false ${EXPRESSION}
      )
    add_custom_command(TARGET ${TARGET_DEPENDS} PRE_BUILD
      WORKING_DIRECTORY ${SRC_DIR}
      COMMAND
      ${CLANG_FORMAT_PATH} -i ${SOURCE_FILES}
      )
  else()
    message(STATUS "clang-format not found: Code will not be automatically formatted.")
  endif()
endfunction()
