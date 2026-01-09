function(setup_pyarrow_dirs)
    # This replicatees the logic of pyarrow.get_include_dir,
    # pyarrow.create_library_symlinks, and pyarrow.get_library_dirs
    # in CMAKE in order to support cross compilation. Otherwise those
    # functions would cause the arrow native extensions
    # to load at build time which is not possible in
    # cross compilation scenarios.

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_package(PkgConfig QUIET)

    # LOCATE PYARROW HOME
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c 
        "import importlib.util; print(importlib.util.find_spec('pyarrow').submodule_search_locations[0])"
        OUTPUT_VARIABLE PA_HOME
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(NOT PA_HOME)
        message(FATAL_ERROR "Could not find pyarrow module location via Python.")
    endif()

    set(PYARROW_HOME "${PA_HOME}" CACHE PATH "Path to PyArrow installation" FORCE)
    message(STATUS "PyArrow Location: ${PYARROW_HOME}")

    # CREATE SYMLINKS (Replica of create_library_symlinks)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        file(GLOB PYARROW_BUNDLED_LIBS "${PYARROW_HOME}/*.so.*")

        foreach(LIB_PATH ${PYARROW_BUNDLED_LIBS})
            get_filename_component(LIB_NAME "${LIB_PATH}" NAME)
            string(REGEX REPLACE "\\.[^.]+$" "" SYMLINK_NAME "${LIB_NAME}")
            set(FULL_SYMLINK_PATH "${PYARROW_HOME}/${SYMLINK_NAME}")

            if(NOT EXISTS "${FULL_SYMLINK_PATH}")
                message(STATUS "Creating PyArrow Symlink: ${SYMLINK_NAME} -> ${LIB_NAME}")
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E create_symlink "${LIB_NAME}" "${SYMLINK_NAME}"
                    WORKING_DIRECTORY "${PYARROW_HOME}"
                )
            endif()
        endforeach()
    elseif(APPLE)
        # macOS: libarrow.1700.dylib -> libarrow.dylib
        # Python pattern: glob.glob('*.*.dylib')
        file(GLOB PYARROW_BUNDLED_LIBS "${PYARROW_HOME}/*.*.dylib")

        foreach(LIB_PATH ${PYARROW_BUNDLED_LIBS})
            get_filename_component(LIB_NAME "${LIB_PATH}" NAME)

            # Python logic: return '.'.join((hard_path.rsplit('.', 2)[0], 'dylib'))
            # CMake Regex: Match ".VERSION.dylib" at the end and replace with ".dylib"
            string(REGEX REPLACE "\\.[^.]+\\.dylib$" ".dylib" SYMLINK_NAME "${LIB_NAME}")

            set(FULL_SYMLINK_PATH "${PYARROW_HOME}/${SYMLINK_NAME}")
            if(NOT EXISTS "${FULL_SYMLINK_PATH}")
                message(STATUS "Creating PyArrow Symlink: ${SYMLINK_NAME} -> ${LIB_NAME}")
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E create_symlink "${LIB_NAME}" "${SYMLINK_NAME}"
                    WORKING_DIRECTORY "${PYARROW_HOME}"
                )
            endif()
        endforeach()
    endif()

    # GET LIBRARY DIRS (Replica of get_library_dirs)
    set(TEMP_DIRS "${PYARROW_HOME}")

    if(DEFINED ENV{ARROW_HOME})
        list(APPEND TEMP_DIRS "$ENV{ARROW_HOME}/lib")
    endif()

    if(PKG_CONFIG_FOUND)
        foreach(PKG_NAME "arrow" "arrow_python")
            pkg_check_modules(PC_${PKG_NAME} QUIET ${PKG_NAME})
            if(PC_${PKG_NAME}_FOUND)
                list(APPEND TEMP_DIRS ${PC_${PKG_NAME}_LIBRARY_DIRS})
            endif()
        endforeach()
    endif()

    list(REMOVE_DUPLICATES TEMP_DIRS)

    # EXPOSE VARIABLES TO PARENT SCOPE
    list(GET TEMP_DIRS 0 PRIMARY_LIB_DIR)
    set(PYARROW_LIBRARY_DIRS "${PRIMARY_LIB_DIR}" PARENT_SCOPE)
    set(PYARROW_INCLUDE_DIR "${PYARROW_HOME}/include" PARENT_SCOPE)

endfunction()
