function(_yamlcpp_append_debugs _endvar _library)
    if(${_library} AND ${_library}_DEBUG)
        set(_output optimized ${${_library}} debug ${${_library}_DEBUG})
    else()
        set(_output ${${_library}})
    endif()
    set(${_endvar} ${_output} PARENT_SCOPE)
endfunction()

function(_yamlcpp_find_library _name)
    find_library(${_name}
        NAMES ${ARGN}
        HINTS
            $ENV{YAMLCPP_ROOT}
            ${YAMLCPP_ROOT}
        PATH_SUFFIXES ${_yamlcpp_libpath_suffixes}
    )
    mark_as_advanced(${_name})
endfunction()

#

if(NOT DEFINED YAMLCPP_MSVC_SEARCH)
    set(YAMLCPP_MSVC_SEARCH MD)
endif()

set(_yamlcpp_libpath_suffixes lib)
if(MSVC)
    if(YAMLCPP_MSVC_SEARCH STREQUAL "MD")
        list(APPEND _yamlcpp_libpath_suffixes
            msvc/yamlcpp-md/Debug
            msvc/yamlcpp-md/Release)
    elseif(YAMLCPP_MSVC_SEARCH STREQUAL "MT")
        list(APPEND _yamlcpp_libpath_suffixes
            msvc/yamlcpp/Debug
            msvc/yamlcpp/Release)
    endif()
endif()


find_path(YAMLCPP_INCLUDE_DIR yaml-cpp/yaml.h
    HINTS
        $ENV{YAMLCPP_ROOT}/include
        ${YAMLCPP_ROOT}/include
)
mark_as_advanced(YAMLCPP_INCLUDE_DIR)

if(MSVC AND YAMLCPP_MSVC_SEARCH STREQUAL "MD")
    # The provided /MD project files for Google Test add -md suffixes to the
    # library names.
    _yamlcpp_find_library(YAMLCPP_LIBRARY            yaml-cpp-md  yaml-cpp)
    _yamlcpp_find_library(YAMLCPP_LIBRARY_DEBUG      yaml-cpp-mdd yaml-cppd)
else()
    _yamlcpp_find_library(YAMLCPP_LIBRARY            yaml-cpp)
    _yamlcpp_find_library(YAMLCPP_LIBRARY_DEBUG      yaml-cppd)
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(YAMLCPP DEFAULT_MSG YAMLCPP_LIBRARY YAMLCPP_INCLUDE_DIR)

if(YAMLCPP_FOUND)
    set(YAMLCPP_INCLUDE_DIRS ${YAMLCPP_INCLUDE_DIR})
    _yamlcpp_append_debugs(YAMLCPP_LIBRARIES      YAMLCPP_LIBRARY)
    #set(YAMLCPP_BOTH_LIBRARIES ${YAMLCPP_LIBRARIES})
endif()

