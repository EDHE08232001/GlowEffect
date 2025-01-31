set(PROJECT_NAME movie_effect)

################################################################################
# Source groups
################################################################################
set(no_group_source_files
    "all_main.cpp"
)
source_group("" FILES ${no_group_source_files})

set(Include_Files
    "include/def_movies.h"
    "include/old_movies.cuh"
    "include/old_movies.hpp"
)
source_group("Include Files" FILES ${Include_Files})

set(Source_Files__cpp_code
    "source/test_learn.cpp"
)
source_group("Source Files\\cpp code" FILES ${Source_Files__cpp_code})

set(Source_Files__cref_code
    "source/color_polarizer.cpp"
    "source/dynamic_defocus.cpp"
    "source/gui_buttons.cpp"
    "source/gui_sliders.cpp"
    "source/old_movies.cpp"
    "source/old_movies_2.cpp"
)
source_group("Source Files\\cref code" FILES ${Source_Files__cref_code})

set(ALL_FILES
    ${no_group_source_files}
    ${Include_Files}
    ${Source_Files__cpp_code}
    ${Source_Files__cref_code}
)

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
################################################################################
# Includes for CMake from *.props
################################################################################
use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "$ENV{VCTargetsPath}/BuildCustomizations/CUDA 11.3.cmake")

set(ROOT_NAMESPACE movie_effect)

if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
    )
endif()
################################################################################
# MSVC runtime library
################################################################################
get_property(MSVC_RUNTIME_LIBRARY_DEFAULT TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY)
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    string(CONCAT "MSVC_RUNTIME_LIBRARY_STR"
        $<$<CONFIG:Debug>:
            MultiThreadedDebugDLL
        >
        $<$<CONFIG:Release>:
            MultiThreaded
        >
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${MSVC_RUNTIME_LIBRARY_DEFAULT}>
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    string(CONCAT "MSVC_RUNTIME_LIBRARY_STR"
        $<$<CONFIG:Debug>:
            MultiThreadedDebugDLL
        >
        $<$<CONFIG:Release>:
            MultiThreaded
        >
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${MSVC_RUNTIME_LIBRARY_DEFAULT}>
    )
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY ${MSVC_RUNTIME_LIBRARY_STR})

################################################################################
# Include directories
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../common/include;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../all_common/include;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../opencv4.5.5/INSTALL/include"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../common/include;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../all_common/include;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../opencv4.5.5/INSTALL/include"
    )
endif()

################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "WIN32;"
        "WIN64;"
        "_CRT_SECURE_NO_WARNINGS;"
        "_CONSOLE;"
        "_MBCS"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "WIN32;"
        "WIN64;"
        "_CRT_SECURE_NO_WARNINGS;"
        "_CONSOLE;"
        "_MBCS"
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /Od
            >
            $<$<CONFIG:Release>:
                /O2;
                /Oi;
                /Gy
            >
            /W3;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
            /openmp
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /Od
            >
            $<$<CONFIG:Release>:
                /O2;
                /Oi;
                /Gy
            >
            /W3;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
            /openmp
        )
    endif()
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /DEBUG;
                /INCREMENTAL
            >
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF
            >
            /SUBSYSTEM:CONSOLE
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /DEBUG;
                /INCREMENTAL
            >
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF
            >
            /SUBSYSTEM:CONSOLE
        )
    endif()
endif()

################################################################################
# Dependencies
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set(ADDITIONAL_LIBRARY_DEPENDENCIES
        "$<$<CONFIG:Debug>:"
            "opencv_core455d;"
            "opencv_highgui455d;"
            "opencv_imgcodecs455d;"
            "opencv_imgproc455d;"
            "opencv_videoio455d;"
            "opencv_video455d"
        ">"
        "$<$<CONFIG:Release>:"
            "opengl32;"
            "ittnotify;"
            "IlmImf;"
            "ippicvmt;"
            "ippiw;"
            "libjpeg-turbo;"
            "libopenjp2;"
            "libpng;"
            "libprotobuf;"
            "libtiff;"
            "libwebp;"
            "zlib;"
            "quirc;"
            "opencv_core455;"
            "opencv_highgui455;"
            "opencv_imgproc455;"
            "opencv_imgcodecs455;"
            "opencv_videoio455;"
            "opencv_video455"
        ">"
        "cudart_static;"
        "kernel32;"
        "user32;"
        "gdi32;"
        "winspool;"
        "comdlg32;"
        "advapi32;"
        "shell32;"
        "ole32;"
        "oleaut32;"
        "uuid;"
        "odbc32;"
        "odbccp32"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    set(ADDITIONAL_LIBRARY_DEPENDENCIES
        "$<$<CONFIG:Debug>:"
            "opencv_core455d;"
            "opencv_highgui455d;"
            "opencv_imgcodecs455d;"
            "opencv_imgproc455d;"
            "opencv_videoio455d;"
            "opencv_video455d"
        ">"
        "$<$<CONFIG:Release>:"
            "opengl32;"
            "ittnotify;"
            "IlmImf;"
            "ippicvmt;"
            "ippiw;"
            "libjpeg-turbo;"
            "libopenjp2;"
            "libpng;"
            "libprotobuf;"
            "libtiff;"
            "libwebp;"
            "zlib;"
            "quirc;"
            "opencv_core455;"
            "opencv_highgui455;"
            "opencv_imgproc455;"
            "opencv_imgcodecs455;"
            "opencv_videoio455;"
            "opencv_video455"
        ">"
        "cudart_static;"
        "kernel32;"
        "user32;"
        "gdi32;"
        "winspool;"
        "comdlg32;"
        "advapi32;"
        "shell32;"
        "ole32;"
        "oleaut32;"
        "uuid;"
        "odbc32;"
        "odbccp32"
    )
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE "${ADDITIONAL_LIBRARY_DEPENDENCIES}")

if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_link_directories(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../../opencv4.5.5/install/x64/vc16/lib"
        ">"
        "$<$<CONFIG:Release>:"
            "E:/opencv4.5.5/INSTALL/x64/vc16/staticlib"
        ">"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    target_link_directories(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../../opencv4.5.5/install/x64/vc16/lib"
        ">"
        "$<$<CONFIG:Release>:"
            "E:/opencv4.5.5/INSTALL/x64/vc16/staticlib"
        ">"
    )
endif()

