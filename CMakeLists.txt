cmake_minimum_required(VERSION 3.10)

project(NeuralNetworkFromScratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -pedantic")
    message(STATUS "Debug mode - adding debug flags")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    message(STATUS "Release mode - adding optimization flags")
endif()

include_directories(include)

set(LIBRARY_SOURCES
    src/ActivationFunctions.cpp
    src/Layer.cpp
    src/LossFunction.cpp
    src/NeuralNetwork.cpp
    src/SGD.cpp
)

add_library(NeuralNetworkLib STATIC ${LIBRARY_SOURCES})

target_include_directories(NeuralNetworkLib 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()

file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)
    
    set(executable_name "run_${test_name}")
    
    add_executable(${executable_name} ${test_file})
    
    target_link_libraries(${executable_name} NeuralNetworkLib)
    
    add_test(NAME ${test_name} COMMAND ${executable_name})
    
    message(STATUS "Added test: ${test_name} -> executable: ${executable_name}")
endforeach()

add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS ${TEST_SOURCES}
    COMMENT "Running all tests"
)

message(STATUS "")
message(STATUS "==== Build Configuration Summary ====")
message(STATUS "Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Source directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "Binary directory: ${CMAKE_BINARY_DIR}")
message(STATUS "=====================================")
message(STATUS "")
