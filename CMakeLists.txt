cmake_minimum_required(VERSION 3.10.0)

project(flutelearner C CXX)

set(CMAKE_CXX_STANDARD 17)

set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 2.4 COMPONENTS system window graphics audio main REQUIRED)

if (NOT SFML_FOUND)
	message(FATAL_ERROR "Could not find SFML. Please install SFML.")
endif()

add_subdirectory(genericneuralnetwork)

set(flutelearner_includes
    include/fft.hpp
)

add_executable(flutelearner main.cpp ${flutelearner_includes})

target_include_directories(flutelearner
    PUBLIC include
)

target_link_libraries(flutelearner
    PUBLIC neuralnetwork
	PUBLIC sfml-system
#	PUBLIC sfml-window
#	PUBLIC sfml-graphics
	PUBLIC sfml-audio
#	PUBLIC sfml-main
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT flutelearner)

# copy openal32.dll to executable directory on windows
if(WIN32)
	add_custom_command(TARGET flutelearner POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${SFML_DIR}/../../../bin/openal32.dll"
			"$<TARGET_FILE_DIR:flutelearner>"
	)
endif()
