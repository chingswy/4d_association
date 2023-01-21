cmake_minimum_required(VERSION 3.10.0)
project(TotalCapture VERSION 0.1.0)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++17 -pthread -O3")
add_definitions(-DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

find_package(OpenCV REQUIRED)
set(Eigen3_DIR "${PROJECT_SOURCE_DIR}/3rdparty/eigen-3.3.7/share/eigen3/cmake")
find_package(Eigen3 3.3.7 REQUIRED)

add_subdirectory("${PROJECT_SOURCE_DIR}/3rdparty/jsoncpp-1.9.3")
# set(jsoncpp_DIR "${PROJECT_SOURCE_DIR}/3rdparty/jsoncpp-install/lib/cmake/jsoncpp")
# find_package(jsoncpp 1.9.3 REQUIRED)

message(STATUS ${OpenCV_INCLUDE_DIRS})

file(GLOB 4D_FILES "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(association4d SHARED ${4D_FILES})
set_property(TARGET association4d PROPERTY CXX_STANDARD 17)
target_include_directories(association4d 
	    PUBLIC 
	            ${EIGEN3_INCLUDE_DIR}
		            ${OpenCV_INCLUDE_DIRS}
			    )
		    target_link_libraries(association4d jsoncpp_lib stdc++fs)

		    add_executable(mocap "${PROJECT_SOURCE_DIR}/mocap/main.cpp")
		    target_include_directories(mocap PRIVATE "${PROJECT_SOURCE_DIR}/src")
		    target_link_libraries(mocap 
			        association4d
				    ${OpenCV_LIBRARIES}
				    )
			    set_property(TARGET mocap PROPERTY CXX_STANDARD 17)

			    add_executable(evaluate "${PROJECT_SOURCE_DIR}/evaluate_shelf/main.cpp")
			    target_include_directories(evaluate PRIVATE "${PROJECT_SOURCE_DIR}/src")
			    target_link_libraries(evaluate 
				        association4d
					    ${OpenCV_LIBRARIES}
					    )
				    set_property(TARGET evaluate PROPERTY CXX_STANDARD 17)
