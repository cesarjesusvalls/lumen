cmake_minimum_required(VERSION 3.16)
project(g4boundary CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Geant4 11.3 REQUIRED)

add_executable(g4boundary main.cc)
target_link_libraries(g4boundary PRIVATE ${Geant4_LIBRARIES})
