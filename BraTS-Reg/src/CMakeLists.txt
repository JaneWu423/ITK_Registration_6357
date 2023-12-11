CMAKE_MINIMUM_REQUIRED (VERSION 3.22)
project (RegistrationITK)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable (RegistrationITK RegistrationITK1.cxx)

target_link_libraries(RegistrationITK ${ITK_LIBRARIES})
