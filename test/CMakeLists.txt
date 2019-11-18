add_executable(vio_run vio_run.cpp)
target_link_libraries(vio_run vio)

add_executable(euroc2tum euroc2tum.cpp)
# target_link_libraries(euroc2tum ${THIRD_PARTY_LIBS})
