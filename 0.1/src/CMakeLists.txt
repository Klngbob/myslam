add_library(myslam SHARED
    camera.cpp
    config.cpp
    frame.cpp
    map.cpp
    mappoint.cpp
)

target_link_libraries(myslam
    ${THIRD_PARTY_LIBS}
)