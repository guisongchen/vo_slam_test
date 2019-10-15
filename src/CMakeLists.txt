add_library(vo SHARED
    config.cpp
    camera.cpp
    ORBextractor.cpp
    frame.cpp
    mappoint.cpp
    keyframe.cpp
    map.cpp
    matcher.cpp
    optimizer_ceres.cpp
    visualOdometry.cpp
    drawer.cpp
    localMapping.cpp
    sim3Solver.cpp
    loopClosing.cpp)
    
target_link_libraries(vo
    ${THIRD_PARTY_LIBS})
