add_library(vio SHARED
    estimator.cpp
    optimizer.cpp
    backend/problem.cpp
    backend/algorithm_LM.cpp
    backend/loss_function.cpp
    backend/edge_types.cpp
    backend/base_edge.cpp
    backend/vertex_types.cpp
    backend/base_vertex.cpp
    feature_tracker.cpp
    feature_manager.cpp
    initial/initial_alignment.cpp
    initial/initial_sfm.cpp
    camera_model/camera.cpp
    camera_model/camera_factory.cpp
    camera_model/pinhole_camera.cpp
    parameter.cpp
    system.cpp)

target_link_libraries(vio ${THIRD_PARTY_LIBS})
