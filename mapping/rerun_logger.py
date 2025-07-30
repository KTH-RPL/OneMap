"""
Mapping Rerun Logger. Sets up the experiment blueprint and logs the map and robot position.
"""
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from mapping import Navigator
from onemap_utils import log_map_rerun


def log_pos(x, y):
    rr.log("map/position", rr.Points2D([[x, y]], colors=[[255, 0, 0]], radii=[1]))


def setup_blueprint_debug():
    my_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="camera",
                                  name="rgb",
                                  contents=["$origin/rgb",
                                            "$origin/detection"], ),
                rrb.Spatial2DView(origin="camera/depth")
            ),
            rrb.Vertical(
                rrb.Vertical(
                    rrb.TextLogView(origin="object_detections"),
                    rrb.TextLogView(origin="path_updates"),
                ),
                rrb.Spatial2DView(origin="map",
                                  name="Traversable",
                                  contents=["$origin/traversable",
                                            "$origin/position"], ),
            ),
            rrb.Vertical(
                rrb.Tabs(
                    *[rrb.Spatial2DView(origin="map",
                                        name="Similarity",
                                        contents=
                                        ["$origin/similarity/",
                                         "$origin/proj_detect",
                                         "$origin/frontiers",
                                         "$origin/frontiers_far",
                                         "$origin/position"]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityTresholded",
                                        contents=
                                        ["$origin/similarity_th/",
                                         "$origin/proj_detect",
                                         "$origin/position"]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityTresholdedCl",
                                        contents=
                                        ["$origin/similarity_th2/",
                                         "$origin/proj_detect",
                                         "$origin/position"]),
                      ],
                ),
                rrb.Tabs(
                    rrb.Spatial2DView(origin="map",
                                      name="Explored",
                                      contents=["$origin/explored",
                                                "$origin/position",
                                                "$origin/proj_detect",
                                                "$origin/goal_pos",
                                                "$origin/largest_contour",
                                                "$origin/frontier_lines",
                                                "$origin/path",
                                                "$origin/path_simplified",
                                                "$origin/ground_truth",
                                                "$origin/frontiers",
                                                "$origin/frontiers_far", ]),
                    rrb.Spatial2DView(origin="map",
                                      name="Scores",
                                      contents=["$origin/scores",
                                                "$origin/position",
                                                "$origin/goal_pos",
                                                "$origin/path"]),
                    rrb.Spatial2DView(origin="map",
                                      name="Unexplored",
                                      contents=["$origin/frontiers",
                                                "$origin/frontiers_far",
                                                "$origin/largest_contour",
                                                "$origin/position",
                                                "$origin/unexplored"]),
                ),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(my_blueprint)
    rr.log("map/similarity", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity_th", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                             angle=rr.datatypes.Angle(
                                                                                 rad=-np.pi / 2))))
    rr.log("map/similarity_th2", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                              angle=rr.datatypes.Angle(
                                                                                  rad=-np.pi / 2))))
    rr.log("map/traversable", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/confidence", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/explored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/scores", rr.Transform3D(translation=np.array([0, 600, 0]),
                                        rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                      angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/unexplored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))

    # Point data
    rr.log("map/largest_contour", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                 rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                               angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontier_lines", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                              angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/path", rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                                       angle=rr.datatypes.Angle(
                                                                                                           rad=-np.pi))))
    rr.log("map/path_simplified",
           rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                           angle=rr.datatypes.Angle(
                                                                                               rad=-np.pi))))
    rr.log("map/position", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/proj_detect", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/goal_pos", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/ground_truth", rr.Transform3D(translation=np.array([0, 600, 0]),
                                              rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                            angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontiers", rr.Transform3D(translation=np.array([0, 600, 0]),
                                           rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                         angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontiers_far", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                             angle=rr.datatypes.Angle(rad=-np.pi))))

def setup_blueprint():
    my_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="camera",
                                  name="rgb",
                                  contents=["$origin/rgb",
                                            "$origin/detection"], ),
                rrb.Spatial2DView(origin="camera/depth")
            ),
            rrb.Vertical(
                rrb.Tabs(
                    *[rrb.Spatial2DView(origin="map",
                                        name="Similarity",
                                        contents=
                                        ["$origin/similarity/",
                                         "$origin/position"]),
                      ],
                ),
                rrb.Tabs(
                    rrb.Spatial2DView(origin="map",
                                      name="Explored",
                                      contents=["$origin/explored",
                                                "$origin/position",
                                                # "$origin/proj_detect",
                                                "$origin/goal_pos",
                                                # "$origin/largest_contour",
                                                # "$origin/frontier_lines",
                                                "$origin/path",
                                                "$origin/path_simplified",
                                                # "$origin/ground_truth",
                                                # "$origin/frontiers",
                                                # "$origin/frontiers_far",
                                                ]),
                    # rrb.Spatial2DView(origin="map",
                    #                   name="Scores",
                    #                   contents=["$origin/scores",
                    #                             "$origin/position",
                    #                             "$origin/goal_pos",
                    #                             "$origin/path"]),
                    # rrb.Spatial2DView(origin="map",
                    #                   name="Unexplored",
                    #                   contents=["$origin/frontiers",
                    #                             "$origin/frontiers_far",
                    #                             "$origin/largest_contour",
                    #                             "$origin/position",
                    #                             "$origin/unexplored"]),
                ),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(my_blueprint)
    rr.log("map/similarity", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity_th", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                             angle=rr.datatypes.Angle(
                                                                                 rad=-np.pi / 2))))
    rr.log("map/similarity_th2", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                              angle=rr.datatypes.Angle(
                                                                                  rad=-np.pi / 2))))
    rr.log("map/traversable", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/confidence", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/explored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/scores", rr.Transform3D(translation=np.array([0, 600, 0]),
                                        rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                      angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/unexplored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))

    # Point data
    rr.log("map/largest_contour", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                 rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                               angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontier_lines", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                              angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/path", rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                                       angle=rr.datatypes.Angle(
                                                                                                           rad=-np.pi))))
    rr.log("map/path_simplified",
           rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                           angle=rr.datatypes.Angle(
                                                                                               rad=-np.pi))))
    rr.log("map/position", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/proj_detect", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/goal_pos", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/ground_truth", rr.Transform3D(translation=np.array([0, 600, 0]),
                                              rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                            angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontiers", rr.Transform3D(translation=np.array([0, 600, 0]),
                                           rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                         angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontiers_far", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                             angle=rr.datatypes.Angle(rad=-np.pi))))


class RerunLogger:
    def __init__(self, mapper: Navigator, to_file: bool, save_path: str, debug: bool = True):
        self.debug_log = debug
        self.to_file = to_file
        self.mapper = mapper
        rr.init("MON", spawn=False)
        if self.to_file:
            rr.save(save_path)
        else:
            # NOTE Update to match newer rerun version
            rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy") # Connect to Rerun server
            print("Connected to Rerun server.")
        if self.debug_log:
            setup_blueprint_debug()
        else:
            setup_blueprint()
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def log_map(self):
        confidences = self.mapper.get_confidence_map()
        similarities = (self.mapper.get_map() + 1.0) / 2.0

        explored = (self.mapper.one_map.navigable_map == 1).astype(np.float32) * 0.1
        explored[confidences > 0] = 0.5
        explored[self.mapper.one_map.fully_explored_map] = 1.0
        explored[self.mapper.one_map.navigable_map == 0] = 0

        # frontiers = np.zeros((confidences.shape[0], confidences.shape[1]), dtype=np.float32)
        # for i, f in enumerate(self.mapper.frontiers):
        #     frontiers[f[:, 0, 0], f[:, 0, 1]] = 1
        # if (frontiers != 0).sum():
        #     log_map_rerun(frontiers, path="map/frontiers")

        # log_map_rerun(self.mapper.value_mapper.navigable_map, path="map/traversable")
        log_map_rerun(explored, path="map/explored")
        log_map_rerun(similarities[0], path="map/similarity")
        # log_map_rerun(confidences, path="map/confidence")

    def log_pos(self, x, y):
        px, py = self.mapper.one_map.metric_to_px(x, y)
        log_pos(px, py)
