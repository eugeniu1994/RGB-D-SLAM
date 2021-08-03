import g2o
import numpy as np
import cv2

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self, verbose=False):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        super().set_verbose(verbose)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, is_fixed=False):
        v_se2 = g2o.VertexSE3()
        v_se2.set_id(id)
        v_se2.set_estimate(g2o.Isometry3d(pose))
        v_se2.set_fixed(is_fixed)
        super().add_vertex(v_se2)

    def add_edge(self, vertices, measurement=None, information=np.eye(6), robust_kernel=None):
        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(g2o.Isometry3d(measurement))  # relative pose transformation between frames
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

