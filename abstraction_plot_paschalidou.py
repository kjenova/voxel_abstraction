import open3d as o3d

from paschalidou.inference import inference as paschalidou_inference

from loader.load_urocell import load_urocell_preprocessed

_, test = load_urocell_preprocessed("data/chamferData/urocell")

X = paschalidou_inference(test)

for i, points in enumerate(X):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("results/paschalidou/" + str(i + 1) + ".ply", pcd)
