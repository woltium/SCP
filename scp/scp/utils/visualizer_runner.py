import os
import numpy as np
import open3d as o3d

def run_visualization(points_file, colors_file):
    points = np.load(points_file)
    colors = np.load(colors_file)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.visualization.draw_geometries([pcd])
def run_mesh_visualization(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    o3d.visualization.draw_geometries([mesh])
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualizer_runner.py points_file colors_file")
        print("   or: python visualizer_runner.py --mesh mesh_file")
        sys.exit(1)
        
    if sys.argv[1] == "--mesh":
        run_mesh_visualization(sys.argv[2])
    else:
        run_visualization(sys.argv[1], sys.argv[2])