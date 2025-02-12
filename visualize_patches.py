import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import polyscope as ps
import numpy as np
# from typing import Optional
# import torch
# from torch_geometric.loader import DataLoader
# from neural_signatures.datasets.synthetic_datasets import CoeffGenerationMethod
# from sklearn.decomposition import PCA
# import open3d as o3d
from omegaconf import OmegaConf


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.where(norms > 0, vectors / norms, vectors)


def visualize_patch(points: np.ndarray, faces: np.ndarray, name: str, enabled: bool) -> ps.SurfaceMesh:
    mesh = ps.register_surface_mesh(
        name=name,
        vertices=points,
        faces=faces,
        smooth_shade=True,
        edge_width=0.0,
        enabled=enabled
    )

    return mesh


def visualize_point_cloud(points: np.ndarray, name: str, radius: float = 0.005, enabled: bool = False) -> ps.PointCloud:
    cloud = ps.register_point_cloud(
        name=name,
        points=points,
        radius=radius,
        enabled=enabled
    )

    return cloud


def compute_diameter(points: np.ndarray) -> float:
    """
    Compute the diameter of a 3D point cloud.
    The diameter is the maximum distance between any two points.

    Parameters:
    points (np.ndarray): Array of shape (K,3) containing K 3D points

    Returns:
    float: Diameter of the point cloud
    """
    # Compute pairwise distances between all points
    # Using broadcasting to create difference vectors
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]

    # Compute Euclidean distances
    # This creates a KxK matrix where entry (i,j) is the distance between points i and j
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    # Find the maximum distance
    diameter = np.max(distances)

    return diameter


def visualize_coordinate_axes(scale: float = 0.2) -> None:
    """
    Add coordinate axes to the scene.

    Parameters:
    scale (float): Length of the coordinate axes
    """
    # Create origin point
    origin = np.array([[0.0, 0.0, 0.0]])

    # Create axes directions
    directions = np.array([
        [scale, 0.0, 0.0],  # X-axis
        [0.0, scale, 0.0],  # Y-axis
        [0.0, 0.0, scale]  # Z-axis
    ])

    # Register a point cloud for the origin
    axes_cloud = ps.register_point_cloud(
        "coordinate_axes",
        origin,
        enabled=True,
        radius=0.002
    )

    # Add vectors for each axis with different colors
    axes_cloud.add_vector_quantity(
        "x_axis",
        np.array([[1.0, 0.0, 0.0]]) * scale,
        enabled=True,
        color=(1.0, 0.0, 0.0),  # Red for X
        vectortype="ambient"
    )

    axes_cloud.add_vector_quantity(
        "y_axis",
        np.array([[0.0, 1.0, 0.0]]) * scale,
        enabled=True,
        color=(0.0, 1.0, 0.0),  # Green for Y
        vectortype="ambient"
    )

    axes_cloud.add_vector_quantity(
        "z_axis",
        np.array([[0.0, 0.0, 1.0]]) * scale,
        enabled=True,
        color=(0.0, 0.0, 1.0),  # Blue for Z
        vectortype="ambient"
    )


@hydra.main(version_base="1.2", config_path="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.globals.seed)

    # Instantiate dataset and data module
    data_module = hydra.utils.instantiate(cfg.data_module)

    # Get the data loader
    data_loader = data_module.train_dataloader()

    # Initialize polyscope
    ps.init()

    # Set the up direction to Z-axis
    ps.set_up_dir("z_up")
    ps.look_at(camera_location=[1, 1, 1], target=[0, 0, 0])
    ps.set_ground_plane_mode("none")

    for batch in data_loader:
        patch = batch[0]
        name = 'Patch'
        pos = patch.pos.detach().cpu().numpy()
        face = patch.face.detach().cpu().numpy().T
        # normals = patch.normals.detach().cpu().numpy()
        translation = np.array([0, 0, 0])
        pos = pos + translation

        # Display full patch as surface mesh
        mesh = visualize_patch(
            points=pos,
            faces=face,
            name=f"{name} - Mesh",
            enabled=True
        )

        # Display full patch as point cloud
        cloud = visualize_point_cloud(
            points=pos,
            name=f"{name} - Point Cloud",
            radius=0.005,
            enabled=False
        )

        # cloud.add_vector_quantity("normals", normals * 0.1, enabled=True, color=(0.0, 1.0, 1.0), vectortype="ambient")

        H = patch.H.detach().cpu().numpy()
        K = patch.K.detach().cpu().numpy()
        k1 = patch.k1.detach().cpu().numpy()
        k2 = patch.k2.detach().cpu().numpy()
        v1_3d = patch.v1_3d.detach().cpu().numpy()
        v2_3d = patch.v2_3d.detach().cpu().numpy()
        grad_H_3d = patch.grad_H_3d.detach().cpu().numpy()
        grad_K_3d = patch.grad_K_3d.detach().cpu().numpy()
        v1_2d = patch.v1_2d.detach().cpu().numpy()
        v2_2d = patch.v2_2d.detach().cpu().numpy()
        grad_H_2d = patch.grad_H_2d.detach().cpu().numpy()
        grad_K_2d = patch.grad_K_2d.detach().cpu().numpy()

        # Scale factor for vector fields
        scale = 0.02  # Adjust this value to change vector length

        # Normalize and scale vector fields
        v1_3d_norm = normalize_vectors(v1_3d) * scale
        v2_3d_norm = normalize_vectors(v2_3d) * scale
        grad_H_3d_norm = normalize_vectors(grad_H_3d) * scale
        grad_K_3d_norm = normalize_vectors(grad_K_3d) * scale

        # Compute 3D inner products
        v1_dot_gradK_3d = np.abs(np.sum(normalize_vectors(v1_3d) * grad_K_3d, axis=1))  # Element-wise dot product

        # Add vector quantities to the full patch mesh and point cloud
        for structure in [mesh, cloud]:
            structure.add_vector_quantity("v1", v1_3d_norm, enabled=False, color=(1.0, 0.0, 0.0), vectortype="ambient")
            structure.add_vector_quantity("v2", v2_3d_norm, enabled=False, color=(0.0, 0.0, 1.0), vectortype="ambient")
            structure.add_vector_quantity("grad_H", grad_H_3d_norm, enabled=False, color=(0.0, 1.0, 0.0), vectortype="ambient")
            structure.add_vector_quantity("grad_K", grad_K_3d_norm, enabled=False, color=(1.0, 1.0, 0.0), vectortype="ambient")
            structure.add_scalar_quantity("Mean Curvature", H, enabled=False, cmap='coolwarm')
            structure.add_scalar_quantity("Gaussian Curvature", K, enabled=False, cmap='coolwarm')
            structure.add_scalar_quantity("k1", k1, enabled=False, cmap='coolwarm')
            structure.add_scalar_quantity("k2", k2, enabled=False, cmap='coolwarm')
            structure.add_scalar_quantity("dot(v1, grad_K) (3D)", v1_dot_gradK_3d, enabled=True, cmap='coolwarm')

        # Visualize 2D parametrization of the full patch as a point cloud
        param_points = pos.copy()
        param_points[:, 2] = 0  # Set Z coordinate to 0
        param_cloud = visualize_point_cloud(
            param_points,
            f"{name} - Domain",
            radius=0.001,  # Smaller spheres for 2D parametrization
            enabled=False
        )

        # Normalize and scale 2D vector fields
        v1_2d_norm = normalize_vectors(v1_2d) * scale
        v2_2d_norm = normalize_vectors(v2_2d) * scale
        grad_H_2d_norm = normalize_vectors(grad_H_2d) * scale
        grad_K_2d_norm = normalize_vectors(grad_K_2d) * scale

        # Add 2D vector quantities to the parametrization point cloud
        param_cloud.add_vector_quantity("v1_2d", v1_2d_norm, enabled=True, color=(1.0, 0.0, 0.0), vectortype="ambient")
        param_cloud.add_vector_quantity("v2_2d", v2_2d_norm, enabled=False, color=(0.0, 0.0, 1.0), vectortype="ambient")
        param_cloud.add_vector_quantity("grad_H_2d", grad_H_2d_norm, enabled=False, color=(0.0, 1.0, 0.0), vectortype="ambient")
        param_cloud.add_vector_quantity("grad_K_2d", grad_K_2d_norm, enabled=True, color=(1.0, 1.0, 0.0), vectortype="ambient")

        param_cloud.add_scalar_quantity("Mean Curvature", H, enabled=False, cmap='coolwarm')
        param_cloud.add_scalar_quantity("Gaussian Curvature", K, enabled=True, cmap='coolwarm')
        param_cloud.add_scalar_quantity("k1", k1, enabled=False, cmap='coolwarm')
        param_cloud.add_scalar_quantity("k2", k2, enabled=False, cmap='coolwarm')

        param_cloud.add_scalar_quantity("dot(v1, grad_K) (2D)", v1_dot_gradK_3d, enabled=False, cmap='coolwarm')

        # Add coordinate axes to the scene
        visualize_coordinate_axes(scale=0.2)  # Adjust scale as needed

        # Show the polyscope GUI
        ps.show()

        # Clear all structures for the next batch
        ps.remove_all_structures()

if __name__ == "__main__":
    main()