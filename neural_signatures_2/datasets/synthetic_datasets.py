from enum import Enum
from typing import Tuple, Optional
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
from scipy.spatial import Delaunay


class CoeffGenerationMethod(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"


class PolynomialSurfaceDataset(Dataset):
    def __init__(
            self,
            epoch_size: int,
            grid_points_count: int,
            order_range: Tuple[int, int],
            grid_radius_range: Tuple[float, float],
            grid_offset_range: Tuple[float, float],
            points_scale_range: Tuple[float, float],
            coefficient_scale_range: Tuple[float, float],
            coeff_generation_method: CoeffGenerationMethod,
            seed: int,
    ):
        super().__init__()
        self._epoch_size = epoch_size
        self._grid_points_count = grid_points_count
        self._order_range = order_range
        self._grid_radius_range = grid_radius_range
        self._grid_offset_range = grid_offset_range
        self._points_scale_range = points_scale_range
        self._coefficient_scale_range = coefficient_scale_range
        self._coeff_generation_method = coeff_generation_method
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def len(self):
        return self._epoch_size

    @staticmethod
    def _get_num_coeffs(order: int) -> int:
        """Calculate the number of coefficients for a given polynomial order."""
        return sum(1 for x in range(order + 1) for y in range(order + 1) if 0 < x + y <= order)

    def _generate_coeffs_and_order(self) -> Tuple[torch.Tensor, int, float]:
        """Generate random coefficients and order for a polynomial."""
        order = self._rng.integers(low=self._order_range[0], high=self._order_range[1] + 1)
        num_coeffs = self._get_num_coeffs(order=order)
        coefficient_scale = self._rng.uniform(low=self._coefficient_scale_range[0], high=self._coefficient_scale_range[1])

        if self._coeff_generation_method == CoeffGenerationMethod.UNIFORM:
            coeffs = torch.tensor(2 * (self._rng.uniform(size=num_coeffs) - 0.5) * coefficient_scale)
        elif self._coeff_generation_method == CoeffGenerationMethod.NORMAL:
            coeffs = torch.tensor(self._rng.normal(size=num_coeffs) * coefficient_scale)
        else:
            raise ValueError(f"Invalid coefficient generation method: {self._coeff_generation_method}")

        return coeffs, order, coefficient_scale

    def _evaluate_patch_points(self, x: torch.Tensor, y: torch.Tensor, data: Data) -> torch.Tensor:
        pairs = [(i, j) for i in range(data.order + 1) for j in range(data.order + 1) if 0 < i + j <= data.order]
        z = torch.zeros_like(x)
        for c, pair in zip(data.coeffs, pairs):
            z += c * (x ** pair[0]) * (y ** pair[1])
        return z

    def _compute_derivatives(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Compute first and second derivatives of z with respect to x and y."""
        dz_dx, dz_dy = torch.autograd.grad(outputs=z.sum(), inputs=[x, y], create_graph=True)
        d2z_dx2, d2z_dxdy = torch.autograd.grad(outputs=dz_dx.sum(), inputs=[x, y], create_graph=True)
        _, d2z_dy2 = torch.autograd.grad(outputs=dz_dy.sum(), inputs=[x, y], create_graph=True)
        return dz_dx, dz_dy, d2z_dx2, d2z_dxdy, d2z_dy2

    def _compute_shape_operator(
            self,
            dz_dx: torch.Tensor,
            dz_dy: torch.Tensor,
            d2z_dx2: torch.Tensor,
            d2z_dxdy: torch.Tensor,
            d2z_dy2: torch.Tensor
    ) -> torch.Tensor:
        """Compute the shape operator of the surface."""
        E = 1 + dz_dx ** 2
        F = dz_dx * dz_dy
        G = 1 + dz_dy ** 2
        L = d2z_dx2 / torch.sqrt(1 + dz_dx ** 2 + dz_dy ** 2)
        M = d2z_dxdy / torch.sqrt(1 + dz_dx ** 2 + dz_dy ** 2)
        N = d2z_dy2 / torch.sqrt(1 + dz_dx ** 2 + dz_dy ** 2)
        det = E * G - F ** 2
        shape_operator = torch.stack([
            torch.stack([G * L - F * M, G * M - F * N], dim=-1),
            torch.stack([E * M - F * L, E * N - F * M], dim=-1)
        ], dim=-2) / det.unsqueeze(-1).unsqueeze(-2)
        return shape_operator

    # def _compute_principal_curvatures(
    #         self,
    #         shape_operator: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Compute principal curvatures and directions from the shape operator."""
    #     eigenvalues, eigenvectors = torch.linalg.eig(shape_operator)
    #     k1, k2 = eigenvalues.real[..., 0], eigenvalues.real[..., 1]
    #     v1, v2 = eigenvectors.real[..., 0], eigenvectors.real[..., 1]
    #     return k1, k2, v1, v2

    def _compute_principal_curvatures(
            self,
            shape_operator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute principal curvatures and directions from the shape operator.
        Returns principal curvatures (|k1| >= |k2|) and their corresponding directions."""
        eigenvalues, eigenvectors = torch.linalg.eig(shape_operator)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        # Find indices where we need to swap to ensure descending order by absolute value
        # swap_mask = torch.abs(eigenvalues[..., 0]) < torch.abs(eigenvalues[..., 1])
        swap_mask = eigenvalues[..., 0] < eigenvalues[..., 1]

        # Create sorted eigenvalues
        k1 = torch.where(swap_mask, eigenvalues[..., 1], eigenvalues[..., 0])
        k2 = torch.where(swap_mask, eigenvalues[..., 0], eigenvalues[..., 1])

        # Apply same swapping to eigenvectors
        v1 = torch.where(swap_mask[..., None],
                         eigenvectors[..., 1],
                         eigenvectors[..., 0])
        v2 = torch.where(swap_mask[..., None],
                         eigenvectors[..., 0],
                         eigenvectors[..., 1])

        # # Ensure positive x-axis projection for both principal directions
        # flip_mask_v1 = v1[..., 0] < 0  # Check x-component
        # flip_mask_v2 = v2[..., 0] < 0
        #
        # v1 = torch.where(flip_mask_v1[..., None], -v1, v1)
        # v2 = torch.where(flip_mask_v2[..., None], -v2, v2)

        return k1, k2, v1, v2

    def _compute_curvatures(
            self,
            k1: torch.Tensor,
            k2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean (H) and Gaussian (K) curvatures from principal curvatures."""
        H = (k1 + k2) / 2
        K = k1 * k2
        return H, K

    def _compute_curvature_gradients(
            self,
            H: torch.Tensor,
            K: torch.Tensor,
            x: torch.Tensor,
            y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients of mean and Gaussian curvatures."""
        grad_H = torch.stack(torch.autograd.grad(outputs=H.sum(), inputs=[x, y], create_graph=True), dim=-1)
        grad_K = torch.stack(torch.autograd.grad(outputs=K.sum(), inputs=[x, y], create_graph=True), dim=-1)
        return grad_H, grad_K

    def _compute_jacobian(
            self,
            dz_dx: torch.Tensor,
            dz_dy: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Jacobian of the surface parameterization."""
        return torch.stack([
            torch.stack([torch.ones_like(dz_dx), torch.zeros_like(dz_dx)], dim=-1),
            torch.stack([torch.zeros_like(dz_dy), torch.ones_like(dz_dy)], dim=-1),
            torch.stack([dz_dx, dz_dy], dim=-1)
        ], dim=-2)

    def _map_to_3d(
            self,
            jacobian: torch.Tensor,
            v1: torch.Tensor,
            v2: torch.Tensor,
            grad_H: torch.Tensor,
            grad_K: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map 2D vector fields to 3D using the Jacobian."""
        v1_3d = torch.einsum('ijk,ik->ij', jacobian, v1)
        v2_3d = torch.einsum('ijk,ik->ij', jacobian, v2)
        grad_H_3d = torch.einsum('ijk,ik->ij', jacobian, grad_H)
        grad_K_3d = torch.einsum('ijk,ik->ij', jacobian, grad_K)
        return v1_3d, v2_3d, grad_H_3d, grad_K_3d

    def _generate_grid_points(self, grid_range: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a regular grid of points."""
        grid_points_count_sqrt = int(np.sqrt(self._grid_points_count))
        x_linspace = torch.linspace(grid_range[0], grid_range[1], grid_points_count_sqrt)
        y_linspace = torch.linspace(grid_range[0], grid_range[1], grid_points_count_sqrt)
        x, y = torch.meshgrid(x_linspace, y_linspace, indexing='ij')
        return x.flatten(), y.flatten()

    @staticmethod
    def _normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.norm(vectors, axis=-1, keepdims=True)
        return torch.where(norms > 0, vectors / norms, vectors)

    @staticmethod
    def _project_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        return torch.abs(torch.sum(v1 * v2, dim=1))

    def get(self, idx):
        # Generate random parameters
        coeffs, order, coefficient_scale = self._generate_coeffs_and_order()

        if len(self._grid_radius_range) == 2:
            grid_radius = float(self._rng.uniform(low=self._grid_radius_range[0], high=self._grid_radius_range[1]))
        else:
            grid_radius = self._grid_radius_range[0]

        if len(self._points_scale_range) == 2:
            points_scale = float(self._rng.uniform(low=self._points_scale_range[0], high=self._points_scale_range[1]))
        else:
            points_scale = self._points_scale_range[0]

        if len(self._grid_offset_range) == 2:
            grid_offset = float(self._rng.uniform(low=self._grid_offset_range[0], high=self._grid_offset_range[1]))
        else:
            grid_offset = self._grid_offset_range[0]

        # Create base data object
        data = Data(coeffs=coeffs, order=order, coefficient_scale=coefficient_scale,
                   grid_radius=grid_radius, grid_offset=grid_offset, points_scale=points_scale)

        # Generate grid points
        grid_range = (-data.grid_radius + data.grid_offset, data.grid_radius + data.grid_offset)
        x, y = self._generate_grid_points(grid_range=grid_range)
        x = x.requires_grad_()
        y = y.requires_grad_()

        # Evaluate surface and compute differential quantities
        z = self._evaluate_patch_points(x=x, y=y, data=data)
        dz_dx, dz_dy, d2z_dx2, d2z_dxdy, d2z_dy2 = self._compute_derivatives(x=x, y=y, z=z)
        shape_operator = self._compute_shape_operator(dz_dx=dz_dx, dz_dy=dz_dy, 
                                                    d2z_dx2=d2z_dx2, d2z_dxdy=d2z_dxdy, d2z_dy2=d2z_dy2)
        k1, k2, v1_2d, v2_2d = self._compute_principal_curvatures(shape_operator=shape_operator)
        H, K = self._compute_curvatures(k1=k1, k2=k2)
        grad_H_2d, grad_K_2d = self._compute_curvature_gradients(H=H, K=K, x=x, y=y)
        jacobian = self._compute_jacobian(dz_dx=dz_dx, dz_dy=dz_dy)
        v1_3d, v2_3d, grad_H_3d, grad_K_3d = self._map_to_3d(jacobian=jacobian, 
                                                             v1=v1_2d, v2=v2_2d,
                                                             grad_H=grad_H_2d, grad_K=grad_K_2d)

        v1_3d_normalized = PolynomialSurfaceDataset._normalize_vectors(v1_3d)
        v2_3d_normalized = PolynomialSurfaceDataset._normalize_vectors(v2_3d)

        H1 = PolynomialSurfaceDataset._project_vectors(v1=v1_3d_normalized, v2=grad_H_3d)
        H2 = PolynomialSurfaceDataset._project_vectors(v1=v2_3d_normalized, v2=grad_H_3d)

        K1 = PolynomialSurfaceDataset._project_vectors(v1=v1_3d_normalized, v2=grad_K_3d)
        K2 = PolynomialSurfaceDataset._project_vectors(v1=v2_3d_normalized, v2=grad_K_3d)

        # Store computed quantities
        data.pos = torch.stack([x, y, z], dim=1) * data.points_scale
        data.face = torch.from_numpy(Delaunay(data.pos[:, :2].detach().numpy()).simplices.T)
        data.H = H
        data.K = K
        data.H1 = H1
        data.K1 = K1
        data.H2 = H2
        data.K2 = K2
        data.grad_H_2d = grad_H_2d
        data.grad_K_2d = grad_K_2d
        data.grad_H_3d = grad_H_3d
        data.grad_K_3d = grad_K_3d
        data.k1 = k1
        data.k2 = k2
        data.v1_2d = v1_2d
        data.v2_2d = v2_2d
        data.v1_3d = v1_3d
        data.v2_3d = v2_3d
        data.jacobian = jacobian

        return data