"""
ASTER: Attentional Scene Text Recognition
Rectification Network using Thin Plate Spline (TPS) Transformation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LocalizationNetwork(nn.Module):
    """
    Localization Network for TPS
    Predicts control points for TPS transformation
    Input: Image of shape (B, C, H, W)
    Output: Control points of shape (B, num_fiducial, 2)
    """
    
    def __init__(self, num_fiducial=20, img_height=32, img_width=100, num_channels=3):
        super(LocalizationNetwork, self).__init__()
        self.num_fiducial = num_fiducial
        self.img_height = img_height
        self.img_width = img_width
        
        # CNN layers for feature extraction
        self.conv_layers = nn.Sequential(
            # Block 1: (B, 3, 32, 100) -> (B, 64, 16, 50)
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: (B, 64, 16, 50) -> (B, 128, 8, 25)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: (B, 128, 8, 25) -> (B, 256, 4, 13)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 1)),
            
            # Block 4: (B, 256, 4, 13) -> (B, 512, 2, 7)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate the flattened feature size
        self.feature_size = 512 * 2 * 7  # After pooling
        
        # Fully connected layers to predict control points
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_fiducial * 2)  # 2 coordinates (x, y) per fiducial point
        )
        
        # Initialize with identity transformation
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to predict identity transformation"""
        self.fc[-1].weight.data.fill_(0)
        
        # Initialize with evenly spaced points along top and bottom
        num_points = self.num_fiducial // 2
        ctrl_pts_x = torch.linspace(-1, 1, num_points)
        ctrl_pts_y_top = torch.ones(num_points) * -1
        ctrl_pts_y_bottom = torch.ones(num_points) * 1
        
        bias = torch.cat([ctrl_pts_x, ctrl_pts_x])
        bias = torch.stack([bias, torch.cat([ctrl_pts_y_top, ctrl_pts_y_bottom])], dim=1)
        self.fc[-1].bias.data = bias.view(-1)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input image (B, C, H, W)
        Returns:
            ctrl_points: Control points (B, num_fiducial, 2) in normalized coordinates [-1, 1]
        """
        B = x.size(0)
        
        # Extract features
        features = self.conv_layers(x)
        features = features.view(B, -1)
        
        # Predict control points
        ctrl_points = self.fc(features)
        ctrl_points = ctrl_points.view(B, self.num_fiducial, 2)
        
        # Apply tanh to ensure points are in [-1, 1]
        ctrl_points = torch.tanh(ctrl_points)
        
        return ctrl_points


class GridGenerator(nn.Module):
    """
    Grid Generator for TPS
    Generates sampling grid for image transformation
    """
    
    def __init__(self, num_fiducial=20, img_height=32, img_width=100):
        super(GridGenerator, self).__init__()
        self.num_fiducial = num_fiducial
        self.img_height = img_height
        self.img_width = img_width
        
        # Create target grid (regular grid)
        self.target_height = img_height
        self.target_width = img_width
        
        # Create normalized target control points
        self.target_control_points = self._create_target_control_points()
        
    def _create_target_control_points(self):
        """Create target control points (regular grid)"""
        # Place control points along top and bottom edges
        num_points_per_edge = self.num_fiducial // 2
        
        # Top edge
        top_x = torch.linspace(-1, 1, num_points_per_edge)
        top_y = torch.ones(num_points_per_edge) * -1
        
        # Bottom edge
        bottom_x = torch.linspace(-1, 1, num_points_per_edge)
        bottom_y = torch.ones(num_points_per_edge) * 1
        
        target_ctrl_pts = torch.cat([
            torch.stack([top_x, top_y], dim=1),
            torch.stack([bottom_x, bottom_y], dim=1)
        ], dim=0)
        
        return target_ctrl_pts
        
    def _build_P_matrix(self, points):
        """
        Build P matrix for TPS transformation
        P = [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]
        """
        B, N, _ = points.shape
        x_coord = points[:, :, 0:1]
        y_coord = points[:, :, 1:2]
        
        ones = torch.ones_like(x_coord)
        P = torch.cat([
            ones,
            x_coord, y_coord,
            x_coord * x_coord, x_coord * y_coord, y_coord * y_coord,
            x_coord * x_coord * x_coord, x_coord * x_coord * y_coord, x_coord * y_coord * y_coord, y_coord * y_coord * y_coord
        ], dim=2)
        
        return P
        
    def _build_K_matrix(self, points1, points2):
        """
        Build K matrix for TPS transformation
        K[i,j] = ||points1[i] - points2[j]||^2 * log(||points1[i] - points2[j]||)
        """
        B, N1, _ = points1.shape
        _, N2, _ = points2.shape
        
        # Expand dimensions for broadcasting
        p1 = points1.unsqueeze(2)  # (B, N1, 1, 2)
        p2 = points2.unsqueeze(1)  # (B, 1, N2, 2)
        
        # Calculate squared distance
        diff = p1 - p2
        r2 = torch.sum(diff * diff, dim=3)  # (B, N1, N2)
        
        # Avoid log(0) by adding small epsilon
        r2 = torch.clamp(r2, min=1e-8)
        r = torch.sqrt(r2)
        
        K = r2 * torch.log(r)
        
        return K
        
    def forward(self, source_control_points):
        """
        Generate sampling grid
        Args:
            source_control_points: Source control points (B, num_fiducial, 2)
        Returns:
            grid: Sampling grid (B, H, W, 2) in normalized coordinates [-1, 1]
        """
        B = source_control_points.size(0)
        device = source_control_points.device
        
        # Move target control points to same device
        target_ctrl_pts = self.target_control_points.to(device)
        target_ctrl_pts = target_ctrl_pts.unsqueeze(0).expand(B, -1, -1)
        
        # Build P matrix for target points
        P = self._build_P_matrix(target_ctrl_pts)  # (B, num_fiducial, 10)
        
        # Build K matrix
        K = self._build_K_matrix(target_ctrl_pts, target_ctrl_pts)  # (B, num_fiducial, num_fiducial)
        
        # Build L matrix
        num_P = P.size(2)
        L_top = torch.cat([K, P], dim=2)  # (B, num_fiducial, num_fiducial + 10)
        L_bottom = torch.cat([
            P.transpose(1, 2),
            torch.zeros(B, num_P, num_P, device=device)
        ], dim=2)  # (B, 10, num_fiducial + 10)
        L = torch.cat([L_top, L_bottom], dim=1)  # (B, num_fiducial + 10, num_fiducial + 10)
        
        # Solve for TPS coefficients
        # Target = L * coeffs => coeffs = L^{-1} * target
        # Target is source_control_points with padding
        target = torch.cat([
            source_control_points,
            torch.zeros(B, num_P, 2, device=device)
        ], dim=1)  # (B, num_fiducial + 10, 2)
        
        # Solve linear system
        try:
            coeffs = torch.linalg.solve(L, target)  # (B, num_fiducial + 10, 2)
        except:
            # Fallback to least squares if singular
            coeffs = torch.linalg.lstsq(L, target).solution
        
        # Create dense grid of target points
        grid_y = torch.linspace(-1, 1, self.target_height, device=device)
        grid_x = torch.linspace(-1, 1, self.target_width, device=device)
        
        # Create meshgrid - compatible with different PyTorch versions
        grid_points_list = []
        for gy in grid_y:
            for gx in grid_x:
                grid_points_list.append(torch.tensor([gx, gy], device=device))
        grid_points = torch.stack(grid_points_list)  # (H*W, 2)
        grid_points = grid_points.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
        
        # Build P matrix for grid points
        P_grid = self._build_P_matrix(grid_points)  # (B, H*W, 10)
        
        # Build K matrix for grid points
        grid_points_expanded = grid_points.unsqueeze(2)  # (B, H*W, 1, 2)
        target_ctrl_pts_expanded = target_ctrl_pts.unsqueeze(1)  # (B, 1, num_fiducial, 2)
        
        diff = grid_points_expanded - target_ctrl_pts_expanded
        r2 = torch.sum(diff * diff, dim=3)  # (B, H*W, num_fiducial)
        r2 = torch.clamp(r2, min=1e-8)
        r = torch.sqrt(r2)
        K_grid = r2 * torch.log(r)  # (B, H*W, num_fiducial)
        
        # Combine for final transformation matrix
        phi = torch.cat([K_grid, P_grid], dim=2)  # (B, H*W, num_fiducial + 10)
        
        # Transform grid points: source = phi * coeffs
        source_grid = torch.bmm(phi, coeffs)  # (B, H*W, 2)
        source_grid = source_grid.view(B, self.target_height, self.target_width, 2)
        
        return source_grid


class TPSRectification(nn.Module):
    """
    Thin Plate Spline Rectification Module
    Rectifies distorted/curved text images into horizontal ones
    """
    
    def __init__(self, num_fiducial=20, img_height=32, img_width=100, num_channels=3):
        super(TPSRectification, self).__init__()
        
        self.localization = LocalizationNetwork(num_fiducial, img_height, img_width, num_channels)
        self.grid_gen = GridGenerator(num_fiducial, img_height, img_width)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input image (B, C, H, W) - can be distorted/curved text
        Returns:
            rectified: Rectified image (B, C, H, W) - horizontal text
            ctrl_points: Predicted control points (B, num_fiducial, 2)
        """
        # Predict control points
        ctrl_points = self.localization(x)
        
        # Generate sampling grid
        grid = self.grid_gen(ctrl_points)
        
        # Sample from input using grid
        rectified = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return rectified, ctrl_points


if __name__ == "__main__":
    # Test the rectification module
    print("Testing TPS Rectification Module...")
    
    # Create dummy input
    B, C, H, W = 2, 3, 32, 100
    x = torch.randn(B, C, H, W)
    
    # Create rectification module
    rectifier = TPSRectification(num_fiducial=20, img_height=H, img_width=W, num_channels=C)
    
    # Forward pass
    rectified, ctrl_points = rectifier(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Rectified shape: {rectified.shape}")
    print(f"Control points shape: {ctrl_points.shape}")
    print(f"Control points range: [{ctrl_points.min():.4f}, {ctrl_points.max():.4f}]")
    
    print("\nTPS Rectification Module test passed!")
