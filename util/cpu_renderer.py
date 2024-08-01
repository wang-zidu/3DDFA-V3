from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .cython_renderer import mesh_core_cython
import torch
import torch.nn as nn
import numpy as np


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0, n/-x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

class MeshRenderer_cpu(nn.Module):
    def __init__(self,
                rasterize_fov,
                znear=0.1,
                zfar=10, 
                rasterize_size=224):
        super(MeshRenderer_cpu, self).__init__()

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)).matmul(
                torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size

    def forward(self, vertex, tri, feat = None, visible_vertice = False):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, N, C), features
        """
        assert vertex.device == torch.device('cpu')
        assert tri.device == torch.device('cpu')
        assert vertex.shape[0] == 1 # only support batchsize = 1

        device = vertex.device
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 

        vertex_ndc = vertex @ ndc_proj.t()
        vertex_ndc = vertex_ndc[..., :] / vertex_ndc[..., 3:]

        # initial
        c = 3
        h = self.rasterize_size
        w = self.rasterize_size
        vertices = vertex_ndc[0,:,:3].detach().contiguous().numpy().astype(np.float32).copy()[:,:3]

        vertices[:,0] = vertices[:,0] * w/2
        vertices[:,1] = vertices[:,1] * h/2
        vertices[:,0] = vertices[:,0] + w/2
        vertices[:,1] = vertices[:,1] + h/2

        vertices[:,2] = - vertex[..., 2].detach()

        triangles = tri.numpy().astype(np.int32).copy()
        if feat is not None:
            colors = feat.detach().numpy()[0].astype(np.float32).copy()
        else:
            colors = np.zeros_like(vertices)

        depth_buffer = np.zeros([h, w], dtype = np.float32) - 999999. #set the initial z to the farest position
        triangle_buffer = np.zeros([h, w], dtype = np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
        barycentric_weight = np.zeros([h, w, 3], dtype = np.float32)  # 
        image = np.zeros((h, w, c), dtype = np.float32)

        mesh_core_cython.MeshRenderer_cpu_core(
                image, vertices, triangles,
                colors,
                depth_buffer, triangle_buffer, barycentric_weight, 
                vertices.shape[0], triangles.shape[0], 
                h, w, c)
        depth_buffer[depth_buffer == - 999999] = 0
        depth_buffer = - depth_buffer[np.newaxis, np.newaxis, :]

        unique_visible_verts_idx = None
        if visible_vertice:
            visible_faces = np.unique(triangle_buffer)
            visible_faces = visible_faces[visible_faces!=-1]
            visible_verts_idx = tri[visible_faces]
            unique_visible_verts_idx = np.unique(visible_verts_idx).astype(np.int32)
            unique_visible_verts_idx = torch.from_numpy(unique_visible_verts_idx)

        mask =  (triangle_buffer > 0).astype(np.float32)[np.newaxis, np.newaxis, :]
        image = image.transpose(2,0,1)[np.newaxis, :]
        return torch.from_numpy(mask), torch.from_numpy(depth_buffer), torch.from_numpy(image), unique_visible_verts_idx

class MeshRenderer_UV_cpu(nn.Module):
    def __init__(self,
                rasterize_size=224):
        super(MeshRenderer_UV_cpu, self).__init__()

        self.rasterize_size = rasterize_size

    def forward(self, vertex, tri, feat = None, visible_vertice = False):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, N, C), features
        """
        assert vertex.device == torch.device('cpu')
        assert tri.device == torch.device('cpu')
        assert vertex.shape[0] == 1 # only support batchsize = 1

        device = vertex.device
        # ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        # if vertex.shape[-1] == 3:
        #     vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
        vertex[..., 1] = -vertex[..., 1] 

        vertex_ndc = vertex # @ ndc_proj.t()
        # vertex_ndc = vertex_ndc[..., :] / vertex_ndc[..., 3:]

        # initial
        c = 3
        h = self.rasterize_size
        w = self.rasterize_size
        vertices = vertex_ndc[0,:,:3].detach().contiguous().numpy().astype(np.float32).copy()[:,:3]

        vertices[:,0] = vertices[:,0] * w/2
        vertices[:,1] = vertices[:,1] * h/2
        vertices[:,0] = vertices[:,0] + w/2
        vertices[:,1] = vertices[:,1] + h/2

        vertices[:,2] = - vertex[..., 2].detach()

        triangles = tri.numpy().astype(np.int32).copy()
        if feat is not None:
            colors = feat.detach().numpy()[0].astype(np.float32).copy()
        else:
            colors = np.zeros_like(vertices)

        depth_buffer = np.zeros([h, w], dtype = np.float32) - 999999. #set the initial z to the farest position
        triangle_buffer = np.zeros([h, w], dtype = np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
        barycentric_weight = np.zeros([h, w, 3], dtype = np.float32)  # 
        image = np.zeros((h, w, c), dtype = np.float32)

        mesh_core_cython.MeshRenderer_cpu_core(
                image, vertices, triangles,
                colors,
                depth_buffer, triangle_buffer, barycentric_weight, 
                vertices.shape[0], triangles.shape[0], 
                h, w, c)
        depth_buffer[depth_buffer == - 999999] = 0
        depth_buffer = - depth_buffer[np.newaxis, np.newaxis, :]

        unique_visible_verts_idx = None
        if visible_vertice:
            visible_faces = np.unique(triangle_buffer)
            visible_faces = visible_faces[visible_faces!=-1]
            visible_verts_idx = tri[visible_faces]
            unique_visible_verts_idx = np.unique(visible_verts_idx).astype(np.int32)
            unique_visible_verts_idx = torch.from_numpy(unique_visible_verts_idx)

        mask =  (triangle_buffer > 0).astype(np.float32)[np.newaxis, np.newaxis, :]
        image = image.transpose(2,0,1)[np.newaxis, :]
        return torch.from_numpy(mask), torch.from_numpy(depth_buffer), torch.from_numpy(image), unique_visible_verts_idx
