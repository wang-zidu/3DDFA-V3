import numpy as np
import torch
import os
import cv2
import torch.nn.functional as F
# from util.nv_diffrast import MeshRenderer, MeshRenderer_UV
import argparse
from . import networks

def get_colors_from_uv(colors, uv_coords):
    res = bilinear_interpolate_numpy(colors, uv_coords[:, 0], uv_coords[:, 1])
    return res

def bilinear_interpolate_numpy(img, x, y):

    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    i_a = img[y0, x0]
    i_b = img[y1, x0]
    i_c = img[y0, x1]
    i_d = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa[..., np.newaxis] * i_a + wb[..., np.newaxis] * i_b + wc[..., np.newaxis] * i_c + wd[..., np.newaxis] * i_d

def process_uv(uv_coords, uv_h = 224, uv_w = 224):
    uv_coords[:,0] = uv_coords[:,0] * (uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1] * (uv_h - 1)
    # uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def bilinear_interpolate(img, x, y):
    
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, img.shape[1] - 1)
    x1 = torch.clamp(x1, 0, img.shape[1] - 1)
    y0 = torch.clamp(y0, 0, img.shape[0] - 1)
    y1 = torch.clamp(y1, 0, img.shape[0] - 1)

    i_a = img[y0, x0]
    i_b = img[y1, x0]
    i_c = img[y0, x1]
    i_d = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa.unsqueeze(-1) * i_a + wb.unsqueeze(-1) * i_b + wc.unsqueeze(-1) * i_c + wd.unsqueeze(-1) * i_d

class face_model:
    def __init__(self, args):

        self.args = args

        self.device = self.args.device
        model = np.load("./assets/face_model.npy",allow_pickle=True).item()

        # mean shape, size (107127, 1)
        self.u = torch.tensor(model['u'], requires_grad=False, dtype=torch.float32, device=self.device)
        # face identity bases, size (107127, 80)
        self.id = torch.tensor(model['id'], requires_grad=False, dtype=torch.float32, device=self.device)
        # face expression bases, size (107127, 64)
        self.exp = torch.tensor(model['exp'], requires_grad=False, dtype=torch.float32, device=self.device)
        # mean albedo, size (107127, 1)
        self.u_alb = torch.tensor(model['u_alb'], requires_grad=False, dtype=torch.float32, device=self.device)
        # face albedo bases, size (107127, 80)
        self.alb = torch.tensor(model['alb'], requires_grad=False, dtype=torch.float32, device=self.device)
        # for computing vertex normals, size (35709, 8), see https://github.com/sicxu/Deep3DFaceRecon_pytorch/issues/132
        self.point_buf = torch.tensor(model['point_buf'], requires_grad=False, dtype=torch.int64, device=self.device)
        # triangle faces, size (70789, 3)
        self.tri = torch.tensor(model['tri'], requires_grad=False, dtype=torch.int64, device=self.device)
        # vertex uv coordinates, size (35709, 3), range (0, 1.)
        self.uv_coords = torch.tensor(model['uv_coords'], requires_grad=False, dtype=torch.float32, device=self.device)
        
        if args.extractTex:
            uv_coords_numpy = process_uv(model['uv_coords'].copy(), 1024, 1024)
            self.uv_coords_torch = (torch.tensor(uv_coords_numpy, requires_grad=False, dtype=torch.float32, device=self.device) / 1023 - 0.5) * 2
            if self.device == 'cpu':
                from util.cpu_renderer import MeshRenderer_UV_cpu
                self.uv_renderer = MeshRenderer_UV_cpu(
                        rasterize_size=int(1024.)
                )
                self.uv_coords_torch = self.uv_coords_torch + 1e-6 # For CPU renderer, a slight perturbation may be needed to avoid certain rendering artifacts. Users can comment out 1e-6 to compare different texture effects.
            else:
                from util.nv_diffrast import MeshRenderer_UV
                self.uv_renderer = MeshRenderer_UV(
                        rasterize_size=int(1024.)
                )
            self.uv_coords_numpy = uv_coords_numpy.copy()
            self.uv_coords_numpy[:,1] = 1024 - self.uv_coords_numpy[:,1] - 1

        # vertex indices for 68 landmarks, size (68,)
        if self.args.ldm68:
            self.ldm68 = torch.tensor(model['ldm68'], requires_grad=False, dtype=torch.int64, device=self.device)
        # vertex indices for 106 landmarks, size (106,)
        if self.args.ldm106 or self.args.ldm106_2d:
            self.ldm106 = torch.tensor(model['ldm106'], requires_grad=False, dtype=torch.int64, device=self.device)
        # vertex indices for 134 landmarks, size (134,)
        if self.args.ldm134:
            self.ldm134 = torch.tensor(model['ldm134'], requires_grad=False, dtype=torch.int64, device=self.device)

        # segmentation annotation indices for 8 parts, [right_eye, left_eye, right_eyebrow, left_eyebrow, nose, up_lip, down_lip, skin]
        if self.args.seg_visible:
            self.annotation = model['annotation']

        # segmentation triangle faces for 8 parts
        if self.args.seg:
            self.annotation_tri = [torch.tensor(i, requires_grad=False, dtype=torch.int64, device=self.device) for i in model['annotation_tri']]

        # face profile parallel, list
        if self.args.ldm106_2d:
            self.parallel = model['parallel']
            # parallel for profile matching
            self.v_parallel = - torch.ones(35709, device=self.device).type(torch.int64)
            for i in range(len(self.parallel)):
                self.v_parallel[self.parallel[i]]=i

        # focal = 1015, center = 112
        self.persc_proj = torch.tensor([1015.0, 0, 112.0, 0, 1015.0, 112.0, 0, 0, 1], requires_grad=False, dtype=torch.float32, device=self.device).reshape([3, 3]).transpose(0,1)
        self.camera_distance = 10.0
        self.init_lit = torch.tensor([0.8, 0, 0, 0, 0, 0, 0, 0, 0], requires_grad=False, dtype=torch.float32, device=self.device).reshape([1, 1, -1])
        self.SH_a = torch.tensor([np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)], requires_grad=False, dtype=torch.float32, device=self.device)
        self.SH_c = torch.tensor([1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)], requires_grad=False, dtype=torch.float32, device=self.device)
        if self.device == 'cpu':
            from util.cpu_renderer import MeshRenderer_cpu
            self.renderer = MeshRenderer_cpu(
                        rasterize_fov=2 * np.arctan(112. / 1015) * 180 / np.pi, znear=5., zfar=15., rasterize_size=int(2 * 112.)
            )
        else:
            from util.nv_diffrast import MeshRenderer
            self.renderer = MeshRenderer(
                        rasterize_fov=2 * np.arctan(112. / 1015) * 180 / np.pi, znear=5., zfar=15., rasterize_size=int(2 * 112.)
            )

        if args.backbone == 'resnet50':
            self.net_recon = networks.define_net_recon(
                net_recon='resnet50', use_last_fc=False, init_path=None
            )
            self.net_recon.load_state_dict(torch.load("assets/net_recon.pth", map_location=torch.device('cpu'))['net_recon'])
            self.net_recon = self.net_recon.to(self.device)
            self.net_recon.eval()

        if args.backbone == 'mbnetv3':
            self.net_recon = networks.define_net_recon_mobilenetv3(
                net_recon='recon_mobilenetv3_large', use_last_fc=False, init_path=None
            )
            self.net_recon.load_state_dict(torch.load("assets/net_recon_mbnet.pth", map_location=torch.device('cpu'))['net_recon'])
            self.net_recon = self.net_recon.to(self.device)
            self.net_recon.eval()

        self.input_img = None

    def compute_shape(self, alpha_id, alpha_exp):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3), face vertice without rotation or translation

        Parameters:
            alpha_id         -- torch.tensor, size (B, 80), identity parameter
            alpha_exp        -- torch.tensor, size (B, 64), expression parameter
        """
        batch_size = alpha_id.shape[0]
        face_shape = torch.einsum('ij,aj->ai', self.id, alpha_id) + torch.einsum('ij,aj->ai', self.exp, alpha_exp) + self.u.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])

    def compute_albedo(self, alpha_alb, normalize=True):
        """
        Return:
            face_albedo     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.), without lighting

        Parameters:
            alpha_alb        -- torch.tensor, size (B, 80), albedo parameter
        """
        batch_size = alpha_alb.shape[0]
        face_albedo = torch.einsum('ij,aj->ai', self.alb, alpha_alb) + self.u_alb.reshape([1, -1])
        if normalize:
            face_albedo = face_albedo / 255.
        return face_albedo.reshape([batch_size, -1, 3])

    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        v1 = face_shape[:, self.tri[:, 0]]
        v2 = face_shape[:, self.tri[:, 1]]
        v3 = face_shape[:, self.tri[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
        
        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm

    def compute_texture(self, face_albedo, face_norm, alpha_sh):
        """
        Return:
            face_texture        -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_albedo         -- torch.tensor, size (B, N, 3), from albedo model, range (0, 1.)
            face_norm           -- torch.tensor, size (B, N, 3), rotated face normal
            alpha_sh            -- torch.tensor, size (B, 27), SH parameter
        """
        batch_size = alpha_sh.shape[0]
        v_num = face_albedo.shape[1]
        a = self.SH_a
        c = self.SH_c
        alpha_sh = alpha_sh.reshape([batch_size, 3, 9])
        alpha_sh = alpha_sh + self.init_lit
        alpha_sh = alpha_sh.permute(0, 2, 1)
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2],
             a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ alpha_sh[..., :1]
        g = Y @ alpha_sh[..., 1:2]
        b = Y @ alpha_sh[..., 2:]
        face_texture = torch.cat([r, g, b], dim=-1) * face_albedo
        return face_texture

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3), pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), use radian
        """
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]
        return face_proj

    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)

    def split_alpha(self, alpha):
        """
        Return:
            alpha_dict     -- a dict of torch.tensors

        Parameters:
            alpha          -- torch.tensor, size (B, 256)
        """
        alpha_id = alpha[:, :80]
        alpha_exp = alpha[:, 80: 144]
        alpha_alb = alpha[:, 144: 224]
        alpha_a = alpha[:, 224: 227]
        alpha_sh = alpha[:, 227: 254]
        alpha_t = alpha[:, 254:]
        return {
            'id': alpha_id,
            'exp': alpha_exp,
            'alb': alpha_alb,
            'angle': alpha_a,
            'sh': alpha_sh,
            'trans': alpha_t
        }

    def get_landmarks_68(self, v2d):
        """
        Return:
            landmarks_68_3d         -- torch.tensor, size (B, 68, 2)

        Parameters:
            v2d                     -- torch.tensor, size (B, N, 2)
        """
        return v2d[:, self.ldm68]

    def get_landmarks_106(self, v2d):
        """
        Return:
            landmarks_106_3d         -- torch.tensor, size (B, 106, 2)

        Parameters:
            v2d                      -- torch.tensor, size (B, N, 2)
        """
        return v2d[:, self.ldm106]

    def get_landmarks_134(self, v2d):
        """
        Return:
            landmarks_134            -- torch.tensor, size (B, 134, 2)

        Parameters:
            v2d                      -- torch.tensor, size (B, N, 2)
        """
        return v2d[:, self.ldm134]

    def get_landmarks_106_2d(self, v2d, face_shape, alpha_dict):
        """
        Return:
            landmarks_106_2d         -- torch.tensor, size (B, 106, 2)

        Parameters:
            v2d                     -- torch.tensor, size (B, N, 2)
            face_shape              -- torch.tensor, size (B, N, 3), face vertice without rotation or translation
            alpha_dict              -- a dict of torch.tensors
        """

        temp_angle = alpha_dict['angle'].clone()
        temp_angle[:,2] = 0
        rotation_without_roll = self.compute_rotation(temp_angle)
        v2d_without_roll = self.to_image(self.to_camera(self.transform(face_shape, rotation_without_roll, alpha_dict['trans'])))

        visible_parallel = self.v_parallel.clone()
        # visible_parallel[visible_idx == 0] = -1

        ldm106_dynamic=self.ldm106.clone()
        for i in range(16):
            temp=v2d_without_roll.clone()[:,:,0]
            temp[:,visible_parallel!=i] = 1e5
            ldm106_dynamic[i]=torch.argmin(temp)

        for i in range(17,33):
            temp=v2d_without_roll.clone()[:,:,0]
            temp[:,visible_parallel!=i] = -1e5
            ldm106_dynamic[i]=torch.argmax(temp)

        return v2d[:, ldm106_dynamic]

    def add_directionlight(self, normals, lights):
        '''
        see https://github.com/yfeng95/DECA/blob/master/decalib/utils/renderer.py
        '''
        light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def compute_gray_shading_with_directionlight(self, face_texture, normals):
        '''
        see https://github.com/yfeng95/DECA/blob/master/decalib/utils/renderer.py
        '''
        batch_size = normals.shape[0]
        light_positions = torch.tensor(
                [
                [-1,1,1],
                [1,1,1],
                [-1,-1,1],
                [1,-1,1],
                [0,0,1]
                ]
        )[None,:,:].expand(batch_size, -1, -1).float()
        light_intensities = torch.ones_like(light_positions).float()*1.7
        lights = torch.cat((light_positions, light_intensities), 2).to(face_texture.device)

        shading = self.add_directionlight(normals, lights)
        texture =  face_texture*shading
        return texture

    def segmentation(self, v3d):

        seg = torch.zeros(224,224,8).to(v3d.device)
        for i in range(8):
            mask, _, _, _ = self.renderer(v3d.clone(), self.annotation_tri[i])
            seg[:,:,i] = mask.squeeze()
        return seg

    def segmentation_visible(self, v3d, visible_idx):

        seg = torch.zeros(224,224,8).to(v3d.device)
        for i in range(8):
            temp = torch.zeros_like(v3d)
            temp[:,self.annotation[i],:] = 1
            temp[:,visible_idx == 0,:] = 0
            _, _, temp_image, _ = self.renderer(v3d.clone(), self.tri, temp.clone())
            temp_image = temp_image.mean(axis=1)
            mask = torch.where(temp_image >= 0.5, torch.tensor(1.0).to(v3d.device), torch.tensor(0.0).to(v3d.device))
            seg[:,:,i] = mask.squeeze()
        return seg

    def forward(self):
        assert self.net_recon.training == False
        alpha = self.net_recon(self.input_img)

        alpha_dict = self.split_alpha(alpha)
        face_shape = self.compute_shape(alpha_dict['id'], alpha_dict['exp'])
        rotation = self.compute_rotation(alpha_dict['angle'])
        face_shape_transformed = self.transform(face_shape, rotation, alpha_dict['trans'])

        # face vertice in 3d
        v3d = self.to_camera(face_shape_transformed)

        # face vertice in 2d image plane
        v2d = self.to_image(v3d)

        # compute face texture with albedo and lighting
        face_albedo = self.compute_albedo(alpha_dict['alb'])
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_texture = self.compute_texture(face_albedo, face_norm_roted, alpha_dict['sh'])

        # render shape with texture
        _, _, pred_image, visible_idx_renderer = self.renderer(v3d.clone(), self.tri, torch.clamp(face_texture, 0, 1).clone(), visible_vertice = True)

        # render shape
        gray_shading = self.compute_gray_shading_with_directionlight(torch.ones_like(face_albedo)*0.78,face_norm_roted)
        mask, _, pred_image_shape, _ = self.renderer(v3d.clone(), self.tri, gray_shading.clone())

        result_dict = {
            'v3d': v3d.detach().cpu().numpy(),
            'v2d': v2d.detach().cpu().numpy(),
            'face_texture': np.clip(face_texture.detach().cpu().numpy(), 0, 1),
            'tri': self.tri.detach().cpu().numpy(),
            'uv_coords': self.uv_coords.detach().cpu().numpy(),
            'render_shape': pred_image_shape.detach().cpu().permute(0, 2, 3, 1).numpy(),
            'render_face': pred_image.detach().cpu().permute(0, 2, 3, 1).numpy(),
            'render_mask': mask.detach().cpu().permute(0, 2, 3, 1).numpy(),
        }

        # compute visible vertice according to normal and renderer
        if self.args.seg_visible or self.args.extractTex:
            visible_idx = torch.zeros(35709).type(torch.int64).to(v3d.device)
            visible_idx[visible_idx_renderer.type(torch.int64)] = 1
            visible_idx[(face_norm_roted[..., 2] < 0)[0]] = 0
            # result_dict['visible_idx'] = visible_idx

        # landmarks 68 3d
        if self.args.ldm68:
            v2d_68 = self.get_landmarks_68(v2d)
            result_dict['ldm68'] = v2d_68.detach().cpu().numpy()

        # landmarks 106 3d
        if self.args.ldm106:
            v2d_106 = self.get_landmarks_106(v2d)
            result_dict['ldm106'] = v2d_106.detach().cpu().numpy()

        # landmarks 106 2d
        if self.args.ldm106_2d:
            # v2d_106_2d = self.get_landmarks_106_2d(v2d, face_shape, alpha_dict, visible_idx)
            v2d_106_2d = self.get_landmarks_106_2d(v2d, face_shape, alpha_dict)
            result_dict['ldm106_2d'] = v2d_106_2d.detach().cpu().numpy()

        # landmarks 134
        if self.args.ldm134:
            v2d_134 = self.get_landmarks_134(v2d)
            result_dict['ldm134'] = v2d_134.detach().cpu().numpy()

        # segmentation in 2d without visible mask
        if self.args.seg:
            seg = self.segmentation(v3d)
            result_dict['seg'] = seg.detach().cpu().numpy()

        # segmentation in 2d with visible mask
        if self.args.seg_visible:
            seg_visible = self.segmentation_visible(v3d, visible_idx)
            result_dict['seg_visible'] = seg_visible.detach().cpu().numpy()

        # use median-filtered-weight pca-texture for texture blending at invisible region, todo: poisson blending should give better-looking results
        if self.args.extractTex:
            _, _, uv_color_pca, _ = self.uv_renderer(self.uv_coords_torch.unsqueeze(0).clone(), self.tri, (torch.clamp(face_texture, 0, 1)).clone())
            img_colors = bilinear_interpolate(self.input_img.permute(0, 2, 3, 1).detach()[0], v2d[0, :, 0].detach(), 223 - v2d[0, :, 1].detach())
            _, _, uv_color_img, _ = self.uv_renderer(self.uv_coords_torch.unsqueeze(0).clone(), self.tri, img_colors.unsqueeze(0).clone())
            _, _, uv_weight, _ = self.uv_renderer(self.uv_coords_torch.unsqueeze(0).clone(), self.tri, (1 - torch.stack((visible_idx,)*3, axis=-1).unsqueeze(0).type(torch.float32).to(self.tri.device)).clone())

            median_filtered_w = cv2.medianBlur((uv_weight.detach().cpu().permute(0, 2, 3, 1).numpy()[0]*255).astype(np.uint8), 31)/255.

            uv_color_pca = uv_color_pca.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
            uv_color_img = uv_color_img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]

            res_colors = ((1 - median_filtered_w) * np.clip(uv_color_img, 0, 1) + median_filtered_w * np.clip(uv_color_pca, 0, 1))
            # result_dict['extractTex_uv'] = res_colors
            v_colors = get_colors_from_uv(res_colors.copy(), self.uv_coords_numpy.copy())
            result_dict['extractTex'] = v_colors

        return result_dict
