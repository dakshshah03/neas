import os
import glob
import yaml
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from .tigre import ConeGeometry, TIGREDataset

class SimDataset(Dataset):
    def __init__(self, path, n_rays=1024, type="train", device="cuda", num_views=None, n_mask_rays=0):
        super().__init__()
        self.type = type
        self.device = device
        self.n_rays = n_rays
        self.n_mask_rays = n_mask_rays
        
        ###########################
        #everything in this block is different than the original tigre dataset, but otherwise stuff should be functionally the same
        # Load configs
        with open(os.path.join(path, "ct_configuration.yml"), "r") as f:
            ct_config = yaml.safe_load(f)
            
        DSD = float(ct_config["source_to_detector"])
        DSO = float(ct_config["source_to_patient"])
        n_det = [int(ct_config["height"]), int(ct_config["width"])]
        s_det = [float(ct_config["detector_size"]), float(ct_config["detector_size"])]
        d_det = [s_det[0] / n_det[0], s_det[1] / n_det[1]]
        
        sVoxel = [float(ct_config["scanner_fov_z"]), float(ct_config["scanner_fov_y"]), float(ct_config["scanner_fov_x"])]
        # Default voxel grid for evaluation
        nVoxel = [int(ct_config["height"]), int(ct_config["height"]), int(ct_config["width"])]
        dVoxel = [sVoxel[0]/nVoxel[0], sVoxel[1]/nVoxel[1], sVoxel[2]/nVoxel[2]]
        
        #######################
        
        geo_data = {
            "DSD": DSD,
            "DSO": DSO,
            "nDetector": n_det,
            "dDetector": d_det,
            "nVoxel": nVoxel,
            "dVoxel": dVoxel,
            "offOrigin": [0.0, 0.0, 0.0],
            "offDetector": [0.0, 0.0],
            "accuracy": 0.5,
            "mode": "cone",
            "filter": "ram-lak"
        }
        
        self.geo = ConeGeometry(geo_data)
        
        # Scale to fit unit sphere
        max_extent = np.max(self.geo.sVoxel)
        half_diagonal = np.sqrt(np.sum((self.geo.sVoxel / 2) ** 2))
        self.scene_scale = 1.0 / half_diagonal
        self.geo.DSO *= self.scene_scale
        self.geo.DSD *= self.scene_scale
        self.geo.dDetector *= self.scene_scale
        self.geo.sDetector *= self.scene_scale
        self.geo.dVoxel *= self.scene_scale
        self.geo.sVoxel *= self.scene_scale
        self.geo.offOrigin *= self.scene_scale
        self.geo.offDetector *= self.scene_scale
        
        self.near, self.far = self.get_near_far(self.geo)
        
        # Load projections and rays
        png_files = sorted(glob.glob(os.path.join(path, "xray*.png")))
        total_views = len(png_files)
        
        # The angles should just be evenly spaced like in standard TIGRE dataset
        all_angles = np.linspace(0, 2 * np.pi, total_views, endpoint=False)
        
        if type == "train":
            if num_views is not None and num_views < total_views:
                step = total_views / num_views
                indices = [int(np.round(i * step)) for i in range(num_views)]
                png_files = [png_files[i] for i in indices]
                all_angles = all_angles[indices]
                
        elif type == "val":
            pass # Keep all for val for now
            
        projs = []
        for img_path in png_files:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None: continue
            
            # Typically 0-255 images for these simulation PNGs. 255 is background (I_0)
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            
            eps = 1e-6
            # projs -> projection thickness. Image is I, formula I = e^-projs -> projs = -ln(I)
            proj = -np.log(np.clip(img, eps, 1.0))
            projs.append(proj)
            
        self.projs = torch.tensor(np.stack(projs, axis=0), dtype=torch.float32, device=device)
        self.projs_intensity = torch.exp(-self.projs)
        
        # Compute rays correctly using TIGRE's functions
        self.angles = all_angles
        rays = self.get_rays(all_angles, self.geo, device)
        self.rays = torch.cat(
            [
                rays,
                torch.ones_like(rays[..., :1]) * self.near,
                torch.ones_like(rays[..., :1]) * self.far,
            ],
            dim=-1,
        )
        
        self.n_samples = len(projs)
        
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1], device=device),
                torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0], device=device),
                indexing="ij",
            ), -1
        )
        self.coords = torch.reshape(coords, [-1, 2])
        self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
        self.image = torch.zeros(tuple(nVoxel), dtype=torch.float32, device=device)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.type == "train":
            projs_valid = (self.projs[index] > 0).flatten()
            coords_valid = self.coords[projs_valid]
            select_inds = np.random.choice(
                coords_valid.shape[0], size=[self.n_rays], replace=False
            )
            select_coords = coords_valid[select_inds].long()
            rays = self.rays[index, select_coords[:, 0], select_coords[:, 1]]
            projs = self.projs[index, select_coords[:, 0], select_coords[:, 1]]
            projs_intensity = self.projs_intensity[index, select_coords[:, 0], select_coords[:, 1]]
            out = {
                "projs": projs,
                "projs_intensity": projs_intensity,
                "rays": rays,
            }

            # Mask rays: sample from zero-attenuation (air) pixels for the mask loss.
            if self.n_mask_rays > 0:
                coords_zero = self.coords[~projs_valid]
                if coords_zero.shape[0] > 0:
                    replace = coords_zero.shape[0] < self.n_mask_rays
                    mask_inds = np.random.choice(
                        coords_zero.shape[0], size=[self.n_mask_rays], replace=replace
                    )
                    mask_coords = coords_zero[mask_inds].long()
                    out["mask_rays"] = self.rays[index, mask_coords[:, 0], mask_coords[:, 1]]
                    out["mask_projs_intensity"] = self.projs_intensity[index, mask_coords[:, 0], mask_coords[:, 1]]
        elif self.type == "val":
            rays = self.rays[index]
            projs = self.projs[index]
            projs_intensity = self.projs_intensity[index]
            out = {
                "projs": projs,
                "projs_intensity": projs_intensity,
                "rays": rays,
            }
        return out

    def get_voxels(self, geo: ConeGeometry):
        """
        Get the voxels.
        """
        n1, n2, n3 = geo.nVoxel
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        offOrigin = geo.offOrigin

        xyz = np.meshgrid(
            np.linspace(-s1, s1, n1),
            np.linspace(-s2, s2, n2),
            np.linspace(-s3, s3, n3),
            indexing="ij",
        )
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        voxel = voxel + offOrigin[None, None, None, :]
        return voxel

    def get_rays(self, angles, geo: ConeGeometry, device):
        """
        Get rays given one angle and x-ray machine geometry.
        """

        W, H = geo.nDetector
        DSD = geo.DSD
        rays = []

        for angle in angles:
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle)).to(device)
            rays_o, rays_d = None, None
            if geo.mode == "cone":
                i, j = torch.meshgrid(
                    torch.linspace(0, W - 1, W, device=device),
                    torch.linspace(0, H - 1, H, device=device),
                    indexing="ij",
                )  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)
                rays_d = torch.sum(
                    torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1
                )  # pose[:3, :3] *
                rays_o = pose[:3, -1].expand(rays_d.shape)
            elif geo.mode == "parallel":
                i, j = torch.meshgrid(
                    torch.linspace(0, W - 1, W, device=device),
                    torch.linspace(0, H - 1, H, device=device),
                    indexing="ij",
                )  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack(
                    [torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)],
                    -1,
                )
                rays_d = torch.sum(
                    torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1
                )  # pose[:3, :3] *
                rays_o = torch.sum(
                    torch.matmul(
                        pose[:3, :3],
                        torch.stack([uu, vv, torch.zeros_like(uu)], -1)[..., None],
                    ).to(device),
                    -1,
                ) + pose[:3, -1].expand(rays_d.shape)

            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(torch.concat([rays_o, rays_d], dim=-1))

        return torch.stack(rays, dim=0)

    def angle2pose(self, DSO, angle):
        phi1 = -np.pi / 2
        R1 = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(phi1), -np.sin(phi1)],
                [0.0, np.sin(phi1), np.cos(phi1)],
            ]
        )
        phi2 = np.pi / 2
        R2 = np.array(
            [
                [np.cos(phi2), -np.sin(phi2), 0.0],
                [np.sin(phi2), np.cos(phi2), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        R3 = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rot = np.dot(np.dot(R3, R2), R1)
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T

    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):
        """
        Compute the near and far threshold.
        """
        dist1 = np.linalg.norm(
            [geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2]
        )
        dist2 = np.linalg.norm(
            [geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2]
        )
        dist3 = np.linalg.norm(
            [geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2]
        )
        dist4 = np.linalg.norm(
            [geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2]
        )
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far
