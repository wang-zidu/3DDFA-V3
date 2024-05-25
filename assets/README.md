# Download

- Hugging Face: [download link](https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/tree/main/assets)

# File Structure

```
assets
├── face_model.npy              # (face model)
├── large_base_net.pth
├── net_recon.pth               # (backbone)
├── retinaface_resnet50_2020-07-20_old_torch.pth
├── similarity_Lm3D_all.mat
├── indices_38365_35709.npy     # (optional)
├── indices_53215_35709.npy     # (optional)
├── indices_53215_38365.npy     # (optional)
├── indices_53490_35709.npy     # (optional)
├── meanshape-68ldms.obj        # (optional)
├── meanshape-106ldms.obj       # (optional)
├── meanshape-134ldms.obj       # (optional)
├── meanshape-parallel.obj      # (optional)
└── meanshape-seg.obj           # (optional)
```

# 3D Mesh Masks and Some Useful Attributes
- Load the masks (annotations) using the following method:
    ```
    import numpy as np
    model = np.load("./assets/face_model.npy",allow_pickle=True).item()
    ```
    `model['annotation']` and `model['annotation_tri']`: segmentation annotation indices and triangle faces for 8 parts.

    `model['ldm106']`: vertex indices for 106 landmarks.

    `model['parallel']`: parallel for 33 face profile landmarks, used for dynamic 2D and 3D [landmark marching](https://openaccess.thecvf.com/content_cvpr_2015/papers/Zhu_High-Fidelity_Pose_and_2015_CVPR_paper.pdf)

    `model['ldm134']`: vertex indices for 134 landmarks.

    `model['ldm68']`: vertex indices for 68 landmarks.

    <br>![teaser](/examples/teaser/annotation_and_ldms.png)


- The commonly used [BFM](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) has 4 different topological structures with 53,490 / 53,215 / 38,365 / 35,709 vertices. We summarize the correspondence indices and provide them in the form of `indices_A_B.npy`, where `A` and `B` represent the vertex numbers (index starts from 0). These indices and obj files are not essential for running [3DDFA_V3](https://arxiv.org/abs/2312.00311), but they may help beginners better and more quickly use [BFM](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) in some cases.

# Other Assets
- `net_recon.pth` is the checkpoint of our [3DDFA_V3](https://arxiv.org/abs/2312.00311).
- `face_model.npy` is the face model and attributes based on [3DDFA_V3](https://arxiv.org/abs/2312.00311), [BFM](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model), [Exp_Pca](https://github.com/Juyong/3DFace), and [Deep3D](https://github.com/sicxu/Deep3DFaceRecon_pytorch).
- `large_base_net.pth` and `retinaface_resnet50_2020-07-20_old_torch.pth` are used for face detector from [HRN](https://github.com/youngLBW/HRN) and [retinaface](https://github.com/biubug6/Pytorch_Retinaface).
- `similarity_Lm3D_all.mat` is used for cropping faces from [Deep3D](https://github.com/sicxu/Deep3DFaceRecon_pytorch).
