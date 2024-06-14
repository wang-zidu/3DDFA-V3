# 3D Face Reconstruction with the Geometric Guidance of Facial Part Segmentation

By [Zidu Wang](https://scholar.google.com/citations?user=7zD5f0IAAAAJ&hl=zh-CN&oi=ao), [Xiangyu Zhu](https://xiangyuzhu-open.github.io/homepage/), [Tianshuo Zhang](tianshuo.zhang@nlpr.ia.ac.cn), [Baiqin Wang](wangbaiqin2024@ia.ac.cn) and [Zhen Lei](http://www.cbsr.ia.ac.cn/users/zlei/).

This repository is the official implementation of [3DDFA_V3](https://arxiv.org/abs/2312.00311) in [CVPR2024 (Highlight)](https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers).

![teaser](/examples/teaser/teaser.jpg)

[3DDFA_V3](https://arxiv.org/abs/2312.00311) uses the geometric guidance of facial part segmentation for face reconstruction, improving the alignment of reconstructed facial features with the original image and excelling at capturing extreme expressions. The key idea is to transform the target and prediction into semantic point sets, optimizing the distribution of point sets to ensure that the reconstructed regions and the target share the same geometry.

## News

* [06/14/2024] We provide a fast version based on [MobileNet-V3](https://arxiv.org/abs/1905.02244), which achieves similar results to the ResNet-50 version at a higher speed. Please note that if your environment supports ResNet-50, we still strongly recommend using the ResNet-50 version. (The MobileNet-V3 version is still under testing, and we may update it further in the future.)

## Getting Started
### Environment
  ```bash
  # Clone the repo:
  git clone https://github.com/wang-zidu/3DDFA-V3
  cd 3DDFA-V3

  conda create -n TDDFAV3 python=3.8
  conda activate TDDFAV3

  # The pytorch version is not strictly required.
  pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
  # or: conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

  pip install -r requirements.txt

  # Some results in the paper are rendered by pytorch3d and nvdiffrast
  # This repository only uses nvdiffrast for convenience.
  git clone https://github.com/NVlabs/nvdiffrast.git
  cd nvdiffrast
  pip install .
  cd ..
  ```  

### Usage
1. Please refer to this [README](https://github.com/wang-zidu/3DDFA-V3/blob/main/assets/) to prepare assets and pretrained models.

2. Run demos.


    ```
    python demo.py --inputpath examples/ --savepath examples/results --device cuda --iscrop 1 --detector retinaface --ldm68 1 --ldm106 1 --ldm106_2d 1 --ldm134 1 --seg_visible 1 --seg 1 --useTex 1 --extractTex 1 --backbone resnet50
    ```

     - `--inputpath`: path to the test data, should be a image folder.

     - `--savepath`: path to the output directory, where results (obj, png files) will be stored.

     - `--iscrop`: whether to crop input image, set false only when the test image are well cropped and resized into (224,224,3).

     - `--detector`: face detector for cropping image, support for retinaface (recommended) and mtcnn.

     - `--ldm68`, `--ldm106`, `--ldm106_2d` and `--ldm134`: save and show landmarks.

     - `--backbone`: backbone for reconstruction, support for resnet50 and mbnetv3.
  
     <br>With the 3D mesh annotations provided by [3DDFA_V3](https://arxiv.org/abs/2312.00311), we can generate 2D facial segmentation results based on the 3D mesh:


      - `--seg_visible`: save and show segmentation in 2D with visible mask. When a part becomes invisible due to pose changes, the corresponding region will not be displayed. All segmentation results of the 8 parts will be shown in a single subplot. 

      - `--seg`: save and show segmentation in 2D. When a part becomes invisible due to pose changes, the corresponding segmented region will still be displayed (obtained from 3D estimation), and the segmentation information of the 8 parts will be separately shown in 8 subplots.


    <br>We provide two types of 3D mesh files in OBJ format as output.


     - `--useTex`: save .obj use texture from BFM model.

     - `--extractTex`: save .obj use texture extracted from the input image. We use median-filtered-weight pca-texture for texture blending at invisible region (Poisson blending should give better-looking results).

3. Results.
     - `image_name.png`: the visualization results.
     - `image_name.npy`: landmarks, segmentation, etc.
     - `image_name_pcaTex.obj`: 3D mesh files in OBJ format using texture from the BFM model.
     - `image_name_extractTex.obj`: 3D mesh files in OBJ format using texture extracted from the input image.

    <br>![teaser](/examples/teaser/result.png)<br>
    <br>![teaser](/examples/teaser/result2.png)<br>

## 3D Mesh Part Masks
Please refer to this [README](https://github.com/wang-zidu/3DDFA-V3/blob/main/assets/) to download our masks (annotations).

![teaser](/examples/teaser/annotation.jpg)

We provide a new 3D mesh part masks aligned with the semantic regions in 2D face segmentation. The current version is based on [BFM](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) (with 35,709 vertices), which shares the same topology as the face models used by [Deep3D](https://github.com/sicxu/Deep3DFaceRecon_pytorch), [MGCNet](https://github.com/jiaxiangshang/MGCNet), [HRN](https://github.com/youngLBW/HRN), etc. We also provide some other useful attributes.

## Synthetic Expression Data
Please refer to this [README](https://github.com/wang-zidu/3DDFA-V3/tree/main/data) to download data.

![teaser](/examples/teaser/data.png)

Based on [MaskGan](https://github.com/switchablenorms/CelebAMask-HQ/tree/master), we introduce a new synthetic face dataset including closed-eye, open-mouth, and frown expressions.



## Citation
If you use our work in your research, please cite our publication:
```
@article{wang20233d,
  title={3D Face Reconstruction with the Geometric Guidance of Facial Part Segmentation},
  author={Wang, Zidu and Zhu, Xiangyu and Zhang, Tianshuo and Wang, Baiqin and Lei, Zhen},
  journal={arXiv preprint arXiv:2312.00311},
  year={2023}
}
```

## Acknowledgements
There are some functions or scripts in this implementation that are based on external sources. We thank the authors for their excellent works. Here are some great resources we benefit: [Deep3D](https://github.com/sicxu/Deep3DFaceRecon_pytorch), [DECA](https://github.com/yfeng95/DECA), [HRN](https://github.com/youngLBW/HRN), [3DDFA-V2](https://github.com/cleardusk/3DDFA_V2/tree/master), [Nvdiffrast](https://github.com/NVlabs/nvdiffrast), [Pytorch3D](https://pytorch3d.org/), [Retinaface](https://github.com/biubug6/Pytorch_Retinaface), [MTCNN](https://github.com/ipazc/mtcnn), [MaskGan](https://github.com/switchablenorms/CelebAMask-HQ/tree/master), [DML-CSR](https://github.com/deepinsight/insightface/tree/master/parsing/dml_csr), [REALY](https://realy3dface.com/).


## Contact

 We plan to train [3DDFA_V3](https://arxiv.org/abs/2312.00311) with a larger dataset and switch to more strong backbones or face models. Additionally, we will provide a fast version based on [MobileNet](https://arxiv.org/abs/1905.02244). If you have any suggestions or requirements, please feel free to contact us at wangzidu0705@gmail.com. 




