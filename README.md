# wsss_sam

PyTorch implementation of [An Alternative to WSSS? An Empirical Study of the Segment Anything Model (SAM) on Weakly-Supervised Semantic Segmentation Problems](https://arxiv.org/abs/2305.01586)

**This is an inference-only method, no training is needed.**

## Install

You should set the environment variable manually as follows if you want to build a local GPU environment for Grounded-SAM:
```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
```

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install Grounding DINO:

```bash
python -m pip install -e GroundingDINO
```

## Download the pretrained weights

```bash
cd wsss_sam

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```


## Running
Prepare pascal and COCO data.

Set the data path accordingly and run:
```
bash run_coco.sh
```
```
bash run_pascal.sh
```


## Citation

```
@misc{sun2023alternative,
      title={An Alternative to WSSS? An Empirical Study of the Segment Anything Model (SAM) on Weakly-Supervised Semantic Segmentation Problems}, 
      author={Weixuan Sun and Zheyuan Liu and Yanhao Zhang and Yiran Zhong and Nick Barnes},
      year={2023},
      eprint={2305.01586},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
This work is heavily built upon the codebase provided by [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/README.md?plain=1), Thanks for their great code.


