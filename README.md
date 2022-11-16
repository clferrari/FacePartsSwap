# What makes you, you? Analyzing recognition by swapping face parts.

Official implementation of the paper "What makes you, you? Analyzing recognition by swapping face parts", appeared at IEEE 26th International Conference on Pattern Recognition (ICPR) 2022.

[[ArXiv](https://arxiv.org/pdf/2206.11759.pdf)]

![alt text](https://github.com/clferrari/FacePartsSwap/blob/main/conf/method.png)

## Changelog

16 Nov. 2022 - Bug fix: solved gradient problem for misaligned swapping masks.

12 Sep. 2022 - Added missing files and requirements.

07 Sep. 2022 - Repository created.

### Requirements

Tested with Python 3.7. Extra packages required are Shapely, face_alignment (https://pypi.org/project/face-alignment/), scipy, matplotlib (ver 3.0).

The code and model to segment the face are borrowed from https://github.com/zllrunning/face-parsing.PyTorch. The pre-trained model can be found in res/cp folder.

### Usage

First download the auxiliary 3D data required from this [[link](https://drive.google.com/file/d/1ZL66ZWivvNKZ-gcBIDmjbmzpMf_gimOi/view?usp=sharing)] and place it into `3D_data' folder.

Simply run main_c.py to use the default options. The following can be set:

--src_path : path of the source face image

--dst_path : path of the target face image

--part : part to be swapped (face, nose, eyes, mouth, eyebrows)

--debug : if True, saves intermediate results

--cropImg : if the face needs to be detected and cropped, this can be set to True


### Limitations

Some known limitations of the approach, as described in the paper, are:

1. In case of landmark detection errors or strong pose differences, the method can fail in performing the swap, mostly for the entire face. 

2. Difficulty in handling eyeglasses

Help us making the method more robust by reporting any issues you find using our code!

### Citation

If you find this work useful, please cite us!!

```
@article{ferrari2022makes,
  title={What makes you, you? Analyzing Recognition by Swapping Face Parts},
  author={Ferrari, Claudio and Serpentoni, Matteo and Berretti, Stefano and Del Bimbo, Alberto},
  journal={arXiv preprint arXiv:2206.11759},
  year={2022}
}
```

```
@article{ferrari2017dictionary,
  title={A dictionary learning-based 3D morphable shape model},
  author={Ferrari, Claudio and Lisanti, Giuseppe and Berretti, Stefano and Del Bimbo, Alberto},
  journal={IEEE Transactions on Multimedia},
  volume={19},
  number={12},
  pages={2666--2679},
  year={2017},
  publisher={IEEE}
}

```

### License

The software is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.(see LICENSE).

### Contacts

For any inquiry, feel free to drop an email to ferrari.claudio88@gmail.com