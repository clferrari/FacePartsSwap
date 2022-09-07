# What makes you, you? Analyzing recognition by swapping face parts.

Official implementation of the paper "What makes you, you? Analyzing recognition by swapping face parts", appeared at IEEE 26th International Conference on Pattern Recognition (ICPR) 2022.

[[ArXiv](https://arxiv.org/pdf/2206.11759.pdf)]

![alt text](https://github.com/clferrari/FacePartsSwap/blob/master/conf/method.png)

## Changelog

07 Sept. 2022 - Repository created

### Requirements

Tested with Python 3.7. Extra packages required are Shapely, face_alignment, spicy.

### Usage

Simply run main.py. The following options can be set:

--src_path : path of the source face image
--dst_path : path of the target face image
--part : part to be swapped (face, nose, eyes, mouth, eyebrows)
--debug : if True, saves intermediate results
--cropImg : if the face needs to be detected and cropped, this can be set to True

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