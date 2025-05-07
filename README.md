# CFRNet: Road Extraction in Remote Sensing Images Based on Cascade Fusion Network
## The Visual Feature Maps of Different Scales
![Figure_1](https://github.com/XYQ1517/CFRNet/assets/104625070/d0935814-5197-4494-bb58-6c5a3c4dbdc4)
We compared the feature maps of sub-backbone and the fused feature map, 
as shown in the figure. By examining the feature maps, we can observe that 
features at different scales contain distinct information. The fusion of 
these multi-scale features aids us in more effectively extracting road details. 
In the fused feature map, we can clearly see the overall contour of the road.


## The Pseudo-code of CFRNet

![image](https://github.com/XYQ1517/CFRNet/assets/104625070/2c3a905a-3f2d-40c4-878c-6a9a1befdc74)

## Weight

The weight can be found here: https://pan.baidu.com/s/1G6L-JZhjMRHM0ftaWgrJXQ?pwd=ptmn
Note that when testing the CFRNet weight on the DeepGlobe, you need to change all the "TAFF" in CFRNet.py to "MSAF".

## Cite 

@ARTICLE{10549925,
  author={Xiong, Youqiang and Li, Lu and Yuan, Di and Wang, Haoqi and Ma, Tianliang and Wang, Zhongqi and Yang, Yuping},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={CFRNet: Road Extraction in Remote Sensing Images Based on Cascade Fusion Network}, 
  year={2024},
  volume={21},
  number={},
  pages={1-5},
  keywords={Feature extraction;Roads;Remote sensing;Data mining;Convolution;Fuses;Semantics;Cascad fusion network;multiscale;road extraction;sub-backbone;triple-level adaptive feature fusion (TAFF) module},
  doi={10.1109/LGRS.2024.3409758}
  }
