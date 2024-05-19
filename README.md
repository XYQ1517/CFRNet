# CFRNet
## The Visual Feature Maps of Different Scales
![Figure_1](https://github.com/XYQ1517/CFRNet/assets/104625070/d0935814-5197-4494-bb58-6c5a3c4dbdc4)
We compared the feature maps of sub-backbone and the fused feature map, 
as shown in the figure. By examining the feature maps, we can observe that 
features at different scales contain distinct information. The fusion of 
these multi-scale features aids us in more effectively extracting road details. 
In the fused feature map, we can clearly see the overall contour of the road.


## The Pseudo-code of CFRNet


  Input: Remote Sensing Image
  Output: Predicted Road Image
  1:	x_0← MbBlock(Stem(Input))
  2:	x_1 ← Downsample(x_0)
  3:	for i ← 1 to 3 do : // three CFRNet stages in Model
  4:		x_1^' ← MbBlock(x_1)
  5:		x_2^' ← MbBlock(Downsample(x_1^'))
  6:		x_3^' ← MbBlock(Downsample(x_2^'))
  7:		TAFF_out ← TAFF(x_1^', x_2^', x_3^')
  8:		x_1 ← x_1 + TAFF_out
  9:	end for
  10:	x_1^'' ← MbBlock(Upsample(x_1))
  11:	Output ← Pred (MbBlock(Concat(x_0, x_1^''))
