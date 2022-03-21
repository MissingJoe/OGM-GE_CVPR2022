# OGM-GE implemented in PyTorch

Here is the source code of OGM-GE training strategy proposed in *Balanced Multimodal Learning via On-the-fly Gradient Modulation*, which is a easy way to apply to different vanilla multimodal fusion networks or other multimodal networks, helping to achieve a more balanced and adequate training. More analysis and details can be seen in the paper.

This code includes four main dirs: 
'data to store data of database; 
'dataloader' that including three dataloaders(audi_visual,audio,visual); 
'models' that including vanilla audio_visual models; 
'tempresult' to save more temporary result, 
and we also provid pre-processing, training, testing, and tools in addition.

**Paper Title: "Balanced Multimodal Learning via On-the-fly Gradient Modulation"
Authors: Xiaokang Peng*, Yake Wei*, Andong Deng, Dong Wang, and Di Hu 
Accepted by: IEEE Conference on Computer Vision and Pattern Recognition(CVPR 2022)
The link of paper will be released soon**


## What's the imbalance phenomenon in multimodal learning task?
We observe that potential of multimodal information is not fully exploited even when the multimodal model outperforms its uni-modal counterpart. We conduct linear probing experiments to explore the quality of jointly trained encoders, and find them under-optimized (the yellow line) compared with the uni-modal model (the red line). We proposed the OGM-GE method to improve the optimization process adaptively and achieved consistent improvement (the blue line). We improve both the multimodal performance and uni-model representation as shown in the figure.
<div  align="center">    
<img src="demo/five lines.PNG" width = "80%" />
</div>

## Method Introduction
Pipeline of our OGM-GE method, consisting two steps:
1. On-the-fly gradient modulation for different modalities.
2. Adaptive Gaussion noise enhancement on gradient.
<div  align="center">    
<img src="demo/pipeline.PNG" width = "80%" />
</div>


## Main Dependencies
+ Ubuntu 16.04
+ CUDA Version: 11.1
+ PyTorch 1.8.1
+ torchvision 0.9.1
+ python 3.7.6


## Data Preparation
### Original dataset
Original Dataset：
[VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)
[Kinetics-Sounds](https://github.com/cvdfoundation/kinetics-dataset)
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
[AVE](https://sites.google.com/view/audiovisualresearch)
[ESC50](https://github.com/karoldvl/ESC-50/archive/master.zip)



### Pre-processing

For example, we provide code to pre-process videos into RGB streams at different frame rates and audio wav files in the file ```'tools'```. By running python files with ```'python preprocess_audio.py'``` and ```'python video2frame-1fps.py'```. The only thing you need to change is the data or other file addresses.

You can also adjust the frame rates by changing the parameter in the ```'preprocess_audio.py'``` file, or the in the ```'video2frame-1fps.py'``` file.


After downloading and processing data, you should build the data dir  following proper structure. We give an example of AVE dataset:
```
AVE
│------ visual
│------ audio
│--- other files
```

&nbsp;



### Train the model

Before training, you should new a directory named **`model`** to save checkpoint file. 
```python train_amd.py```
You can change related settings in file ```train_amd.py```.
&nbsp;


### Test and Eval

You can test the performance of trained model by simply running
```python test.py```
And you can also learn more information(mAP, and so on) by running
```python eval.py```


### Citations
The citation of our paper will be released soon.

### References



### Contact us

If you have any detailed questions or suggestions, you can email us:
**xiaokangpeng@ruc.edu.cn**
