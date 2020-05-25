## Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks

One of the challenges in modeling cognitive events from electroencephalogram (EEG) data is finding representations that are invariant to inter- and intra-subject differences, as well as to inherent noise associated with such data. A novel approach for learning such representations from multi-channel EEG time-series, and demonstrate its advantages in the context of mental load classification task is proposed. 

![](images/train.png)

First, we transform EEG activities into a sequence of topology-preserving multi-spectral images, as opposed to standard EEG analysis techniques that ignore such spatial information. Next, we train a deep recurrent-convolutional network inspired by state-of-the-art video classification to learn robust representations from the sequence of images. The proposed approach is designed to preserve the spatial, spectral, and temporal structure of EEG which leads to finding features that are less sensitive to variations and distortions within each dimension.

#
![](images/transform.png)
(FIG.)Topology-preserving and non-topology-preserving projections of electrode locations.  A)2-D projection of electrode locations using non-topology-preserving simple orthographic projection.B) Location of electrodes in the original 3-D space.  C) 2-D projection of electrode locations usingtopology-preserving azimuthal equidistant projection.
#

## Keynote
These implementations have not been checked by original authors, only reimplemented from the paper description and open source code from original authors.

## Learning Models
* First try Convolution Neural Net (CNN)
* New is a 3D convolutional NN (CNN3D) in the frequency domain
* Then try Long-Short Term Memory Recurrent Neural Net (LSTM)
* Finally do, Mix-LSTM / 1D-Conv

![](images/model_architecture.png)
# 

## Results


## Reference
Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

http://arxiv.org/abs/1511.06448
