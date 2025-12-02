# VapSwin2SR
Hybrid Attention-based Super-resolution for Satellite Imagery Using Vast Receptive Field and Swin Attention to Improve Vehicle Detection (IJIES 2025). [Paper link](https://openurl.ebsco.com/EPDB%3Agcd%3A12%3A21317036/detailv2sid=ebsco%3Aplink%3Ascholar&id=ebsco%3Agcd%3A188702150&crl=c&link_origin=scholar.google.com)

Abdul Aziz Ar Rasyid (azrsyd.id@gmail.com)

>Satellite imagery-based vehicle detection presents a promising approach to support intelligent transportation systems. However, satellite images often suffer from low spatial resolution and external distortions due to long-range imaging. To address these challenges, we propose a hybrid super-resolution framework that integrates Vast Receptive Field Pixel Attention mechanisms with shifted-window attention named VapSwinSR. The proposed model design begins with the VapSR architecture, which utilizes depth-wise dilated convolution, and is then modified by adding a self-attention mechanism. This modification led to an increase in the size of the receptive field, accompanied by a more effective attention weight, resulting in improved super-resolution image quality. Improvements in satellite image resolution have also been shown to improve vehicle detection performance. Experimental results on publicly available satellite imagery datasets including xView and DOTA showed that the proposed method achieved a 2.7% PSNR improvement over Swin Transformer-based methods with fewer parameters, and a 9.95% improvement over classical methods. In the vehicle detection experiment using YOLOv8, the use of super-resolved images resulted in an average increase of 14.26% in mAP50. These findings indicate that improving satellite image resolution not only enhances visual quality but also significantly boosts the effectiveness of vehicle detection, particularly for small objects commonly found in urban traffic environments.

<p align="center">
  <a href="https://arxiv.org/abs/2209.11345"><img src="/framework.jpg" alt="vapswin2sr" width="800" border="0"></a>
</p>
<!-- ![framework.jpg](./framework.jpg) -->
