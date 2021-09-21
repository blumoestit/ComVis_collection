# https://analyticsindiamag.com/semantic-vs-instance-vs-panoptic-which-image-segmentation-technique-to-choose/

# The Mask R_CNN model is trained on Microsoft Coco dataset, a dataset with 80 common object categories.

import pixellib
from pixellib.instance import instance_segmentation
segment_image = instance_segmentation()

segment_image.load_model("mask_rcnn_coco.h5") 

segment_image.segmentImage("test_img_3.jpg", output_image_name = "output_img_3.png")