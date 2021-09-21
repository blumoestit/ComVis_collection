# https://analyticsindiamag.com/semantic-vs-instance-vs-panoptic-which-image-segmentation-technique-to-choose/
# Load the xception model trained on pascal voc for segmenting objects. The model can be downloaded from github.
# deeplabv3_xception_tf_dim_ordering_tf_kernels.h5 

import pixellib
from pixellib.semantic import semantic_segmentation

segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5") 
segment_image.segmentAsPascalvoc("test img.png", output_image_name = "test output.png")





