# Data structure
Please download the HOT dataset following instruction from the [project website](https://hot.is.tue.mpg.de/). The `images` folder contains raw images, the `annotations` folder contains contact label images, and the `segments` folder contains body segmentation images to facilitate the training process.

Note that currently the HOT dataset annotations are released at image-level, not instance level, similar to semantic and panoptic segmentation in scene understanding.

```
|-- Dataset Root
|	|-- HOT-Annotated
|		|-- images
|		|-- annotations
|		|-- segments
|	|-- HOT-Generated
|		|-- images
|		|-- annotations
|		|-- segments
```

# Contact and human-part label semantics
{'Head': 1, 'Chest': 2, 'Back': 3, 'leftUpperArm,': 4, 'leftForeArm': 5, 'LeftHand': 6, 'rightUpperArm': 7,
 'rightForeArm': 8, 'rightHand': 9, 'Butt': 10, 'Hip': 11, 'leftThigh': 12, 'leftCalf': 13, 'leftFoot': 14,
 'RightThigh': 15, 'rightCalf': 16, 'rightFoot': 17}

# Human Part Labels  
`./vertex_label_new.npy` contains part labels for SMPL body vertices (10475,).
