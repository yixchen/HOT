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

# Human part labels  
`./vertex_label_new.npy` contains part labels for SMPL body vertices (10475,).

# Instance-level contact annotations
The contact labels in **HOT v1.0** are similar to the semantic segmentation map, where different instances of contact that belong to the same human part are mixed together. To facilitate relevant tasks and applications, we release the instance-level contact annotations in form of polygons similar to instance segmentation tasks. Example annotations:
```
{
    "imsize": {
        "width": 640,
        "height": 480
    },
    "filename": "train2015_HICO_train2015_00000001",
    "object": [
        {
            "id": 0,
            "semantic_label": "rightfoot",
            "semantic_id": 17,
            "polygon": {
                "x": [
                    207.331283, 210.185029, 216.670814, 229.382954, 250.915761, 257.660978, 258.958135
                    ],
                "y": [
                    266.567323, 275.128563, 280.576622, 289.656719, 298.477387, 301.331133, 300.812278
                    ]
            }
        }
    ]
}

```
Code snippet is attached to process the instance-level annotations to align with the images like the following.
  <p align="center">
<img src="../assets/instance_contact.png" alt="drawing" width="350"/>
</p>