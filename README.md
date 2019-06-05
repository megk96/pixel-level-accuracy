# Mask R-CNN Pixel Level Accuracy for Each Class

Considering an open-source Mask R-CNN implementation, which is a deep multitask learning solution for object detection, instance segmentation and keypoint annotations. The current task involves given an annotated instance segmentation image and the original image, the pixel level accuracy of each instance/class from the image is calculated. This implementation does not contain any fine-tuning to support extra classes than what the model has been trained for initially.

## Implementation
This implementation uses ``matterport's Mask_RCNN API``. 
### Installation
Clone the API's repository with
```
$ git clone https://github.com/matterport/Mask_RCNN.git
```
Change to the ``Mask_RCNN`` repository with 
```
$ cd Mask_RCNN
```

Inside the ``Mask_RCNN`` folder, clone the ``pixel-level-accuracy`` repository with 
```
$ git clone https://github.com/megk96/pixel-level-accuracy.git
```
Change to the ``pixel-level-accuracy`` repository with 
```
$ cd pixel-level-accuracy
```

Make sure you have the Jupyter Notebook as the next step is an ipynb implementation for visualization, and directly involves using the API. Can be installed with
```
$ python3 -m pip install --upgrade pip
$ python3 -m pip install jupyter
```

### Running Locally
The main result from the API is obtained from the ``save-instances.ipynb``. 
```
Rub all blocks of save-instances.ipynb. 
```
This generates the file ``instanses.json`` in the pwd. 
This could not be uploaded on the repository due to its large size. But it is an essential requirement before running the main code.
```
$ python pixel-level.py
```
This generates individual annotation masks for each class or instance and stores it in ``individual_masks`` folder. The results of this step can already be found in the repository. 
The final required output in json format of per-instance pixel-level accuracy can be found in ``output.json`` which is sorted by descended order of accuracies. 

## Methodology
### Instance Masks
Instance Masks of all the predictions of the pre-trained model are obtained and stored as ``instanses.json`` which contain:
1. Binary Instance masks for each prediction. 
2. Bounding box values of corresponding region of interest. 
3. The class they belong to. 

### Annotation Masks
The annotation makes are extracted from ``annotated_image.png``. 
1. They are separated by color and correspondingly stored as individual binary masks. 
2. The bounding box values are obtained for each mask. 
3. The corresponding class is stored. 

### IoU: Intersection over Union
The challenge while devising your own evaluation metric is corresponding each annotation to its possible instance predicted by the model. 
This is done by:
1. Filtering them out by matching the instance class and the annotation class. Eg "Car #6" = "car"
2. Calculating the IoU values of the areas of the corresponding bounding boxes. 
3. For each annotation, the instance which has the biggest overlap is the most probable match. So the largest IoU value is obtained.

### Pixel Accuracy
Now that the correct match of instance and annotation is obtained, the overall accuracy is calculated. 
1. Accuracy is defined as accuracy = TP+TN / TP+TN+FP+FN. The problem here is, the True Negative (TN) is extremely high when you consider the whole image, and as the image becomes smaller, very high accuracies are reported, but they do not correspond to the model's capability accurately. 
2. To reduce the effect of unnecessary, trivial true negatives, a smaller bounding box is considered instead of the entire image. 
3. This bounding box is obtained as essentially the union of both the annotation and instance mask bounding boxes. 

This, in my opinion, is a more accurate representation of fine-grained pixel level accuracy. 
Of course, this can easily be changed to the whole image, by changing the limits where the accuracy is calculated. 
