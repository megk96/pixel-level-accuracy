from PIL import Image
import json
import webcolors
import os
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import collections

#This function is used to create individual masks from the RGB mask of the annotations.
def create_individual_masks(mask):
    width, height = mask.size
    with open('legend.json', 'r') as f:
        legend = json.load(f)
        legend = legend['legend']
    masks = {}
    for x in range(width):
        for y in range(height):
            pixel = mask.getpixel((x, y))[:3]
            color = webcolors.rgb_to_hex(pixel)
            instance_class = legend[str(color)]
            instance_mask = masks.get(instance_class)
            if instance_mask is None:
                masks[instance_class] = Image.new('1', (width, height))
            masks[instance_class].putpixel((x, y), 1)

    return masks

# This function extracts the bounding box coordinates of the annotation masks and returns a dictionary. 
def create_mask_annotation(mask, name):
    contours = measure.find_contours(mask, 0.5, positive_orientation='low')
    polygons = []
    for contour in contours:
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)

    multi_poly = MultiPolygon(polygons)
    y1, x1, y2, x2 = multi_poly.bounds
    bbox = [x1, y1, x2, y2]

    annotation = {
        'bbox': bbox,
        'category': name,
        'mask': mask
    }

    return annotation

#This function finds the IoU values given two bounding boxes. 
def find_IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

#This function finds the pixel level accuracy
def find_pixel_accuracy(annotation, instances):
    maxIoU = 0
    index = -1
#The correct match of annotation-instance is found based on maximum overlap and match of category. 
    for i, instance in enumerate(instances):
        word = annotation['category']
        word = word.split(' ')[0]
        if word.lower() == instance['class']:
            IoU = find_IoU(annotation['bbox'], instance['bbox'])
            if IoU > maxIoU:
                maxIoU = IoU
                index = i
                print(IoU)
                print(index)
                
#If match found, the pixel accuracy is calculated.                
    if index != -1:
#To reduce the effect of unnecessary, trivial true negatives, a smaller bounding box is considered instead of the entire image.
        instance = instances[index]
        boxA = annotation['bbox']
        boxB = instance['bbox']
        print(boxA)
        print(boxB)
        x1 = int(min(boxA[0], boxB[0]))
        y1 = int(min(boxA[1], boxB[1]))
        x2 = int(max(boxA[2], boxB[2]))
        y2 = int(max(boxA[3], boxB[3]))
#This bounding box is obtained as essentially the union of both the annotation and instance mask bounding boxes.        
        annotation_mask = np.asarray(annotation['mask'])
        instance_mask = np.asarray(instance['mask'])
        print(annotation_mask.shape)
        print(instance_mask.shape)
        total = 0
        correct_pred = 0
        print(x1, y1, x2, y2)
        for x in range(x1, x2):
            for y in range(y1, y2):
                total += 1
                annotation_pixel = annotation_mask[x, y]
                instance_pixel = instance_mask[x, y]
                if annotation_pixel == instance_pixel:
                    correct_pred += 1
                print(float(correct_pred / total))
        return float(correct_pred / total)
    else:
        return 0


def main():
    rgb_mask = Image.open("annotated_image.png")
    masks = create_individual_masks(rgb_mask)

    for mask in masks:
        masks[mask].save(os.path.join('individual_masks', mask + '.png'))

    annotations = []

    for name, mask in masks.items():
        annotation = create_mask_annotation(mask, name)
        annotations.append(annotation)

    with open('instances.json') as json_file:
        instances = json.load(json_file)

    output = {}

    for annotation in annotations:
        print(annotation['category'])
        output[annotation['category']] = find_pixel_accuracy(annotation, instances)

    output = collections.OrderedDict(sorted(output.items(), key=lambda x: x[1], reverse=True))
    with open('output.json', 'w') as outfile:
        json.dump(output, outfile, indent=4)


if __name__ == '__main__':
    main()
