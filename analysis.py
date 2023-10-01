from glob import glob
from PIL import Image
import pandas as pd
import random
import cv2

def get_annotations_list(img, df):
    """show all the bounding box of given image"""
    image = cv2.imread(img)
    h, w, c = image.shape
    print(f"image size {image.shape}")
    idx = img.split('.')[0].split('/')[1]
    df = df.loc[df['ImageID']==idx]
    print(f"total number of annotation in givem images are {df.shape}")
    XMin = df['XMin'].to_list()
    XMax = df['XMax'].to_list()
    YMin = df['YMin'].to_list()
    YMax = df['YMax'].to_list()
    labels = df['LabelName'].to_list()
    annotation_list = [(int(xmin*w), int(ymin*h), int(xmax*w),int(ymax*h))
                       for xmin, ymin, xmax, ymax in zip(XMin, YMin, XMax, YMax)]
    for i, box in enumerate(annotation_list):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        label = labels[i]
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

    cv2.imshow("image with bounding box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    images = glob('train_0/*')
    print(f"Total number of images are {len(images)}")

    df = pd.read_csv('train-annotations-bbox.csv')
    print(f"shape of label file is {df.shape}")

    print(f"df columns: {df.columns}")
    print(f"total unique image id are {df['ImageID'].nunique()}")


    # select a random image
    idx = random.randint(0, len(images))
    img = images[idx]
    #print(f"selected image is {img}")
    #print(f"idx is {img.split('.')[0].split('/')[1]}")
    idx = img.split('.')[0].split('/')[1]
    #print(df[df['ImageID']=="000002b66c9c498e"])
    #print(df[df['ImageID']==idx])


    print(get_annotations_list(img, df))
