import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

import albumentations
from PIL import Image

import imageio



def returnImageMask(image_path, augment=False, preprocess=False):
    mask_path = image_path.split(".")[0] + "_mask.tif"

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(type(mask))

    if augment:
        transform = [
            albumentations.Resize(height=224, width=224, interpolation=Image.BILINEAR),
        ]
        transform_pipeline = albumentations.Compose(transform)
        sample = transform_pipeline(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]

    if preprocess:
        transform = []
        transform.append(albumentations.Lambda(image=convert_to_tensor))
        transform_pipeline = albumentations.Compose(transform)

        sample = transform_pipeline(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]

    mask = (mask / 255).astype(np.float32)
    mask = np.expand_dims(mask, axis=0)

    return image, mask

def convert_to_tensor(x,**kwargs):
    return x.transpose(2,0,1).astype("float32")

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if image.shape[0] == 3:
            image = image.transpose([1, 2, 0])
        plt.imshow(image)
    plt.show()

image_path = "demo/train/1_1.tif"

device = "cuda"
best_model = torch.load('./best_model.pth')

def predict_return_ndarray(image_path):
    image_vis, mask_vis = returnImageMask(image_path, False, False)
    image_vis = image_vis.astype('uint8')
    mask_vis = mask_vis.astype('uint8')

    image, gt_mask = returnImageMask(image_path, True, True)

    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = pr_mask.squeeze().cpu().numpy().round()

    kernel = np.ones((5, 5), np.uint8)
    pr_mask_er = cv2.erode(pr_mask, kernel, iterations=4)
    pr_mask_er = cv2.dilate(pr_mask_er, kernel, iterations=4)

    pr_mask = cv2.resize(pr_mask, (580, 420))
    pr_mask_er = cv2.resize(pr_mask_er, (580, 420))

    mask_vis = mask_vis.squeeze()

    return image_vis, mask_vis, pr_mask, pr_mask_er


def pred_mask(x):
    x = x.round()
    kernel = np.ones((8, 8), np.uint8)
    x = cv2.erode(x, kernel, iterations=4)
    x = cv2.dilate(x, kernel, iterations=4)
    x = cv2.resize(x, (580, 420))

    return x



