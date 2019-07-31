# 'halp' Team Submission for USYD Challenge

## Install

To run our ensemble, simply run the `predict.py` file as shown below.

```bash
$ python predict.py /path/to/input/files
```

## Things we tried

### Image preprocessing

Image preprocessing involved the following steps.
1. Crop excess black pixels/space (usually on the left and right of the fundoscope image) by detecting the image's contours using OpenCV's `findContours` method.
2. Detect the radius and centre of the fundoscope's circle using OpenCV's `HoughCircles` function.
3. Pad the image accordingly so the circle detected in step 2 is centred.

The result is an image that maintains the original aspect ratio and ensures the circle is centre.
![Imgur](https://i.imgur.com/n0pLj0a.png)

We also experimented with the following preprocessing filters:

- Gaussian Filter
```python
def subtract_gaussian_blur(img):
    gb_img = cv2.GaussianBlur(img, (0, 0), 5)
    return cv2.addWeighted(img, 4, gb_img, -4, 128)
```
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
```python
def get_clahe_img(img, clip_limit, grid_size):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab = lab.astype(np.uint8)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img
```
- Grey Scale Images

However, we settled with standard RGB images since we didn't observe significant improvements using the above filters.

### Image Sizes
We experimented with image height and widths of 256, 384 and 512 pixels and observed a positive relationship between image size and our model's predictive power so we settled on using images of 512 pixels high and wide.

### Models Used




### Optimizers we experimented with
- Stochastic Gradient Descent (with and without momentum)
- Adam
- Adagrad

Conclusion is that SGD typically converged to a more globally optimal solution than dynamic optimizers like Adam.

### Learning rate schedule

We experimented with constant learning rates ranging from 0.1 to 0.00001 as well as scheduled learning rates like the one below. We settled on the following learning rate schedule since our solution uses models with pre-trained weights, justifying our lower initial learning rate.

```python
def lr_schedule(epoch):
    lr = 0.0003
    if epoch > 25:
        lr = 0.00015
    if epoch > 30:
        lr = 0.000075
    if epoch > 35:
        lr = 0.00003
    if epoch > 40:
        lr = 0.00001
    return lr
```
