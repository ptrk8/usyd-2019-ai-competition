import cv2


def display_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_largest_dimension(contours):
    """Accepts an array of contours and returns the dimensions of the contour with the largest area.
    :returns x_coord, y_coord, width, height"""
    contours = map(lambda c: {
        'contour': c,
        'area': cv2.contourArea(c)
    }, contours)
    # Sort contours in descending order by area
    contours = sorted(contours, key=lambda c: c['area'], reverse=True)
    # Get the dimensions of the contour with the largest area
    contour_largest = contours[0]['contour']
    # return dimensions of our bounding rectangle
    return cv2.boundingRect(contour_largest)


def crop_by_contour(img, x_coord, y_coord, width, height):
    return img[y_coord:y_coord + height, x_coord:x_coord + width]


def get_centre_coords(lst_circles):
    # Sort circles
    lst_circles = sorted(lst_circles, key=lambda lst: lst[2], reverse=True)
    return tuple(lst_circles[0])


def get_padding(x_centre, y_centre, radius, img_width, img_height):
    circumference = radius * 2

    top_pad = radius - y_centre
    left_pad = radius - x_centre
    top = top_pad if top_pad > 0 else 0
    left = left_pad if left_pad > 0 else 0

    right_pad = circumference - (img_width + left_pad)
    bottom_pad = circumference - (img_height + top_pad)
    right = right_pad if right_pad > 0 else 0
    bottom = bottom_pad if bottom_pad > 0 else 0

    return int(top), int(bottom), int(left), int(right)


def process_img(path, cv2_mode, img_size):
    img_color = cv2.imread(path, cv2_mode)

    img_grey = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    # Set desired threshold
    thresh = 15
    _, img_binary_contours = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    # Find the contours of image
    contours, _ = cv2.findContours(img_binary_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Get the largest contour
    x_coord, y_coord, width, height = get_largest_dimension(contours)

    img_grey = crop_by_contour(img_grey, x_coord, y_coord, width, height)
    img_color = crop_by_contour(img_color, x_coord, y_coord, width, height)

    thresh = 7
    _, img_binary_circle = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    # Blur the image to allow us to detect circles using Hough Circles
    img_binary_circle = cv2.GaussianBlur(img_binary_circle, (15, 15), cv2.BORDER_DEFAULT)
    # Detect all the centres
    circles = cv2.HoughCircles(img_binary_circle,
                               cv2.HOUGH_GRADIENT,
                               2,
                               img_binary_circle.shape[0] * 1,
                               param1=40, param2=40,
                               minRadius=int(img_binary_circle.shape[0] * 0.4),
                               maxRadius=int(img_binary_circle.shape[0] * 0.7))
    height, width = img_grey.shape
    # if there are no circles then we have a problem
    if isinstance(circles, type(None)):
        x_centre, y_centre, radius = width / 2, height / 2, width / 2
    else:
        # Get the dimensions of the circle with the largest radius
        x_centre, y_centre, radius = get_centre_coords(circles[0])
    # print(circles)
    if radius < height / 2 or radius < width / 2:
        x_centre, y_centre, radius = width / 2, height / 2, width / 2

    top, bottom, left, right = get_padding(x_centre, y_centre, radius, img_color.shape[1], img_color.shape[0])

    img_color = cv2.copyMakeBorder(img_color, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT,
                                   value=img_color[0, 0].tolist())

    return cv2.resize(img_color, (img_size, img_size))