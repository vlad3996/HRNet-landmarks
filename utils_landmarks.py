import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_five_landmarks_from_net(landmarks):
    """
    Return 5 landmarks needed in face alignment
    """
    num_lmks = landmarks.shape[0]

    if num_lmks == 5:
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        mouse_left = landmarks[3]
        mouse_right = landmarks[4]

    elif num_lmks == 14:
        left_eye = (landmarks[0] + landmarks[1] + landmarks[2] + landmarks[3]) / 4
        right_eye = (landmarks[4] + landmarks[5] + landmarks[6] + landmarks[7]) / 4
        nose = landmarks[8]
        mouse_left = landmarks[10]
        mouse_right = landmarks[11]

    elif num_lmks == 18:
        left_eye = (landmarks[1] + landmarks[2] + landmarks[3] + landmarks[4]) / 4
        right_eye = (landmarks[7] + landmarks[8] + landmarks[9] + landmarks[10]) / 4
        nose = landmarks[12]
        mouse_left = landmarks[14]
        mouse_right = landmarks[15]

    elif num_lmks == 68:
        left_eye = (landmarks[37] + landmarks[38] + landmarks[40] + landmarks[41]) / 4
        right_eye = (landmarks[43] + landmarks[44] + landmarks[46] + landmarks[47]) / 4
        nose = landmarks[30]
        mouse_left = landmarks[48]  # or landmarks[60]
        mouse_right = landmarks[54]  # or landmarks[64]

    elif num_lmks == 98:
        left_eye = landmarks[96]
        right_eye = landmarks[97]
        nose = landmarks[53]
        mouse_left = landmarks[88]
        mouse_right = landmarks[92]

    elif num_lmks == 19:
        left_eye = landmarks[7]
        right_eye = landmarks[10]
        nose = landmarks[13]
        mouse_left = landmarks[15]
        mouse_right = landmarks[17]
    else:
        raise NotImplementedError(f"{num_lmks} not supported !")
    return np.array([left_eye, right_eye, nose, mouse_left, mouse_right])


def set_circles_on_img(image, circles_list, circle_size=5, color=(255, 0, 0), is_copy=False):
    """
    Set circles on image
    Input :
        - image        : numpy array
        - circles_list : [circle1, circle2, ...] or [[circle1_1, circle1_2, ...], [circle2_1, circle2_2, ...]]
        - circle_size  : radius of circle
        - color   : used color for circles
        - is_copy : if False then circles are plotted on the same image. Otherwise copy of image used and returned
    Output :
        - image or copy of image with drawn circles
    """
    temp_image = image.copy() if is_copy else image
    if isinstance(circles_list[0][0], list) or isinstance(circles_list[0][0], np.ndarray):
        for circle_list in circles_list:
            for circle in circle_list:
                cv2.circle(temp_image, (int(np.round(circle[0])), int(np.round(circle[1]))), circle_size, color, -1)
    else:
        for circle in circles_list:
            cv2.circle(temp_image, (int(np.round(circle[0])), int(np.round(circle[1]))), circle_size, color, -1)
    return temp_image


def show_landmarks(img, lmks, circle_size=3, color=(255, 0, 0), figsize=(10,8), is_copy=True):
    """
    Plot landmarks on image
    :param img: source image
    :param lmks: landmarks to chow
    :param circle_size: landmarks size
    :param color: landmarks color
    :param is_copy: plot on source image or use copy
    """
    plt.figure(figsize=figsize)
    plt.imshow(set_circles_on_img(img, lmks, circle_size=circle_size, color=color, is_copy=is_copy))
    plt.show()


def alignment_orig(src_img, src_pts, ncols=96, nrows=112, custom_align=None):
    """
    Original alignment function for MTCNN
    :param src_img: input image
    :param src_pts: landmark points
    :return:
    """
    from matlab_cp2tform import get_similarity_transform_for_cv2

    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]

    if custom_align is not None:
        row[0] += custom_align[0]
        row[1] += custom_align[1]
        
    elif ncols == 112:
        for row in ref_pts:
            row[0] += 8.0
    elif ncols == 128:
        for row in ref_pts:
            row[0] += 16.0

    if nrows == 128 and custom_align is None:
        for row in ref_pts:
            row[1] += 16.0

    # print(ref_pts)

    crop_size = (ncols, nrows)
    src_pts = np.array(src_pts).reshape(5, 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


