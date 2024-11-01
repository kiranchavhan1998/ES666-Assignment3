import pdb
import glob
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import random
import math

sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm =0, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def match(i1,i2,direction = None):
    imageSet1 = get_sift_corners(i1)
    imageSet2 = get_sift_corners(i2)
    matches = flann.knnMatch(
        imageSet2['des'],
        imageSet1['des'],
        k =2)
    good_corners =[]
    for i,(m,n) in enumerate(matches):
        if(m.distance) < (0.5*n.distance):
            good_corners.append((m.trainIdx,m.queryIdx))
    if len(good_corners) >=10:
        pointsCurrent = imageSet2['kp']
        pointsPrevious = imageSet1['kp']

        matchedPointsCurrent = np.float32(
            [pointsCurrent[i].pt for (_,i) in good_corners]
            )
        matchedPointsPrev = np.float32(
            [pointsPrevious[i].pt for (i,_) in good_corners]
            )
        H,inliers_curr,inliers_prev = ransac(matchedPointsCurrent,matchedPointsPrev,4)
        H = hom_calc(inliers_curr,inliers_prev)
        return H
    return None

def hom_calc(current,previous):
    a_vals = []
    for i in range(len(current)):
        p1 = np.matrix([current[i][0],current[i][1],1])
        p2 = np.matrix([previous[i][0],previous[i][1], 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        a_vals.append(a1)
        a_vals.append(a2)

    matrixA = np.matrix(a_vals)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    H = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    H = (1/H.item(8)) * H
    return H

def ransac(current,previous, thresh):
    maxInliers_curr, maxInliers_prev =[],[]
    finalH = None
    random.seed(2)
    for i in range(1000):
        currFour = np.empty((0, 2))
        preFour = np.empty((0,2))
        for j in range(4):
            random_pt = random.randrange(0, len(current))
            curr = current[random_pt]
            pre = previous[random_pt]
            currFour = np.vstack((currFour,curr))
            preFour = np.vstack((preFour,pre))


        #call the homography function on those points
        h = hom_calc(currFour,preFour)
        inliers_curr = []
        inliers_prev =[]
        for i in range(len(current)):
            d = geo_dist(current[i],previous[i], h)
            if d < 10:
                inliers_curr.append([current[i][0],current[i][1]])
                inliers_prev.append([previous[i][0],previous[i][1]])

        if len(inliers_curr) > len(maxInliers_curr):
            maxInliers_curr = inliers_curr
            maxInliers_prev = inliers_prev
            finalH = h

        if len(maxInliers_curr) > (len(current)*thresh):
            break

    return finalH, maxInliers_curr,maxInliers_prev


def geo_dist(current,previous, h):

    p1 = np.transpose(np.matrix([current[0], current[1], 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([previous[0], previous[1], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)

def get_sift_corners(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    kp,des = sift.detectAndCompute(gray,None)
    return {'kp':kp,'des':des}


def homo_coords(coordinate):
    x = coordinate[0]/coordinate[2]
    y = coordinate[1]/coordinate[2]
    return x, y


def wrap(a,b):
    H = match(a,b,"left")
    if H is None:
        return a , H
    h1, w1 = b.shape[:2]
    h2, w2 = a.shape[:2]

    row_number, column_number = int(b.shape[0]), int(b.shape[1])
    homography = H
    up_left_cor = homo_coords(np.dot(homography, [[0],[0],[1]]))
    up_right_cor = homo_coords(np.dot(homography, [[column_number-1],[0],[1]]))
    low_left_cor = homo_coords(np.dot(homography, [[0],[row_number-1],[1]]))
    low_right_cor = homo_coords(np.dot(homography, [[column_number-1],[row_number-1],[1]]))
    corners2 =np.float32([up_left_cor,low_left_cor,low_right_cor,up_right_cor]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    Hnew = Ht.dot(H)
    homography = Hnew

    offset_x = math.floor(xmin)
    offset_y = math.floor(ymin)

    max_x = math.ceil(xmax)
    max_y = math.ceil(ymax)

    size_x = max_x - offset_x
    size_y = max_y - offset_y

    dsize = [size_x,size_y]
    homography_inverse = np.linalg.inv(homography)

    tmp = np.zeros((dsize[1], dsize[0], 3))
    tmp1= np.zeros((dsize[1],dsize[0],3))

    for x in range(size_x):
        for y in range(size_y):
            point_xy = homo_coords(np.dot(homography_inverse, [[x], [y], [1]]))
            point_x = int(point_xy[0])
            point_y = int(point_xy[1])

            if (point_x >= 0 and point_x < column_number and point_y >= 0 and point_y < row_number):
                tmp[y, x, :] = b[point_y, point_x, :]

    tmp1[t[1]:h2+t[1], t[0]:w2+t[0]] = a
    tmp = np.where(np.all(tmp == 0, axis=-1, keepdims=True), tmp1, tmp)
    tmp1 = np.where(np.all(tmp1 == 0, axis=-1, keepdims=True), tmp, tmp1)
    img_final = tmp.astype(a.dtype)
    return img_final , H

def stitching_func(images_list,centerIdx):
    a = images_list[centerIdx]
    hom = []
    for b in images_list[centerIdx+1:]:
        a , homo = wrap(a,b)
        if homo is not None:
            hom = np.append(hom,homo)
    group1 = a
    a = images_list[centerIdx]
    for b in images_list[0:centerIdx][::-1]:
        a, homo= wrap(a,b)
        if homo is not None:
            hom = np.append(hom,homo)
    group2 = a
    a = group1
    b = group2
    result,homo = wrap(a,b)
    if homo is not None:
        hom = np.append(hom,homo)
    leftImage = result
    homog = hom
    hom = np.array(hom)
    hom = hom.reshape((-1,3,3))
    return result,hom

def Convert_xy(x, y):
    global center, f

    xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
    yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]

    return xt, yt


def change_xy_to_cylindrical(start_image):
    global w, h, center, f
    h, w = start_image.shape[:2]
    center = [w // 2, h // 2]
    f = 600     # 1100 field; 1000 Sun; 1500 Rainier; 1050 Helens

    updated_image = np.zeros(start_image.shape, dtype=np.uint8)

    AllCoordinates_of_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = AllCoordinates_of_ti[:, 0]
    ti_y = AllCoordinates_of_ti[:, 1]

    ii_x, ii_y = Convert_xy(ti_x, ti_y)

    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)

    GoodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w-2)) * \
                    (ii_tl_y >= 0) * (ii_tl_y <= (h-2))

    ti_x = ti_x[GoodIndices]
    ti_y = ti_y[GoodIndices]

    ii_x = ii_x[GoodIndices]
    ii_y = ii_y[GoodIndices]

    ii_tl_x = ii_tl_x[GoodIndices]
    ii_tl_y = ii_tl_y[GoodIndices]

    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx)       * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx)       * (dy)

    updated_image[ti_y, ti_x, :] = ( weight_tl[:, None] * start_image[ii_tl_y,     ii_tl_x,     :] ) + \
                                        ( weight_tr[:, None] * start_image[ii_tl_y,     ii_tl_x + 1, :] ) + \
                                        ( weight_bl[:, None] * start_image[ii_tl_y + 1, ii_tl_x,     :] ) + \
                                        ( weight_br[:, None] * start_image[ii_tl_y + 1, ii_tl_x + 1, :] )


    # Getting x coorinate to remove black region from right and left in the transformed image
    min_x = min(ti_x)

    # Cropping out the black region from both sides (using symmetricity)
    updated_image = updated_image[:, min_x : -min_x, :]

#     return updated_image, ti_x-min_x, ti_y
    return updated_image