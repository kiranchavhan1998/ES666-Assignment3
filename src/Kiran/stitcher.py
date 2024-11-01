import cv2
import numpy as np
import glob
import os
import random
import math

class PanaromaStitcher:
    def __init__(self, max_size=2000, ratio_threshold=0.75, ransac_threshold=5.0):
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.max_size = max_size
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold

    def make_panaroma_for_images_in(self, path):
        image_paths = sorted(glob.glob(os.path.join(path, '*')))
        print(f'Found {len(image_paths)} images for stitching.')

        images = [cv2.imread(each) for each in image_paths]
        images = [cv2.resize(each,(480,320)) for each in images]        
        images = [self.round_coords(each) for each in images]
        center = int(len(images)/2)
        if len(images) < 2:
            print("At least two images are required to create a panorama.")
            return None, []

        homography_matrix_list = []

        stitched_image = images[center]
        
        for next_image in images[center+1:]:
            H = self.find_homography_ransac(stitched_image, next_image)
            if H is not None:
                stitched_image = self.custom_warp_images(stitched_image, next_image, H)
                homography_matrix_list = np.append(homography_matrix_list,H)
        group1 = stitched_image
        stitched_image = images[center]
        for next_image in images[0:center][::-1]:
            H = self.find_homography_ransac(stitched_image, next_image)
            if H is not None:
                stitched_image = self.custom_warp_images(stitched_image, next_image, H)
                homography_matrix_list = np.append(homography_matrix_list,H)
        group2 = stitched_image
        stitched_image = group1
        next_image = group2

        H = self.find_homography_ransac(stitched_image, next_image)
        if H is not None:
            stitched_image = self.custom_warp_images(stitched_image, next_image, H)
            homography_matrix_list = np.append(homography_matrix_list,H)

        homography_matrix_list = np.array(homography_matrix_list)
        homography_matrix_list = homography_matrix_list.reshape((-1,3,3))

        return stitched_image, homography_matrix_list

    def round_coords(self,InitialImage):
        global w, h, center, f
        h, w = InitialImage.shape[:2]
        center = [w // 2, h // 2]
        f = 600     # 1100 field; 1000 Sun; 1500 Rainier; 1050 Helens

        modified_image = np.zeros(InitialImage.shape, dtype=np.uint8)

        AllCoordinates_of_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
        ti_x = AllCoordinates_of_ti[:, 0]
        ti_y = AllCoordinates_of_ti[:, 1]

        ii_x, ii_y = self.Convert_xy(ti_x, ti_y)

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

        modified_image[ti_y, ti_x, :] = ( weight_tl[:, None] * InitialImage[ii_tl_y,     ii_tl_x,     :] ) + \
                                          ( weight_tr[:, None] * InitialImage[ii_tl_y,     ii_tl_x + 1, :] ) + \
                                          ( weight_bl[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x,     :] ) + \
                                          ( weight_br[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x + 1, :] )


        # Getting x coorinate to remove black region from right and left in the transformed image
        min_x = min(ti_x)

        # Cropping out the black region from both sides (using symmetricity)
        modified_image = modified_image[:, min_x : -min_x, :]

    #     return modified_image, ti_x-min_x, ti_y
        return modified_image
    
    def Convert_xy(self,x, y):
        global center, f

        xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
        yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]

        return xt, yt

    def custom_warp_images(self, img1, img2, H):
        print("Here")
        h1, w1 = img2.shape[:2]
        h2, w2 = img1.shape[:2]
        print(img1.size,img2.size)


        row_number, column_number = int(img2.shape[0]), int(img2.shape[1])
        homography = H
        up_left_cor = self.homogeneous_coordinate(np.dot(homography, [[0],[0],[1]]))
        up_right_cor = self.homogeneous_coordinate(np.dot(homography, [[column_number-1],[0],[1]]))
        low_left_cor = self.homogeneous_coordinate(np.dot(homography, [[0],[row_number-1],[1]]))
        low_right_cor = self.homogeneous_coordinate(np.dot(homography, [[column_number-1],[row_number-1],[1]]))
        corners2 =np.float32([up_left_cor,low_left_cor,low_right_cor,up_right_cor]).reshape(-1, 1, 2)
        all_corners = np.concatenate((corners2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)



        # Create the output image
        t =[-xmin,-ymin]
        Ht = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
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
                point_xy = self.homogeneous_coordinate(np.dot(homography_inverse, [[x], [y], [1]]))
                point_x = int(point_xy[0])
                point_y = int(point_xy[1])

                if (point_x >= 0 and point_x < column_number and point_y >= 0 and point_y < row_number):
                    tmp[y, x, :] = img1[point_y, point_x, :]

        tmp1[t[1]:h2+t[1], t[0]:w2+t[0]] = img2
        cv2.imwrite("tmp10.jpg", tmp1)
        cv2.imwrite("tmp0.jpg", tmp)
        tmp = np.where(np.all(tmp == 0, axis=-1, keepdims=True), tmp1, tmp)
        tmp1 = np.where(np.all(tmp1 == 0, axis=-1, keepdims=True), tmp, tmp1)
        cv2.imwrite("tmp1.jpg", tmp1)
        cv2.imwrite("tmp.jpg", tmp)
        alpha = 0.5
        image = cv2.addWeighted(tmp1, alpha, tmp, 1 - alpha, 0)
        image = image.astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        coords = cv2.findNonZero(gray)  # Returns all non-zero points
        x, y, w, h = cv2.boundingRect(coords)  # Get bounding box from points
        img_final = image[y:y+h, x:x+w]
        img_final = img_final.astype(img1.dtype)
        cv2.imwrite("Cropped_Image.jpg", img_final)
        return img_final 


    def homogeneous_coordinate(self,coordinate):
        x = coordinate[0]/coordinate[2]
        y = coordinate[1]/coordinate[2]
        return x, y

    def apply_homography(self, points, H):
        points_transformed = np.dot(points, H[:2, :2].T) + H[:2, 2]
        points_transformed /= points_transformed[:, 2:]
        return points_transformed[:, :2]

    def calculate_geometric_distance(self,current, previous,h):
        """
        Calculate the geometric distance between two points p1 and p2.
        :param p1: The first point (x, y) in transformed space.
        :param p2: The second point (x, y) in target space.
        :return: Euclidean distance between the points.
        """
        p1 = np.transpose(np.matrix([current[0], current[1], 1]))
        estimatep2 = np.dot(h, p1)
        estimatep2 = (1/estimatep2.item(2))*estimatep2
    
        p2 = np.transpose(np.matrix([previous[0], previous[1], 1]))
        error = p2 - estimatep2
        return np.linalg.norm(error)

    def hom_calc(self,current,previous):
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
    def ransac(self,current,previous):
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
            
            a_vals = []
            for i in range(len(currFour)):
                p1 = np.matrix([currFour[i][0],currFour[i][1],1])
                p2 = np.matrix([preFour[i][0],preFour[i][1], 1])

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
            h = np.reshape(v[8], (3, 3))

            #normalize and now we have h
            h = (1/h.item(8)) * h
            inliers_curr = []
            inliers_prev =[]
            for i in range(len(current)):
                d = self.calculate_geometric_distance(current[i],previous[i], h)
                if d < 10:
                    inliers_curr.append([current[i][0],current[i][1]])
                    inliers_prev.append([previous[i][0],previous[i][1]])

            if len(inliers_curr) > len(maxInliers_curr):
                maxInliers_curr = inliers_curr
                maxInliers_prev = inliers_prev
                finalH = h

            if len(maxInliers_curr) > (len(current)*self.ransac_threshold):
                break
        return finalH


    def find_homography_ransac(self, img1, img2):
        keypoints1, descriptors1 = self.detector.detectAndCompute(img1, None)
        keypoints2, descriptors2 = self.detector.detectAndCompute(img2, None)
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for match1, match2 in matches:
            if match1.distance < self.ratio_threshold * match2.distance:
                good_matches.append(match1)

        if len(good_matches) >= 4:
            src_points = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
            dst_points = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
            # print("Before")
            # print(src_points.shape,src_points)
            src_points = np.squeeze(src_points)
            dst_points = np.squeeze(dst_points)
            # print("After")
            # print(src_points.shape,src_points)
            H = self.ransac(src_points,dst_points)
            #print("Homography:",H)
            return H



# # Example usage
# stitcher = PanoramaStitcher()
# image_path = 'path_to_your_images'
# stitched_image, homography_matrices = stitcher.make_panorama_for_images_in(image_path)
# cv2.imshow('Stitched Image', stitched_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()