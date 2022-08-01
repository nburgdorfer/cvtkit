import numpy as np
import os
import sys
import cv2

def load_intrinsics(intrins_file):
    with open(intrins_file, 'r') as f:
        lines = f.readlines()
        K = []

        for l in lines:
            line = l.strip().split(' ')
            row = []

            for elem in line:
                row.append(float(elem))

            K.append(row)

    K = np.asarray(K)

    return K


def match_features(query_img, train_img, max_features=500):
    query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
      
    orb = cv2.ORB_create(max_features)
      
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = list(matcher.match(queryDescriptors,trainDescriptors) )
    matches.sort(key = lambda x:x.distance)

    pts1 = []
    pts2 = []
    for i in range(8):
        m = matches[i]

        pts2.append(trainKeypoints[m.trainIdx].pt)
        pts1.append(queryKeypoints[m.queryIdx].pt)
    pts1  = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    return (pts1, pts2)

def compute_homography(img1_filename, img2_filename):
    img1 = cv2.imread(img1_filename)
    img2 = cv2.imread(img2_filename)

    (height, width, _) = img1.shape

    (pts1, pts2) = match_features(img1, img2)

    # Compute fundamental matrix
    H, mask = cv2.findHomography(pts2, pts1, method=cv2.RANSAC)

    #warped_img = np.zeros((height,width))
    #warped_img = cv2.warpPerspective(src=img2, M=H, dsize=(width,height))
    #cv2.imwrite("data/warped.png", warped_img)

    return H

def compute_essential_matrix(img1_filename, img2_filename, K):
    img1 = cv2.imread(img1_filename)
    img2 = cv2.imread(img2_filename)

    # compute matching features
    (pts1, pts2) = match_features(img1, img2)

    # Compute fundamental matrix
    E, mask = cv2.findEssentialMat(pts1,pts2,K)
    return E

def compute_fundamental_matrix(img1_filename, img2_filename):
    img1 = cv2.imread(img1_filename)
    img2 = cv2.imread(img2_filename)

    # compute matching features
    (pts1, pts2) = match_features(img1, img2)
    
    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
    return F

def main():
    if (len(sys.argv) != 4):
        print("Error: usage   python {} <img-1> <img-2>".format(sys.argv[0]))
        sys.exit()

    f1 = sys.argv[1]
    f2 = sys.argv[2]
    K = load_intrinsics(sys.argv[3])

    F = compute_fundamental_matrix(f1, f2)
    print("Fundamental Matrix:\n", F)

    E = compute_essential_matrix(f1, f2, K)
    print("Essential Matrix:\n", E)


    H = compute_homography(f1, f2)
    print("Homography Warping:\n", H)



if __name__=="__main__":
    main()
