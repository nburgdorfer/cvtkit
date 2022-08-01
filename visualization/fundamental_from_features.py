import numpy as np
import os
import sys
import cv2


def compute_fundamental_matrix(img1, img2):
    query_img = cv2.imread(img1)
    train_img = cv2.imread(img2)
      
    # Convert it to grayscale
    query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
      
    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create(500)
      
    # Now detect the keypoints and compute
    # the descriptors for the query image
    # and train image
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)
     
    # Initialize the Matcher for matching
    # the keypoints and then match the
    # keypoints
    matcher = cv2.BFMatcher(crossCheck=True)
    matches = list(matcher.match(queryDescriptors,trainDescriptors) )
    matches.sort(key = lambda x:x.distance)

    # extract points
    pts1 = []
    pts2 = []
    for i in range(8):
        m = matches[i]
        print(m.distance)

        pts2.append(trainKeypoints[m.trainIdx].pt)
        pts1.append(queryKeypoints[m.queryIdx].pt)
    pts1  = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    
    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
    return F




def main():
    if (len(sys.argv) != 3):
        print("Error: usage   python {} <img-1> <img-2>".format(sys.argv[0]))
        sys.exit()

    f1 = sys.argv[1]
    f2 = sys.argv[2]

    F = compute_fundamental_matrix(f1, f2)
    print(F)



if __name__=="__main__":
    main()
