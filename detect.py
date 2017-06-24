#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:04:20 2017

@author: joao.ferreira
"""
import os.path
import pickle
import train
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


class vehicle():
    
    def __init__(self, win=((0,0) ,(0,0))):
        self.age = 0      #for how many frames it exists
        self.kill_deb = 0 #debounce counter to remove vehicle
        self.window = win #((x1,y1),(x2,y2))
        
    def is_car(self):
        #It is only valid if age above threhold
        return (self.age > 2)

    def die(self):
        #It is only removed if no BBoxes for more than N number frames
        return (self.kill_deb > 2)

    def update(self, win):
        # Refresh bounding box position/size, keep vehicle alive by incrementing age and reseting kill_deb
        self.window = win
        if not self.is_car():
            self.age += 1
        self.kill_deb = 0

    def kill(self):
        #Tries to remove, increments the killing counter
        self.kill_deb += 1

    def inside(self,p):
        #Checks if point p inside vehicle bounding box
        return p[0] >= self.window[0][0] and p[0] <= self.window[1][0] and p[1] >= self.window[0][1] and p[1] <= self.window[1][1]


def draw_boxes(img, bboxes, color=(0, 0, 1), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for b in bboxes:
        draw_img = cv2.rectangle(draw_img, b[0], b[1], color, thick)
    # return the image copy with boxes drawn
    return draw_img # Change this line to return image copy with boxes


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cells_per_step=2, vis=False):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)
    
    # Crop and scale interest area
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    #Separate Channels    
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 

    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = train.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, ch=None)
    hog2 = train.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False, ch=None)
    hog3 = train.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False, ch=None)
    bbox_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = train.bin_spatial(subimg, size=spatial_size)
            hist_features = train.color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                if vis:
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,1),6) 
                bbox = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart))
                bbox_list.append(bbox)
                
    if vis:
        return draw_img
    else:
        return bbox_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def get_bboxes(labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        bboxes.append(bbox)
    
    return bboxes
    
def draw_labeled_bboxes(img, labels):
    bboxes = get_bboxes(labels)
    for bbox in bboxes:
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,1), 6)
    # Return the image
    return img

def assign(labels, vehicle_list):
        
    vl = vehicle_list
    for v in vl:
        #Tries to remove vehicle but if it is found on this frame it will not b removed
        v.kill()
        if v.die():
            #If vehicle is not found for several frames it is removed
            vehicle_list.remove(v)
    
    # Iterate through list of bboxes
    for box in labels:
        center = ((box[0][0]+box[1][0])/2, (box[0][1]+box[1][1])/2)
        belonging = False
        for v in vehicle_list:
            if v.inside(center):
                #If window center inside found window on last frame
                belonging = True
                v.update(box)
                break
            
        if not belonging:
            #If not belonging to any vehicle, create new 
            vehicle_list.append(vehicle(box))
        
    return vehicle_list

def get_boxed_vehicle(img, box_list, vis=False):
    # Add heat to each box in box list
    heat = add_heat(np.zeros_like(img[:,:,0]).astype(np.float),box_list)   
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    if vis:
        plt.figure
        plt.imshow(heatmap, cmap="jet")
    
    # Find and return final boxes from heatmap using label function
    return label(heatmap)

class pipeline():
    #Pipeline to process video frames
    def __init__(self, svc, scaler):
        self.vehicle_list = [] #list of vehicles in the video
        self.svc = svc # Classifier of vehicle/non-vehicle
        self.scaler = scaler#Scaler of features
        
    def process(self, img):
        #ROI
        ystart = 380
        ystop = 656
        scl_img = np.array(img).astype(np.float64)/255.0 #convert from jpg to png scale [0,1]
        
        # Scales and stepsize to slide window and search for vehicle
        scales = [1.0, 1.5]
        stepsize = [2,2]
        box_list = []
        for scale, step in zip(scales,stepsize):
            # Find boxes classified as vehicles
            blist = find_cars(scl_img, ystart, ystop, scale, self.svc, self.scaler, 9, 8, 2, (16,16), 32, cells_per_step=step)
            box_list += blist
        #Merge all boxes using heatmap and label
        labels = get_boxed_vehicle(img, box_list)
        bboxes = get_bboxes(labels)
        self.vehicle_list = assign(bboxes, self.vehicle_list)
        
        # Draw boxes on video frame
        vboxes = []
        for v in self.vehicle_list:
            if v.is_car():
                vboxes.append(v.window)
        frame = draw_boxes(img, vboxes, color=(0, 0, 255), thick=6)
        
        return frame
    
    
def test(svc, scaler):
    #Test only one frame
    
    # Read in each one by one
    image = mpimg.imread("test2.png")
    
    ystart = 380
    ystop = 656
    
    if False:
        scale = 2
        out_img = find_cars(image, ystart, ystop, scale, svc, scaler, 9, 8, 2, (16,16), 32,vis=True, cells_per_step=1)
        plt.figure
        plt.imshow(out_img)
        
    elif False:
        
        blist = find_cars(image, ystart, ystop, 1, svc, scaler, 9, 8, 2, (16,16), 32, cells_per_step=2)
        out_img = draw_boxes(image, blist, color=(0, 0, 1), thick=4)
        blist = find_cars(image, ystart, ystop, 1.5, svc, scaler, 9, 8, 2, (16,16), 32, cells_per_step=2)
        out_img = draw_boxes(out_img, blist, color=(0, 1, 0), thick=4)
        plt.figure
        plt.imshow(out_img)
    else:
        scales   = [1, 1.5]
        stepsize = [2, 2]
        box_list = []
        for scale, step in zip(scales,stepsize):
            # Find boxes classified as vehicles
            blist = find_cars(image, ystart, ystop, scale, svc, scaler, 9, 8, 2, (16,16), 32, cells_per_step=step)
            box_list += blist
        
        labels = get_boxed_vehicle(image, box_list, vis=False)
        bboxes = get_bboxes(labels)
        vlist = assign(bboxes, [])
        vlist = assign(bboxes, vlist)
        print(vlist)
    
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        plt.figure
        plt.imshow(draw_img)
    
    
    
if __name__ == "__main__":

    vid = True
    
    #Load Classifier and Feature Scaler
    (svc, scaler) = pickle.load(open("model.p", 'rb'))
    
    if vid:
        # Load video clip
        clip1 = VideoFileClip("project_video.mp4")
        # Create pipeline to process frame by frame
        pipe = pipeline(svc, scaler) 
        white_clip = clip1.fl_image(pipe.process)
        
        # Save processed video with bounding boxes overlayed
        white_clip.write_videofile("output.mp4", audio=False)
        
    else:
        test(svc, scaler)
    
        
     
    
    
    
    
    

