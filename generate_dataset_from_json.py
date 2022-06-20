import glob
import open3d as o3d
import numpy as np
import h5py
import json
import sys

from sklearn.neighbors import KDTree

def pc_normalize(pc):
    centroid = np.mean(pc[:,:3], axis=0)
    pc[:,:3] = pc[:,:3] - centroid
    m = np.max(np.sqrt(np.sum(pc[:,:3]**2, axis=1)))
    pc[:,:3] = pc[:,:3] / m
    return pc


with open('segmentation_index.json') as json_file:
    data = json.load(json_file)

filenames = glob.glob('data/*.ply')


dataList = []
nameList = []
indexList = []
segList = []

dataListTest = []
nameListTest = []
indexListTest = []
segListTest = []


obj_idx = 0
idx = 1



for filename in filenames:
    key = filename.split("/")[-1][:-4]
    if key in data.keys():
        textured_mesh = o3d.io.read_triangle_mesh( filename )
     
        tree = KDTree( np.asarray(data[key]["vertex_color"]), leaf_size=2)
        dist_feature_list, index_feature_list = tree.query(np.asarray(textured_mesh.vertices), k=1) # Instead of only taking nearest, 15 points are used.

        for j in range(len(dist_feature_list)):
            if( dist_feature_list[j] < 0.10 ):
                textured_mesh.vertex_colors[j] = [1.0, 0.0, 0.0]
            else:
                textured_mesh.vertex_colors[j] = [0.47843137254901963, 0.47843137254901963, 0.47843137254901963]
                 
        
        pcd_o3d = textured_mesh.sample_points_uniformly(number_of_points=1024*8)
        pcd_o3d.estimate_normals()
        pointcloud_pointnet = np.concatenate( [np.asarray(pcd_o3d.points), np.asarray(pcd_o3d.normals)], axis = 1 )
   
        # o3d.visualization.draw_geometries([ pcd_o3d ])
        
        colors = np.asarray(pcd_o3d.colors)
        segment = []
        for col in colors:
            segment.append( 1 if col[0]*2 - (col[1]+col[2]) > 1 else 0 )
  
        pointcloud_pointnet = pc_normalize(pointcloud_pointnet)
  
        if data[key]["index"] == "train":
            dataList.append( pointcloud_pointnet )
            indexList.append( idx )
            nameList.append( obj_idx )
            segList.append(segment)        
        elif data[key]["index"] == "test":
            dataListTest.append( pointcloud_pointnet )
            indexListTest.append( idx )
            nameListTest.append( obj_idx )
            segListTest.append(segment)
        else:
            print( "Error not indexed!" )
            sys.exit()

        obj_idx += 1
    
  
DATASET_SIZE = len(dataList)
f = h5py.File( "pin_seg_train_o3d.h5", 'w')

f.create_dataset("data", (DATASET_SIZE, 1024*8, 6), dtype="f4")
f.create_dataset("cat", (DATASET_SIZE, 1024*8), dtype="i")
f.create_dataset("label", (DATASET_SIZE, 1), dtype='i')
f.create_dataset("obj_idx", (DATASET_SIZE, 1), dtype='i')

f['data'][:] = np.array(dataList)
f['cat'][:] = np.array(segList)
f['label'][:] = np.reshape(np.array(indexList), (DATASET_SIZE,1))
f['obj_idx'][:] = np.reshape(np.array(nameList), (DATASET_SIZE,1))

f.close()


DATASET_SIZE = len(dataListTest)  
f = h5py.File( "pin_seg_test_o3d.h5", 'w')

f.create_dataset("data", (DATASET_SIZE, 1024*8, 6), dtype="f4")
f.create_dataset("cat", (DATASET_SIZE, 1024*8), dtype="i")
f.create_dataset("label", (DATASET_SIZE, 1), dtype='i')
f.create_dataset("obj_idx", (DATASET_SIZE, 1), dtype='i')

f['data'][:] = np.array(dataListTest)
f['cat'][:] = np.array(segListTest)
f['label'][:] = np.reshape(np.array(indexListTest), (DATASET_SIZE,1))
f['obj_idx'][:] = np.reshape(np.array(nameListTest), (DATASET_SIZE,1))

f.close()
