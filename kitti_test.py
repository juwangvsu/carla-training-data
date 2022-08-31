import numpy as np
import struct
import open3d as o3d
import sys
#from open3d import *
# handy utitlity to convert kitti bin from/to pcd 
#input: pcd is numpy array 4xn

def convert_pcd_to_kitti_bin(pcd, binFilePath):
    f=open(binFilePath, "wb") 
    for i in range(pcd.shape[0]):
        byte = struct.pack("ffff",pcd[i][0],pcd[i][1],pcd[i][2],pcd[i][3])
        f.write(byte)
    f.close()

def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    list_pcdi = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            list_pcdi.append([x, y, z, intensity])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    np_pcdi = np.asarray(list_pcdi)
    #np_pcdi.tofile('/tmp/testpcd.bin') this will not generate the same file format as kitti bin due to encoding
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd, np_pcdi

print(sys.argv[1])
#pcd = convert_kitti_bin_to_pcd("_out/velodyne/000571.bin")
pcd, np_pcdi = convert_kitti_bin_to_pcd(sys.argv[1])
#o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)
o3d.io.write_point_cloud(sys.argv[2], pcd)
convert_pcd_to_kitti_bin(np_pcdi, '/tmp/testpcd.bin')
