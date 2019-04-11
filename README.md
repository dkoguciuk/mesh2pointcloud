# mesh2pointcloud

This is a mini script to sample [ModelNet40](http://modelnet.cs.princeton.edu/) or [ShapeNetCore55v2](https://www.shapenet.org/) meshes into 3D point clouds, which are further consumed by PointNet or other set--based classification architectures. For now only random sampling is implemented.

----------

### Usage

To convert ShapeNetCore55V2 meshes into 3D point clouds with 1024 points:
```
python3 shapenet_obj_2_hdf5.py -i //media/daniel/HDD/ShapeNetCore_v2_orig -o /media/daniel/HDD/ShapeNetCore_v2 -s 1024
```
To convert ModelNet40 meshes into 3D point clouds with 1024 points:
```
python3 modelnet_off_2_hdf5.py -i //media/daniel/HDD/ShapeNetCore_v2_orig -o /media/daniel/HDD/ShapeNetCore_v2 -s 1024
```

### License

The code is released under MIT License (see LICENSE file for details).
