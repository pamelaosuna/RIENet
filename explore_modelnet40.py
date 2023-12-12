import os, glob
import h5py
import numpy as np

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    all_data, all_label = [], []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64') # class of the 3D point cloud object, one of the 39 classes
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0) # shape: (2468, 2048, 3) # (num_samples, num_points, num_dims)
    all_label = np.concatenate(all_label, axis=0) # shape: (2468, 1)

    return all_data, all_label

if __name__ == '__main__':
    partition = 'test'
    data, label = load_data(partition)