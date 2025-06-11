from scipy.io import loadmat

def get_SantaBarbara_dataset():
    # Load the .mat files
    data_set_before = loadmat('/kaggle/input/sst-change-detection-dataset/ChangeDetectionDataset-master-72b0263758e1f4dc04769bac5a688f951f5ce23a/santaBarbara/mat/barbara_2013.mat')['imgh']
    data_set_after = loadmat('/kaggle/input/sst-change-detection-dataset/ChangeDetectionDataset-master-72b0263758e1f4dc04769bac5a688f951f5ce23a/santaBarbara/mat/barbara_2014.mat')['imgh']
    ground_truth = loadmat('/kaggle/input/sst-change-detection-dataset/ChangeDetectionDataset-master-72b0263758e1f4dc04769bac5a688f951f5ce23a/santaBarbara/mat/barbara_gtChanges.mat')['gt']

    # Convert to float32
    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt


def get_dataset(current_dataset):
    if current_dataset == 'Santabarbara':
        return get_Farmland_dataset()
