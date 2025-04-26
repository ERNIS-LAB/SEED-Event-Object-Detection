import os
import glob
from functools import partial
from metavision_ml.data import box_processing as box_api
from metavision_ml.data import SequentialDataLoader

class seq_dataloader:
    def __init__(self, dataset_path, dataset_type = 'gen4', num_tbins = 12, batch_size = 4, channels = 6, delta_t = 50000, max_incr_per_pixel = 5, preprocess_function_name = "multi_channel_timesurface" ):
        
        label_map_path = os.path.join(dataset_path, 'label_map_dictionary.json')
        train_path = os.path.join(dataset_path, 'train')
        val_path = os.path.join(dataset_path, 'val')
        test_path = os.path.join(dataset_path, 'test')

        files_train = glob.glob(os.path.join(train_path, "*.h5"))
        files_val = glob.glob(os.path.join(val_path, "*.h5"))
        files_test = glob.glob(os.path.join(test_path, "*.h5"))
        
        
        if dataset_type == 'gen1':
            self.wanted_keys = [ 'car', 'pedestrian']
            self.height, self.width = 240, 304
            self.min_box_diag_network = 30
        elif dataset_type == 'gen4':
            self.wanted_keys = ['pedestrian', 'two wheeler', 'car']
            self.height, self.width = 360, 640
            self.min_box_diag_network = 60
        
        self.channels = channels
        self.batch_size = batch_size
        self.num_tbins = num_tbins
        self.dataset_type = dataset_type
        
        array_dim = [num_tbins, channels, self.height, self.width]
        class_lookup = box_api.create_class_lookup(label_map_path, self.wanted_keys)
       
        self.load_boxes_fn = partial(box_api.load_boxes, num_tbins=num_tbins, class_lookup=class_lookup, min_box_diag_network=self.min_box_diag_network)
        
        self.seq_dataloader_train = SequentialDataLoader(files_train, delta_t, preprocess_function_name, array_dim, load_labels=self.load_boxes_fn,
                                      batch_size=batch_size, num_workers=4, padding=True, preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel})
        self.seq_dataloader_val = SequentialDataLoader(files_val, delta_t, preprocess_function_name, array_dim, load_labels=self.load_boxes_fn,
                                      batch_size=batch_size, num_workers=4, padding=True,preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel})
        self.seq_dataloader_test = SequentialDataLoader(files_test, delta_t, preprocess_function_name, array_dim, load_labels=self.load_boxes_fn,
                                      batch_size=batch_size, num_workers=4, padding=True,preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel})


