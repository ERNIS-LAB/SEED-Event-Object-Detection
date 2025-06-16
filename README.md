# SEED: Sparse Convolutional Recurrent Learning for Efficient Event-based Neuromorphic Object Detection

## Citation 
This is the official implementation for SEED. Please refer to and cite the following paper:

```bibtex
@inproceedings{seed2025,
  title={Sparse Convolutional Recurrent Learning for Efficient Event-based Neuromorphic Object Detection},
  author = {Wang, Shenqi and Xu, Yingfu and Yousefzadeh, Amirreza and Eissa, Sherif and Corporaal, Henk and Corradi, Federico and Tang, Guangzhi},
  booktitle={2025 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2025},
  organization={IEEE}
}
```

## Dependencies
### Prophesee Metavision SDK

We are using Metavision [4.2.1](https://docs.prophesee.ai/4.2.1/index.html).

Install Prophesee Metavision SDK using this [LINK](https://docs.prophesee.ai/4.2.1/installation/linux.html)

### Prophesee 1 Megapixel Event Dataset

Download DAT dataset from [HERE](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/).

Follow the tutorial [HERE](https://docs.prophesee.ai/4.2.1/tutorials/ml/data_processing/precomputing_features_hdf5_datasets.html#chapter-tutorials-ml-precomputing-features-hdf5-datasets) to pre-process the DAT file to HDF5 file, the pre-processing method used in this project is ```multi_channel_timesurface```.

The pre-comupted dataset can be downloaded [HERE](https://kdrive.infomaniak.com/app/share/975517/17a07fbc-39b4-4ab7-b006-90f8a712ec08/files/78). However, we noticed that the content of DAT and pre-computed are different. Our results are trained and tested on the preprocessed DAT dataset.

### Enviornment setup
```
python -m pip install -r requirements.txt
```

## Training and Testing
### Training 
Please modify the `dataset_path` in the `train.py` file to your own dataset path.

Use command ```python train.py``` to run the training script.

### Validation
Please modify the `saved_model_path` in the `validate.py` file to saved model directory.

Please modify the `dataset_path` in the `validate.py` file to your own dataset path.

Use command ```python validate.py``` to run the training script.

### Test
Please modify the `saved_model_path` in the `test.py` file to saved model directory.

Select the best performance mode drom the validation and modify the `test_epoch` in `test.py`.

Please modify the `dataset_path` in the `test.py` file to your own dataset path.

Use command ```python test.py``` to run the training script.



