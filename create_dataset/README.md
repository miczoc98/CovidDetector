## Steps to prepare the dataset:

1. Create an 'initial_dataset' directory and enter it.
2. Download cohen dataset:
   `git clone https://github.com/ieee8023/covid-chestxray-dataset.git`
   (/images dir and metadata.csv required).
3. Download Figure1 dataset:
   `git clone https://github.com/agchung/Figure1-COVID-chestxray-dataset.git`
   (/images dir and metadata.csv required).
4. Download Actualmed dataset:
   `git clone https://github.com/agchung/Actualmed-COVID-chestxray-dataset.git`
   (/images dir and metadata.csv required).
5. Download COVID dir and COVID.metadata.xlsx from [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database/version/3)
   and put it inside 'sirm' directory (also, rename COVID.metadata.xlsx to metadata.xlsx).
6. Download [RSNA pneumonia dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)
   (/stage_2_train_images dir, stage_2_train_labels.csv and stage_2_detailed_class_info.csv required).
7. Download images and .csv from [RICORD COVID-19 dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281)
   and put it inside 'ricord' directory. In order to download images, you first have to download .tcia file from the page and then open it, using [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images). After doing that, put 'MIDRC-RICORD-1C' dir containing images directly into 'ricord' directory.
8. Download [ricord_data_set.txt](https://github.com/lindawangg/COVID-Net/blob/master/create_ricord_dataset/ricord_data_set.txt) and put it inside 'ricord' directory.
9. Download [rsna_test_patients_normal.npy](https://github.com/lindawangg/COVID-Net/blob/master/rsna_test_patients_normal.npy) and [rsna_test_patients_pneumonia.npy](https://github.com/lindawangg/COVID-Net/blob/master/rsna_test_patients_pneumonia.npy).
10. Leave 'initial_dataset' dir and run `create_ricord_dataset.py` and `create_dataset.py`. In `create_dataset.py` you can add a parameter which will define number of images in the dataset (default value is 300 images)
11. If everything was done correctly, new dataset (called 'dataset\_{number of images}') should appear in a 'datasets' directory. It contains images randomly selected from above datasets, converted to png and resized to 256x256.
