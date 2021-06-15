import os
import random
import sys
import numpy as np
import pandas as pd
import pydicom as dicom
import cv2
import glob
from shutil import copyfile
from PIL import Image


def resize_image_and_save_png(patient, dir_out, size):
    filename = os.path.basename(patient[1])
    if patient[3] == 'sirm':
        image = cv2.imread(patient[1])
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filename = filename.replace(' ', '')
    elif patient[3] == 'rsna':
        ds = dicom.dcmread(os.path.join(patient[1]))
        img = ds.pixel_array
    else:
        img = cv2.imread(patient[1])

    filename = filename[:filename.rfind('.')] + '.png'
    im = Image.fromarray(img)
    im = im.resize(size)
    im.save(os.path.join(dir_out, filename))


if __name__ == '__main__':

    print("Creating dataset...")
    ###################################################################################
    # Paths to images and csv files of downloaded datasets
    initial_ds_name = 'initial_dataset'

    cohen_imgpath = os.path.join(
        initial_ds_name, 'covid-chestxray-dataset', 'images')
    cohen_csvpath = os.path.join(
        initial_ds_name, 'covid-chestxray-dataset', 'metadata.csv')

    fig1_imgpath = os.path.join(
        initial_ds_name, 'Figure1-COVID-chestxray-dataset', 'images')
    fig1_csvpath = os.path.join(
        initial_ds_name, 'Figure1-COVID-chestxray-dataset', 'metadata.csv')

    actmed_imgpath = os.path.join(
        initial_ds_name, 'Actualmed-COVID-chestxray-dataset', 'images')
    actmed_csvpath = os.path.join(
        initial_ds_name, 'Actualmed-COVID-chestxray-dataset', 'metadata.csv')

    sirm_imgpath = os.path.join(initial_ds_name, 'sirm', 'COVID')
    sirm_csvpath = os.path.join(initial_ds_name, 'sirm', 'metadata.xlsx')

    rsna_datapath = os.path.join(
        initial_ds_name, 'rsna-pneumonia-detection-challenge')
    rsna_csvname = 'stage_2_detailed_class_info.csv'
    rsna_csvname2 = 'stage_2_train_labels.csv'
    rsna_imgpath = 'stage_2_train_images'

    ricord_imgpath = os.path.join(initial_ds_name, 'ricord', 'ricord_images')
    ricord_txt = os.path.join(initial_ds_name, 'ricord', 'ricord_data_set.txt')
    ###################################################################################

    try:
        ds_size = int(sys.argv[1])
        if ds_size > 6000:
            ds_size = 6000
    except Exception:
        ds_size = 300

    seed = 0
    np.random.seed(seed)  # Reset the seed so all runs are the same.
    random.seed(seed)

    train = []
    test = []
    test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

    mapping = dict()
    mapping['COVID-19'] = 'COVID-19'
    mapping['SARS'] = 'pneumonia'
    mapping['MERS'] = 'pneumonia'
    mapping['Streptococcus'] = 'pneumonia'
    mapping['Klebsiella'] = 'pneumonia'
    mapping['Chlamydophila'] = 'pneumonia'
    mapping['Legionella'] = 'pneumonia'
    mapping['E.Coli'] = 'pneumonia'
    mapping['Normal'] = 'normal'
    mapping['Lung Opacity'] = 'pneumonia'
    mapping['1'] = 'pneumonia'

    classes = list(train_count.keys())
    class_length = int(ds_size / len(classes))

    # For each class
    train_ratio = 0.8
    validate_ratio = 0.1

    train_length = int(round(class_length * train_ratio))
    validate_length = int(round(class_length * validate_ratio))
    test_length = class_length - train_length - validate_length

    ds_dir = os.path.join('..', 'datasets', 'dataset_' +
                          str(ds_size))
    os.makedirs(ds_dir, exist_ok=True)
    train_path = os.path.join(ds_dir, 'train')
    validate_path = os.path.join(ds_dir, 'validate')
    test_path = os.path.join(ds_dir, 'test')

    # Create directory for each class
    for cl in classes:
        os.makedirs(os.path.join(train_path, cl), exist_ok=True)
        os.makedirs(os.path.join(validate_path, cl), exist_ok=True)
        os.makedirs(os.path.join(test_path, cl), exist_ok=True)
    ###################################################################################

    # to avoid duplicates
    patient_imgpath = {}

    cohen_csv = pd.read_csv(cohen_csvpath, nrows=None)
    views = ["PA", "AP", "AP Supine", "AP semi erect", "AP erect"]
    cohen_idx_keep = cohen_csv.view.isin(views)
    cohen_csv = cohen_csv[cohen_idx_keep]

    fig1_csv = pd.read_csv(fig1_csvpath, encoding='ISO-8859-1', nrows=None)
    actmed_csv = pd.read_csv(actmed_csvpath, nrows=None)

    sirm_csv = pd.read_excel(sirm_csvpath)

    # get non-COVID19 viral, bacteria, and COVID-19 infections from covid-chestxray-dataset, figure1 and actualmed
    # stored as patient id, image filename and label
    filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
    count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    covid_ds = {'cohen': [], 'fig1': [],
                'actmed': [], 'sirm': [], 'ricord': []}

    for index, row in cohen_csv.iterrows():
        # take final finding in hierarchy, for the case of COVID-19, ARDS
        f = row['finding'].split('/')[-1]
        if f == 'COVID-19' and ('eurorad.org' in row['url'] or 'ml-workgroup' in row['url'] or 'sirm.org' in row['url']):
            # skip COVID-19 positive images from eurorad to not duplicate sirm images
            pass
        elif f in mapping:
            count[mapping[f]] += 1
            entry = [str(row['patientid']), os.path.join(cohen_imgpath, row['filename']),
                     mapping[f], 'cohen']
            filename_label[mapping[f]].append(entry)
            if mapping[f] == 'COVID-19':
                covid_ds['cohen'].append(str(row['patientid']))

    for index, row in fig1_csv.iterrows():
        if not str(row['finding']) == 'nan':
            f = row['finding'].split(',')[0]  # take the first finding
            if f in mapping:
                count[mapping[f]] += 1
                if os.path.exists(os.path.join(fig1_imgpath, row['patientid'] + '.jpg')):
                    entry = [row['patientid'], os.path.join(fig1_imgpath, row['patientid'] +
                             '.jpg'), mapping[f], 'fig1']
                elif os.path.exists(os.path.join(fig1_imgpath, row['patientid'] + '.png')):
                    entry = [row['patientid'], os.path.join(fig1_imgpath, row['patientid'] +
                             '.png'), mapping[f], 'fig1']
                filename_label[mapping[f]].append(entry)
                if mapping[f] == 'COVID-19':
                    covid_ds['fig1'].append(row['patientid'])

    for index, row in actmed_csv.iterrows():
        if not str(row['finding']) == 'nan':
            f = row['finding'].split(',')[0]
            if f in mapping:
                count[mapping[f]] += 1
                entry = [row['patientid'],
                         os.path.join(actmed_imgpath, row['imagename']), mapping[f], 'actmed']
                filename_label[mapping[f]].append(entry)
                if mapping[f] == 'COVID-19':
                    covid_ds['actmed'].append(row['patientid'])

    sirm = set(sirm_csv['URL'])
    cohen = set(cohen_csv['url'])
    # Add base URL to remove sirm images from ieee dataset
    cohen.add('https://github.com/ieee8023/covid-chestxray-dataset')
    discard = ['100', '101', '102', '103', '104', '105',
               '110', '111', '112', '113', '122', '123',
               '124', '125', '126', '217']

    for idx, row in sirm_csv.iterrows():
        patientid = row['FILE NAME']
        if row['URL'] not in cohen and patientid[patientid.find('(')+1:patientid.find(')')] not in discard:
            count[mapping['COVID-19']] += 1
            imagename = patientid + '.' + row['FORMAT'].lower()
            if not os.path.exists(os.path.join(sirm_imgpath, imagename)):
                imagename = "COVID ({}).png".format(
                    imagename.rsplit(".png")[0].split("COVID ")[1])
            entry = [patientid, os.path.join(
                sirm_imgpath, imagename), mapping['COVID-19'], 'sirm']
            filename_label[mapping['COVID-19']].append(entry)
            covid_ds['sirm'].append(patientid)

    # get ricord file names
    with open(ricord_txt) as f:
        ricord_file_names = [line.split()[0] for line in f]

    for imagename in ricord_file_names:
        # since RICORD data is all COVID-19 postive images
        count[mapping['COVID-19']] += 1
        patientid = imagename.split('-')[3] + '-' + imagename.split('-')[4]
        entry = [patientid, os.path.join(
            ricord_imgpath, imagename), mapping['COVID-19'], 'ricord']
        filename_label[mapping['COVID-19']].append(entry)
        covid_ds['ricord'].append(patientid)

    # Create list of RICORD patients to be added to test, equal to 200 images

    # We want to prevent patients present in both train and test
    # Get list of patients who have one image
    ricord_patients = []
    for label in filename_label['COVID-19']:
        if label[3] == 'ricord':
            ricord_patients.append(label[0])

    pt_with_one_image = [x for x in ricord_patients if ricord_patients.count(
        x) == 1]  # contains 176 patients

    # add covid-chestxray-dataset, figure1 and actualmed into dataset
    # since these datasets don't have test dataset, split into train/test by patientid
    # for covid-chestxray-dataset:
    # patient 8 is used as non-COVID19 viral test
    # patient 31 is used as bacterial test
    # patients 19, 20, 36, 42, 86 are used as COVID-19 viral test
    # for figure 1:
    # patients 24, 25, 27, 29, 30, 32, 33, 36, 37, 38

    ds_imgpath = {'cohen': cohen_imgpath, 'fig1': fig1_imgpath,
                  'actmed': actmed_imgpath, 'sirm': sirm_imgpath, 'ricord': ricord_imgpath}

    for key in filename_label.keys():
        arr = np.array(filename_label[key])
        if arr.size == 0:
            continue
        if key == 'pneumonia':
            test_patients = ['8', '31']
        elif key == 'COVID-19':
            test_patients = ['19', '20', '36', '42', '86',
                             '94', '97', '117', '132',
                             '138', '144', '150', '163', '169', '174', '175', '179', '190', '191'
                             'COVID-00024', 'COVID-00025', 'COVID-00026', 'COVID-00027', 'COVID-00029',
                             'COVID-00030', 'COVID-00032', 'COVID-00033', 'COVID-00035', 'COVID-00036',
                             'COVID-00037', 'COVID-00038',
                             'ANON24', 'ANON45', 'ANON126', 'ANON106', 'ANON67',
                             'ANON153', 'ANON135', 'ANON44', 'ANON29', 'ANON201',
                             'ANON191', 'ANON234', 'ANON110', 'ANON112', 'ANON73',
                             'ANON220', 'ANON189', 'ANON30', 'ANON53', 'ANON46',
                             'ANON218', 'ANON240', 'ANON100', 'ANON237', 'ANON158',
                             'ANON174', 'ANON19', 'ANON195',
                             'COVID 119', 'COVID 87', 'COVID 70', 'COVID 94',
                             'COVID 215', 'COVID 77', 'COVID 213', 'COVID 81',
                             'COVID 216', 'COVID 72', 'COVID 106', 'COVID 131',
                             'COVID 107', 'COVID 116', 'COVID 95', 'COVID 214',
                             'COVID 129']
            # Add 178 RICORD patients to COVID-19, equal to 200 images
            test_patients.extend(pt_with_one_image)
            test_patients.extend(['419639-000025', '419639-001464'])
        else:
            test_patients = []
        # go through all the patients
        for patient in arr:
            if patient[0] not in patient_imgpath:
                patient_imgpath[patient[0]] = [patient[1]]
            else:
                if patient[1] not in patient_imgpath[patient[0]]:
                    patient_imgpath[patient[0]].append(patient[1])
                else:
                    continue  # skip since image has already been written
            if patient[0] in test_patients:
                test.append(patient)
                test_count[patient[2]] += 1
            else:
                train.append(patient)
                train_count[patient[2]] += 1

    # add normal and rest of pneumonia cases from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    csv_normal = pd.read_csv(os.path.join(
        rsna_datapath, rsna_csvname), nrows=None)
    csv_pneu = pd.read_csv(os.path.join(
        rsna_datapath, rsna_csvname2), nrows=None)
    patients = {'normal': [], 'pneumonia': []}

    for index, row in csv_normal.iterrows():
        if row['class'] == 'Normal':
            patients['normal'].append(row['patientId'])

    for index, row in csv_pneu.iterrows():
        if int(row['Target']) == 1:
            patients['pneumonia'].append(row['patientId'])

    for key in patients.keys():
        arr = np.array(patients[key])
        if arr.size == 0:
            continue
        test_patients = np.load(os.path.join(
            initial_ds_name, 'rsna_test_patients_{}.npy'.format(key)))
        for patient in arr:
            if patient not in patient_imgpath:
                patient_imgpath[patient] = [patient]
            else:
                continue  # skip since image has already been written
            imgname = patient + '.dcm'
            if patient in test_patients:
                test.append([patient, os.path.join(rsna_datapath,
                            rsna_imgpath, imgname), key, 'rsna'])
                test_count[key] += 1
            else:
                train.append([patient, os.path.join(
                    rsna_datapath, rsna_imgpath, imgname), key, 'rsna'])
                train_count[key] += 1

    # Divide train images into train and validate
    train_nmb = 0
    validate_nmb = 0
    test_nmb = 0
    for cl in classes:
        train_validate_data = [t for t in train if t[2] == cl]
        train_validate_data = random.sample(
            train_validate_data, train_length + validate_length)
        train_data = train_validate_data[:train_length]
        validate_data = train_validate_data[train_length:]
        test_data = [t for t in test if t[2] == cl]
        test_data = random.sample(test_data, test_length)

        for td in train_data:
            # If file already exists (i.e. from previous run) then skip
            filename = os.path.basename(td[1])
            filename = filename.replace(' ', '')
            if not os.path.isfile(os.path.join(train_path, cl, filename[:filename.rfind('.')] + '.png')):
                resize_image_and_save_png(
                    td, os.path.join(train_path, cl), (256, 256))
                train_nmb += 1
        for vd in validate_data:
            # If file already exists (i.e. from previous run) then skip
            filename = os.path.basename(vd[1])
            filename = filename.replace(' ', '')
            if not os.path.isfile(os.path.join(validate_path, cl,  filename[:filename.rfind('.')] + '.png')):
                resize_image_and_save_png(
                    vd, os.path.join(validate_path, cl), (256, 256))
                validate_nmb += 1
        for td in test_data:
            # If file already exists (i.e. from previous run) then skip
            filename = os.path.basename(td[1])
            filename = filename.replace(' ', '')
            if not os.path.isfile(os.path.join(test_path, cl,  filename[:filename.rfind('.')] + '.png')):
                resize_image_and_save_png(
                    td, os.path.join(test_path, cl), (256, 256))
                test_nmb += 1

    print("Created", train_nmb, "train images")
    print("Created", validate_nmb, "validate images")
    print("Created", test_nmb, "test images")
    print("Done")
