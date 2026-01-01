import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

def preprocess_eicu_tables(data_dir):

    patient = pd.read_csv(os.path.join(data_dir, 'patient.csv.gz'))
    vital = pd.read_csv(os.path.join(data_dir, 'vitalPeriodic.csv.gz'))
    lab = pd.read_csv(os.path.join(data_dir, 'lab.csv.gz'))

    # Demographics
    demo = patient[['patientunitstayid', 'age', 'gender', 'ethnicity']].copy()
    demo['gender'] = (demo['gender'] == 'Male').astype(int)
    demo['age'] = pd.to_numeric(demo['age'], errors='coerce')
    demo = demo.loc[demo['age'] != '> 89']
    demo['age'] = demo['age'].astype(float)
    demo = demo.loc[demo['age'] >= 18]
    demo['los'] = pd.to_numeric(
        patient.set_index('patientunitstayid').loc[demo['patientunitstayid']]['unitdischargeoffset'],
        errors='coerce'
    ) / 1440.0
    demo = pd.get_dummies(demo, columns=['ethnicity'], dummy_na=True)
    demo = demo.dropna().groupby('patientunitstayid').first()

    # Vitals
    vital = vital.drop(columns=['observationoffset'], errors='ignore')
    vital = vital.select_dtypes(include=[np.number])
    vital_avg = vital.groupby('patientunitstayid').mean()

    # Labs
    lab = lab[['patientunitstayid', 'labname', 'labresult']].dropna()
    lab['labresult'] = pd.to_numeric(lab['labresult'], errors='coerce')
    lab = lab.dropna()
    lab_wide = lab.pivot_table(index='patientunitstayid', columns='labname', values='labresult', aggfunc='mean')

    return demo, vital_avg, lab_wide

def split_eicu_data(demo, vitals, labs, patient_df, max_samples=None):

    common_ids = demo.index.intersection(vitals.index).intersection(labs.index)
    if max_samples:
        common_ids = common_ids[:max_samples]

    demo = demo.loc[common_ids]
    vitals = vitals.loc[common_ids]
    labs = labs.loc[common_ids]

    features_static = demo.to_numpy(dtype=np.float32)
    features_timeseries = pd.concat([vitals, labs], axis=1).fillna(0).to_numpy(dtype=np.float32)

    label_map = patient_df.drop_duplicates('patientunitstayid').set_index('patientunitstayid')
    labels = (label_map.loc[common_ids]['hospitaldischargestatus'] == 'Expired').astype(int).to_numpy(dtype=np.int64)

    return features_static, features_timeseries, labels

def create_eicu_dataloaders(X_s, X_t, y, batch_size=40, num_workers=1, shuffle=True):

    X_s = torch.tensor(X_s, dtype=torch.float32)
    X_t = torch.tensor(X_t, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = list(zip(X_s, X_t, y))
    train_val, test = train_test_split(dataset, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.0625, random_state=42)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def get_loader(batch_size=40, num_workers=1, shuffle=True, root_dir="../../data/eicu/", max_samples=None):

    demo, vitals, labs = preprocess_eicu_tables(root_dir)
    patient = pd.read_csv(os.path.join(root_dir, 'patient.csv.gz'))
    X_s, X_t, y = split_eicu_data(demo, vitals, labs, patient, max_samples=max_samples)

    train_loader, val_loader, test_loader = create_eicu_dataloaders(
        X_s, X_t, y, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )

    n_classes = len(np.unique(y))


    with open("samples.txt", "w") as f:
        f.write(f"Total samples: {len(y)}\n")
        f.write(f"Train: {len(train_loader.dataset)}\n")
        f.write(f"Val: {len(val_loader.dataset)}\n")
        f.write(f"Test: {len(test_loader.dataset)}\n")

    return train_loader, val_loader, test_loader, n_classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, n_classes = get_loader()
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape, batch[2].shape)
        break
