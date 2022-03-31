import pandas as pd
import os
from fastai.tabular.all import *
from tqdm import tqdm


FEATURES_1 = [
    'BaseExcess',
    'Fibrinogen',
    'AST',
    'Alkalinephos',
    'Bilirubin_total',
    'Lactate',
    'TroponinI',
    'SaO2',
    'Bilirubin_direct',
    'EtCO2'
]

LABELS = [
    'LABEL_BaseExcess',
    'LABEL_Fibrinogen',
    'LABEL_AST',
    'LABEL_Alkalinephos',
    'LABEL_Bilirubin_total',
    'LABEL_Lactate',
    'LABEL_TroponinI',
    'LABEL_SaO2',
    'LABEL_Bilirubin_direct',
    'LABEL_EtCO2',
    'LABEL_Sepsis',
    'LABEL_RRate',
    'LABEL_ABPm',
    'LABEL_SpO2',
    'LABEL_Heartrate'
]

FEATURES = ['Time', 'Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3',
            'BaseExcess', 'RRate', 'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2',
            'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose', 'ABPm', 'Magnesium', 'Potassium',
            'ABPd', 'Calcium', 'Alkalinephos', 'SpO2', 'Bilirubin_direct', 'Chloride', 'Hct',
            'Heartrate', 'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']


class Task:
    def __init__(self) -> None:
        self.df_train = None
        self.df_labels = None
        self.df_test = None
        self.df_train_combined = None
        self.df_predict = pd.DataFrame()

    def load_datasets(self):
        """Load datasets from csv files
        """
        self.df_train = pd.read_csv('train_features.csv')
        self.df_labels = pd.read_csv('train_labels.csv')
        self.df_test = pd.read_csv('test_features.csv')
        self.df_train_combined = pd.DataFrame.merge(
            self.df_train[['pid']+FEATURES],
            self.df_labels[['pid']+LABELS],
            on='pid'
        )

    def load_dls(self, label):
        """Load dataset columns into a DataLoader

        Parameters
        ----------
        label : string
            the selected target label

        Returns
        -------
        dls : TabularDataLoader
            the tabular DataLoader to be used
        """
        splits = RandomSplitter(valid_pct=0.2)(
            range_of(self.df_train_combined))
        to = TabularPandas(
            self.df_train_combined,
            procs=[
                FillMissing(add_col=False, fill_strategy=FillStrategy.mode),
                Normalize
            ],
            cont_names=FEATURES,
            y_names=label,
            y_block=RegressionBlock,
            splits=splits
        )
        dls = to.dataloaders(bs=1024)
        return dls

    def fit_and_predict(self):
        """Data fitting and prediction
        """
        for label in tqdm(LABELS):
            print("Working on label", label)
            dls = self.load_dls(label=label)
            metrics = [accuracy, R2Score(), rmse, mae]
            learn = tabular_learner(dls, metrics=metrics)
            lr = learn.lr_find()
            learn.fit_one_cycle(10, lr)
            dl = learn.dls.test_dl(self.df_test)
            predict = learn.get_preds(dl=dl)
            self.df_predict[label] = predict[0].numpy().flatten()

    def store_result(self):
        """Store results into prediction.zip
        """
        result = self.df_predict
        result['pid'] = self.df_test['pid']
        result = result.groupby(['pid']).mean()
        compression_opts = dict(method='zip', archive_name='prediction.csv')
        result.to_csv(
            'prediction.zip',
            index=True,
            float_format='%.3f',
            compression=compression_opts
        )


def main():
    task = Task()
    task.load_datasets()
    task.fit_and_predict()
    task.store_result()


if __name__ == "__main__":
    main()
