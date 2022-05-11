import pandas as pd
import os
from fastai.tabular.all import *
from tqdm import tqdm


class Task:
    def __init__(self) -> None:
        self.df_train = None
        self.df_pretrain = None
        self.df_labels = None
        self.df_pretrain_labels = None
        self.df_test = None
        self.df_train_combined = None
        self.df_pretrain_combined = None
        self.df_predict = pd.DataFrame()
        self.model = None

    def load_datasets(self):
        """Load datasets from csv files"""
        self.df_train = pd.read_csv("train_features.csv")
        self.df_pretrain = pd.read_csv("pretrain_features.csv")
        self.features = self.df_train.columns.to_list()[2:]
        self.df_labels = pd.read_csv("train_labels.csv")
        self.df_labels = self.df_labels.rename(
            columns={'homo_lumo_gap': 'label'}
        )
        self.df_pretrain_labels = pd.read_csv("pretrain_labels.csv")
        self.df_pretrain_labels = self.df_pretrain_labels.rename(
            columns={'lumo_energy': 'label'}
        )
        self.df_test = pd.read_csv("test_features.csv")
        self.df_train_combined = pd.DataFrame.merge(
            self.df_train,
            self.df_labels,
            on='Id'
        )
        self.df_pretrain_combined = pd.DataFrame.merge(
            self.df_pretrain,
            self.df_pretrain_labels,
            on='Id'
        )

    def load_dls(self, df, bs):
        """Load dataset columns into a DataLoader

        Parameters
        ----------
        label : string
            the selected target label

        Returns
        -------
        dls : TabularDataLoader
            the tabular DataLoader to be used
        df : DataFrame
            the DataFrame to be loaded
        bs : int
            batch size
        """

        splits = RandomSplitter(valid_pct=0.2)(range_of(df))
        to = TabularPandas(
            df=df,
            procs=[Categorify],
            cat_names=self.features,
            y_names='label',
            y_block=RegressionBlock,
            splits=splits,
        )
        dls = to.dataloaders(bs=bs)
        return dls

    def pre_train(self):
        """Pretrain the entire model on the pretrain dataset"""
        dls = self.load_dls(self.df_pretrain_combined, bs=64)
        metrics = [rmse]
        learn = tabular_learner(dls=dls, metrics=metrics)
        learn.unfreeze()
        lr = learn.lr_find()
        learn.fit_one_cycle(5, lr)
        self.model = learn.model

    def fit_and_predict(self):
        """Fit the last layer on the train dataset"""
        dls = self.load_dls(self.df_train_combined, bs=5)
        metrics = [rmse]
        learn = Learner(dls=dls, model=self.model, metrics=metrics)
        lr = learn.lr_find()
        learn.fit_one_cycle(20, lr)
        dl = learn.dls.test_dl(self.df_test)
        predict = learn.get_preds(dl=dl)
        self.df_predict['Id'] = self.df_test['Id']
        self.df_predict['y'] = predict[0].numpy().flatten()
        self.df_predict.to_csv('predictions.csv', index=False)


def main():
    task = Task()
    task.load_datasets()
    task.pre_train()
    task.fit_and_predict()


if __name__ == "__main__":
    main()
