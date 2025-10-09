import pandas as pd
import torch


class PandasDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, device: torch.device | None = None):
        self.df = df
        self.device = device
        # Ensure all columns are the same dtype
        assert len(df.dtypes.unique()) == 1, (
            "PandasDataset: All columns must have the same dtype"
        )
        self.dtype = df.dtypes.iloc[0]
        if isinstance(self.dtype, pd.ArrowDtype):
            self.dtype = self.dtype.pyarrow_dtype.to_pandas_dtype()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tensor = torch.tensor(row.to_numpy(dtype=self.dtype), device=self.device)
        return tensor

    def __getitems__(self, idxs):
        rows = self.df.iloc[idxs]
        tensor = torch.tensor(rows.to_numpy(dtype=self.dtype), device=self.device)
        return [tensor.select(0, i) for i in range(tensor.shape[0])]
