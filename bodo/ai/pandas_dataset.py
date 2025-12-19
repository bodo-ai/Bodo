import pandas as pd
import torch


class PandasDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for Bodo/Pandas DataFrames.
    When rows are accessed, they are converted to a PyTorch tensor.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df (pd.DataFrame): The Pandas DataFrame to wrap, all columns must have the same dtype so they can be converted to a single tensor.
            device (torch.device | None): The device to place the tensors on.
        """
        self.df = df
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
        tensor = torch.tensor(row.to_numpy(dtype=self.dtype))
        return tensor

    def __getitems__(self, idxs):
        rows = self.df.iloc[idxs]
        tensor = torch.tensor(rows.to_numpy(dtype=self.dtype))
        return [tensor.select(0, i) for i in range(tensor.shape[0])]
