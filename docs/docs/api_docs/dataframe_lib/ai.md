# AI
Extensions to the DataFrame library for AI training and inference.

## bodo.ai.tokenize

Refer to [`bodo.pandas.BodoSeries.ai.tokenize`](./series/ai/tokenize.md).

## bodo.ai.embed

Refer to [`bodo.pandas.BodoSeries.ai.embed`](./series/ai/embed.md).

## bodo.ai.llm_generate

Refer to [`bodo.pandas.BodoSeries.ai.llm_generate`](./series/ai/llm_generate.md).

## bodo.ai.prepare_model

``` py
def prepare_model(
    model,
    parallel_strategy: Literal["ddp", "fsdp"] | None = "ddp",
    parallel_strategy_kwargs: dict[str, Any] | None = None,
) -> torch.nn.Module | None:
```

Prepares a PyTorch model for distributed training using the specified parallel strategy. Should only be used within train_loop_per_worker function passed to `bodo.ai.torch_train`. This function also transfers the model to the appropriate device, if accelerators are available. If there are less accelealators than workers some workers will return None which indicates they should not perform training. 

<p class="api-header">Parameters</p>
: __model: *torch.nn.Module*__: The PyTorch model to prepare for distributed training.
: __parallel_strategy: *Literal["ddp", "fsdp"] | None*__: The parallel strategy to use. Supported strategies are:
    - "ddp": Distributed Data Parallel
    - "fsdp": Fully Sharded Data Parallel 
    If None, no parallel strategy is applied and the model is returned as-is.
: __parallel_strategy_kwargs: *dict[str, Any] | None*__: Additional keyword arguments to pass to the parallel strategy constructor (e.g., `torch.nn.parallel.DistributedDataParallel` or `torch.distributed.fsdp.FullyShardedDataParallel`).

## bodo.ai.prepare_dataset

``` py

def prepare_dataset(
    data: DataFrame | Series,
    batch_size: int,
    shuffle: bool = True,
    dataset_func: Callable | None = None,
    collate_fn: Callable = None,
    pin_memory: bool = False,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
```

Prepares a dataset for distributed training by converting a DataFrame or Series into a DataLoader. This function partitions the data among workers, applies optional shuffling, and creates batches for training.

<p class="api-header">Parameters</p>
: __data: *DataFrame | Series*__: The DataFrame or Series to be converted into a dataset.
: __batch_size: *int*__: The size of each batch to be created.
: __shuffle: *bool*__: Whether to shuffle the data before creating batches. Default is True.
: __dataset_func: *Callable | None*__: An optional function to customize the dataset creation process. If provided, this function should accept a DataFrame or Series and return a PyTorch Dataset. If None, a default dataset creation process is used that converts each row into a tensor.
: __collate_fn: *Callable*__: An optional function to customize how batches are collated. Default is None, which uses the default collation behavior of PyTorch DataLoader.
: __pin_memory: *bool*__: Whether to pin memory for faster data transfer to GPU. Default is False.
: __seed: *int*__: A seed value for shuffling the data to ensure reproducibility. Default is 0.

## bodo.ai.torch_train

``` py
def torch_train(
    train_loop_per_worker: Callable[
        [], None] | 
        Callable[[dict], None],
    *args,
    **kwargs
) -> None:
```

Trains a PyTorch model in a distributed manner across multiple workers using the provided training loop function. This function initializes the distributed environment, partitions the dataset, and executes the training loop on each worker.

<p class="api-header">Parameters</p>
: __train_loop_per_worker: *Callable[[], None] | Callable[[dict], None]*__: A user-defined function that contains the training logic to be executed on each worker. This function can optionally accept a dictionary of configuration parameters that will be passed to it.
: __*args: *__: Positional arguments to be passed to the `train_loop_per_worker` function.
: __**kwargs: *__: Keyword arguments to be passed to the `train_loop_per_worker` function.

Example:
The following example demonstrates how to use `bodo.ai.torch_train` to train a simple neural network on a dataset. The training loop is able to handle both CPU and GPU training based on the available hardware. If you know which you will be training on you can simplify the code by removing the irrelevant code.

```py
import bodo.pandas as pd
import bodo.ai.train
import tempfile

df = bd.DataFrame(
    {
        "feature1": pd.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype="float32"),
        "feature2": pd.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype="float32"),
        "label": pd.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0], dtype="float32"),
    }
)

def train_loop(data, config):
    import torch
    import torch.distributed.checkpoint
    import torch.nn as nn

    # Simple linear regression model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(2, 32)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(32, 1)

        def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))

    model = SimpleModel()
    model = bodo.ai.train.prepare_model(model, parallel_strategy="ddp")
    dataloader = bodo.ai.train.prepare_dataset(
        data, batch_size=config.get("batch_size", 2)
    )
    if model is None:
        return

    # train on data
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = next(model.parameters()).device
    for epoch in range(config.get("epochs", 5)):
        for batch in dataloader:
            batch = batch.to(device)
            inputs = batch[:, :2]
            labels = batch[:, 2].unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Create checkpoint.
        base_model = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        torch.distributed.checkpoint.save(
            {"model_state_dict": base_model.state_dict()},
            checkpoint_id=config["checkpoint_dir"],
        )
    print(f"Epoch {epoch}, Loss: {loss.item()}")

bodo.ai.train.torch_train(
    train_loop,
    df,
    {"batch_size": 2, "checkpoint_dir": tempfile.mkdtemp("checkpoint_dir")},
)
```
