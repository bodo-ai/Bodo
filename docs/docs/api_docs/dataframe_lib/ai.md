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

## bodo.ai.torch_train

``` py
def torch_train(
    train_loop_per_worker: Callable[
        [], None] | 
        Callable[[dict], None],
    dataset: BodoDataFrame | BodoSeries,
    train_loop_config: dict | None = None,
) -> None:
```

Trains a PyTorch model in a distributed manner across multiple workers using the provided training loop function. This function initializes the distributed environment, partitions the dataset, and executes the training loop on each worker.

<p class="api-header">Parameters</p>
: __train_loop_per_worker: *Callable[[], None] | Callable[[dict], None]*__: A user-defined function that contains the training logic to be executed on each worker. This function can optionally accept a dictionary of configuration parameters that will be passed to it.
: __dataset: *BodoDataFrame | BodoSeries*__: The dataset to be used for training. This will be partitioned across the workers.
: __train_loop_config: *dict | None*__: A dictionary of configuration parameters to be passed to the `train_loop_per_worker` function. This can include hyperparameters, model configurations, or any other settings needed for training.

Example:
The following example demonstrates how to use `bodo.ai.torch_train` to train a simple neural network on a dataset. The training loop is able to handle both CPU and GPU training based on the available hardware. If you know which you will be training on you can simplify the code by removing the irrelevant code.

``` py
import bodo.pandas as pd
import tempfile

df = bd.DataFrame(
    {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "label": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
    }
)

def train_loop(data, config):
    import torch
    import torch.nn as nn
    import torch.distributed.checkpoint

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(2, 32)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(32, 1)

        def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))

    model = SimpleModel()
    model = bodo.ai.train.prepare_model(model)
    gpu_ranks = bodo.get_gpu_ranks()
    if model is None:
        # Not a worker process so send the data to the GPU workers
        bodo.rebalance(data, dests=gpu_ranks)
        return

    model_device = next(model.parameters()).device
    if model_device.type != "cpu":
        # If we're using an accelerator, rebalance data to match GPU ranks
        data = bodo.rebalance(data, dests=gpu_ranks)

    # train on data
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(config.get("epochs", 5)):
        batch_size = config.get("batch_size", 2)
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batch_tensor = torch.tensor(batch.to_numpy("float32")).to(model_device)
            inputs = batch_tensor[:, :2]
            labels = batch_tensor[:, 2].unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

         # Create checkpoint.
        base_model = (model.module
            if isinstance(model, DistributedDataParallel) else model)
        torch.distributed.checkpoint.state_dict_saver.save(
            {"model_state_dict": base_model.state_dict()},
            checkpoint_id=config["checkpoint_dir"],
        )

        print(f"Epoch {epoch}, Loss: {loss.item()}")


bodo.ai.train.torch_train(train_loop, df, {"batch_size": 2, "checkpoint_dir": tempfile.mkdtemp("checkpoint_dir")})
