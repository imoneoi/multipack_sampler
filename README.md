# Multipack Sampler

The Multipack sampler is designed for padding-free distributed training of large language models. It utilizes an approximate solution to the identical machine scheduling problem to maximize the efficiency of batch processing. On the OpenChat V1 training set, it achieves >99% theoretical efficiency, while the interleaved sampler only achieves ~75%.

## V2 Update

Multipack V2 optimized the packing algorithm complexity from `O(n k log n)` down to `O(n log k log n)` without degrading the packing efficiency, achieving better throughput for a large number of nodes.

The V2 release also has two variants with different packing optimization objective:

 - `MultipackDistributedBatchSampler`: Designed for models with quadratic attention. It will try to optimize packing efficiency as well as balance long/short sequences between each nodes, to minimize the difference of quadratic load.
 - `MultipackDistributedBatchSampler_LinearAttention`: For models with linear attention. Only consider packing efficiency and performs better on it than Quadratic variant, however this algorithm tends to put all long sequences into one node.

## Benchmark

Please refer to `test_multipack.ipynb`

- Efficiency: Percentage of actual batch size to max batch size

   = `number of tokens per batch / max capacity of tokens per batch`

 - Utilization: all nodes waiting for the slowest node
 
   = `number of tokens per batch / max number of tokens on a single node * node count`

L^2 lag: `sqrt(max over node(sum length^2) - min over node(sum length^2))`

```
OpenChat V1 (testdata.json)

Sampler Multipack QuadraticAttention:
Batch count for ranks: [37, 37, 37, 37, 37, 37, 37, 37]
Packing Time: 20ms

L^2 lag avg: 438 max: 717
Efficiency: 98.16%
Utilization: 99.70%
==========

Sampler Multipack LinearAttention:
Batch count for ranks: [36, 36, 36, 36, 36, 36, 36, 36]
Packing Time: 18ms

L^2 lag avg: 6500 max: 6761
Efficiency: 99.64%
Utilization: 99.64%
==========

Sampler Interleaved:
Batch count for ranks: [48, 48, 48, 48, 48, 48, 48, 48]
Packing Time: 0ms

L^2 lag avg: 1914 max: 2000
Efficiency: 75.67%
Utilization: 96.79%
==========
```

## Usage

Compatible with PyTorch `DataLoader`

```python
batch_max_len = 16 * 2048  # batch size * max context length

lengths = np.array([len(tokens) for tokens in data])

sampler = MultipackDistributedBatchSampler(
    batch_max_length=batch_max_len,
    lengths=lengths,
    seed=0
)

dataloader = DataLoader(data, batch_sampler=sampler)
```

## License

MIT
