# Multipack Sampler

The Multipack sampler is designed for padding-free distributed training of large language models. It utilizes an approximate solution to the identical machine scheduling problem to maximize the efficiency of batch processing. On the OpenChat V1 training set, it achieves >99% theoretical efficiency, while the interleaved sampler only achieves ~75%.

## Benchmark

Please refer to `test_multipack.ipynb`

```
OpenChat V1 (testdata.json)

Sampler Multipack:
Overall Efficiency: 0.9963896327548557

Sampler Interleaved:
Overall Efficiency: 0.756684939066569
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
