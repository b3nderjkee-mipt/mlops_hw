# Report Hw3

## System configuration

1. Windows 10
2. Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.20 GHz
3. vCPU = 4, vRAM = 4

## Task description

Cnn for MNIST image classification.

## Model repository tree

```
└───model_repository
    └───onnx-cnn
        └───1
```

## Throughput and Latency

Before optimization

- Concurrency: 1, throughput: 967.488 infer/sec, latency 1032 usec
- Concurrency: 2, throughput: 1490.14 infer/sec, latency 1341 usec
- Concurrency: 3, throughput: 1375.15 infer/sec, latency 2180 usec
- Concurrency: 4, throughput: 1464.87 infer/sec, latency 2729 usec
- Concurrency: 5, throughput: 1392.85 infer/sec, latency 3588 usec

After optimization

- Concurrency: 1, throughput: 633.317 infer/sec, latency 1577 usec
- Concurrency: 2, throughput: 2481.69 infer/sec, latency 804 usec
- Concurrency: 3, throughput: 1965.76 infer/sec, latency 1525 usec
- Concurrency: 4, throughput: 2383.69 infer/sec, latency 1676 usec
- Concurrency: 5, throughput: 3177.52 infer/sec, latency 1572 usec

## Motivation for choosing parameters

All parameters were selected in order to improve throughput and latency. There have been many attempts to change the parameters in different ways. The best parameters have been selected (in my opinion in terms of indicators), which have significantly improved the indicators.
