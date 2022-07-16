# 直接 Reduce
+ 网络结构
```
[V] Engine Layer Information:
Layer(Reduce): ReduceSum, Tactic: 7, tensor0[Float(1,256,1024)] -> tensor1[Float(1,1,1024)]
```

+ 性能测试结果
```
[05/06/2022-04:57:00] [I] === Performance summary ===
[05/06/2022-04:57:00] [I] Throughput: 37418.4 qps
[05/06/2022-04:57:00] [I] Latency: min = 0.0214844 ms, max = 4.28027 ms, mean = 0.0225213 ms, median = 0.0224609 ms, percentile(99%) = 0.0307159 ms
[05/06/2022-04:57:00] [I] End-to-End Host Latency: min = 0.0214844 ms, max = 4.28027 ms, mean = 0.0225213 ms, median = 0.0224609 ms, percentile(99%) = 0.0307159 ms
[05/06/2022-04:57:00] [I] Enqueue Time: min = 0.00170898 ms, max = 0.0249329 ms, mean = 0.00225226 ms, median = 0.00213623 ms, percentile(99%) = 0.00415039 ms
[05/06/2022-04:57:00] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/06/2022-04:57:00] [I] GPU Compute Time: min = 0.0214844 ms, max = 4.28027 ms, mean = 0.0225213 ms, median = 0.0224609 ms, percentile(99%) = 0.0307159 ms
[05/06/2022-04:57:00] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/06/2022-04:57:00] [I] Total Host Walltime: 3.00005 s
[05/06/2022-04:57:00] [I] Total GPU Compute Time: 2.52817 s
```

# 添加一对 Transpose
+ 网络结构
```
[V] Engine Layer Information:
Layer(Reduce): ReduceSum, Tactic: 7, tensor0[Float(1,256,1024)] -> tensor1[Float(1,1,1024)]
[05/06/2022-04:57:04] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
```

+ 性能测试结果
```
[05/06/2022-04:57:07] [I] === Performance summary ===
[05/06/2022-04:57:07] [I] Throughput: 34834.3 qps
[05/06/2022-04:57:07] [I] Latency: min = 0.0214844 ms, max = 0.0358429 ms, mean = 0.0225967 ms, median = 0.022522 ms, percentile(99%) = 0.0317383 ms
[05/06/2022-04:57:07] [I] End-to-End Host Latency: min = 0.0214844 ms, max = 0.0358429 ms, mean = 0.0225967 ms, median = 0.022522 ms, percentile(99%) = 0.0317383 ms
[05/06/2022-04:57:07] [I] Enqueue Time: min = 0.00195312 ms, max = 0.016983 ms, mean = 0.00266382 ms, median = 0.00256348 ms, percentile(99%) = 0.00427246 ms
[05/06/2022-04:57:07] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/06/2022-04:57:07] [I] GPU Compute Time: min = 0.0214844 ms, max = 0.0358429 ms, mean = 0.0225967 ms, median = 0.022522 ms, percentile(99%) = 0.0317383 ms
[05/06/2022-04:57:07] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/06/2022-04:57:07] [I] Total Host Walltime: 3.00006 s
[05/06/2022-04:57:07] [I] Total GPU Compute Time: 2.36147 s
```

# 添加一对 Transpose，并在第一个 Transpose 后添加一个 Identity
+ 网络结构
```
[V] Engine Layer Information:
Layer(Shuffle): Transpose-0, Tactic: 1, tensor0[Float(1,256,1024)] -> tensor2[Float(1,1024,256)]
Layer(NoOp): Identity-0, Tactic: 0, tensor2[Float(1,1024,256)] -> tensor4[Float(1,1024,256)]
Layer(Reduce): ReduceSum, Tactic: 2, tensor4[Float(1,1024,256)] -> tensor3[Float(1,1024,1)]
Layer(NoOp): Transpose-1, Tactic: 0, tensor3[Float(1,1024,1)] -> tensor1[Float(1,1,1024)]
```

+ 性能测试结果
```
[05/06/2022-04:57:15] [I] === Performance summary ===
[05/06/2022-04:57:15] [I] Throughput: 51315.5 qps
[05/06/2022-04:57:15] [I] Latency: min = 0.0131836 ms, max = 0.209961 ms, mean = 0.0140909 ms, median = 0.0143127 ms, percentile(99%) = 0.0144043 ms
[05/06/2022-04:57:15] [I] End-to-End Host Latency: min = 0.0131836 ms, max = 0.209961 ms, mean = 0.0140909 ms, median = 0.0143127 ms, percentile(99%) = 0.0144043 ms
[05/06/2022-04:57:15] [I] Enqueue Time: min = 0.00170898 ms, max = 0.206543 ms, mean = 0.00222832 ms, median = 0.00201416 ms, percentile(99%) = 0.00402832 ms
[05/06/2022-04:57:15] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/06/2022-04:57:15] [I] GPU Compute Time: min = 0.0131836 ms, max = 0.209961 ms, mean = 0.0140909 ms, median = 0.0143127 ms, percentile(99%) = 0.0144043 ms
[05/06/2022-04:57:15] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/06/2022-04:57:15] [I] Total Host Walltime: 3.00003 s
[05/06/2022-04:57:15] [I] Total GPU Compute Time: 2.16926 s
```
