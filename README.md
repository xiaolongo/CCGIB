## Towards Consistency and Complementarity: Multiview Graph Representation Learning via Variational Information Bottleneck

All experiments are conducted with the following setting：
- Operating system: Ubuntu Linux release 20.04
- CPU: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz 
- GPU: NVIDIA GeForce RTX 3090 graphics card 
- Software version: Python 3.8, NumPy 1.20.1, Scipy 1.6.1,  PyTorch 1.11.0, PyTorch Geometric 2.0.4, GraKel 0.1.8

Dataset should be automatically downloaded when you have pytorch_geometric installed properly.

Running commmand: 

For graph classification, run the following command:
```shell
python3 mvgib.py --dataset MUTAG --gpu_id 0 --batch_size 128 --view1 adj --view2 KNN
```

For graph cluster, run the following command:
```shell
python3 mvgib_cluster.py --dataset MUTAG --gpu_id 0 --batch_size 128 --view1 adj --view2 KNN
```
