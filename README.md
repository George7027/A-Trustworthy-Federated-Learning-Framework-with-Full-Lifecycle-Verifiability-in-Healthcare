# 1. Introduction

This project is based on **PyTorch 1.11.0** and **Python 3.9.18**.  
The datasets currently used include **MNIST**, **CIFAR-10** and **MITBIH**.  
Our framework uses **FedAVG** as the baseline federated learning algorithm.

# 2. Usage

You can configure parameters by modifying the `conf.json` file.  
Then, run the following command:

```bash
python server.py -c ./utils/conf.json
