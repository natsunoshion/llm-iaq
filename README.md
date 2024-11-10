# NLP 大作业

## Install

1. Clone this repository and navigate to IAQ folder
```
git clone https://github.com/mit-han-lab/llm-iaq
cd llm-iaq
```

2. Install Package
```
conda create -n iaq python=3.10 -y
conda activate iaq
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install efficient W4A16 (4-bit weight, 16-bit activation) CUDA kernel and optimized FP16 kernels (e.g. layernorm, positional encodings).
```
cd iaq/kernels
python setup.py install
```