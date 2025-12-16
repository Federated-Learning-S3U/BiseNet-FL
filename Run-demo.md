# Steps To Run FL Setup

1. Create Python3.11 Virtual Environment
```
python3.11 -m venv env
source env/bin/activate
```


2. Install dependencies defined in `pyproject.toml`
```
pip install -e .
```

3. Run the flwr app, note the flwr app takes its configs from the `fl-cityscapes-bisenetv2/pyproject.toml`, so modify the configs there as required
```
flwr run fl-cityscapes-bisenetv2
``` 