# Exploration for LLMs

[![PyPI - Version](https://img.shields.io/pypi/v/ellm.svg)](https://pypi.org/project/ellm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ellm.svg)](https://pypi.org/project/ellm)

-----

## Table of Contents

- [Exploration for LLMs](#exploration-for-llms)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [License](#license)

## Installation

```console
pip install -r requirements.txt
pip install flash-attn==2.6.3
pip install -e .
export DS_SKIP_CUDA_CHECK=1 # suppress adam offload error
```

## License

`ellm` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
