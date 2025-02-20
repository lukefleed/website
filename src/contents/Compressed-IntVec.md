---
author: Luca Lombardo
datetime: 2025-02-20
title: Compressed Integer Vector Library
slug: compressed-intvec
featured: true
draft: false
tags:
  - Rust
  - Algorithms And Data Structures
ogImage: ""
description: A Rust library that implements a compressed integer vector with fast random access that stores values with instantaneous codes in a bitstream
---

I developed a Rust library for compressing vectors of `u64` integers using instantaneous codes the from dsi-bitstream library. Offers fast random access via sampling to balance speed and memory.

## Features

- **Efficient Compression**: Leverage various codecs like Gamma, Delta, and Rice coding.
- **Fast Random Access**: Achieve $O(1)$ access with configurable sampling.
- **Memory Analysis**: Integrate with [`mem-dbg`](https://crates.io/crates/mem-dbg) for memory profiling.
- **Flexible Codecs**: Select codecs based on data distribution for optimal compression.

The sampling parameter determines how often full positions are stored to speed up access. Higher values reduce memory overhead but may increase access time. For example, `sampling_param = 32` is usually a good trade-off for large datasets.

---

More can be found at:

* **Crate**: [compressed-intvec](https://crates.io/crates/compressed-intvec)
* **Documentation**: [docs.rs](https://docs.rs/compressed-intvec)
* **Repository**: [GitHub](https://github.com/lukefleed/compressed-intvec)
