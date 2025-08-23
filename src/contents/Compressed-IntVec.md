---
author: Luca Lombardo
datetime: 2025-02-20
title: Engineering a compressed integer vector in Rust
slug: compressed-intvec
featured: true
draft: false
tags:
  - Rust
  - Algorithms And Data Structures
ogImage: ""
description: A Rust library that implements a compressed integer vector with fast random access that stores values with instantaneous codes in a bitstream
---


In Rust, `Vec<T>`, where `T` is a primitive integer type like `u64` or `i32`, is the standard for contiguous, heap-allocated arrays. However, its memory layout is fundamentally tied to the static size of `T`, leading to significant waste when the dynamic range of the stored values is smaller than the type's capacity.

This inefficiency is systemic. Consider storing the value `5` within a `Vec<u64>`. Its 8-byte in-memory representation is:

<div align="center">

`00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000101`

</div>

Only 3 bits are necessary to represent the value, leaving 61 bits as zero-padding. The same principle applies, albeit less dramatically, when storing `5` in a `u32` or `u16`. At scale, this overhead becomes prohibitive. A vector of one billion `u64` elements consumes `10^9 * std::mem::size_of::<u64>()`, or approximately 8 GB of memory, even if every element could fit within a single byte.

The canonical solution is bit packing, which aligns data end-to-end in a contiguous bitstream. However, this optimization has historically come at the cost of random access performance. The O(1) access guarantee of `Vec<T>` is predicated on simple pointer arithmetic: `address = base_address + index * std::mem::size_of::<T>()`. Tightly packing the bits invalidates this direct address calculation, seemingly forcing a trade-off between memory footprint and access latency.

This raises the central question that this post aims to answer: is it possible to design a data structure that decouples its memory layout from the static size of `T`, adapting instead to the data's true dynamic range, without sacrificing the O(1) random access that makes `Vec<T>` so effective?

## Approach #1: Fixed-width bit packing with arithmetic indexing

The first approach is the most direct. If the dynamic range of our data is known, we can define a fixed `bit_width` for every integer. For instance, if the maximum value in a dataset is `1000`, we know every number can be represented in 10 bits, since `2^10 = 1024`. Instead of allocating 64 bits per element, we can store them back-to-back in a contiguous bitstream, forming the core of what from now on we'll refer as `FixedVec`

This immediately solves the space problem, but how can we find the `i`-th element in O(1) time if it doesn't align to a byte boundary? The answer lies in simple arithmetic. The starting bit position of any element is a direct function of its index. Given a backing store of `u64` words, we can locate any value by calculating its absolute bit position and then mapping that to a specific word and an offset within that word.

```rust
let bit_pos = index * bit_width;
let word_index = bit_pos / 64; // Which u64 word to read
let bit_offset = bit_pos % 64; // Where the value starts in that word
```

With these values, the implementation of `get_unchecked` becomes straightforward. It's a two-step process: fetch the correct word from our backing `Vec<u64>`, then use bitwise operations to isolate the specific bits we need.

```rust
// A simplified look at the core get_unchecked logic
unsafe fn get_unchecked(&self, index: usize) -> u64 {
    let bit_width = self.bit_width();
    let bit_pos = index * bit_width;
    let word_index = bit_pos / 64;
    let bit_offset = bit_pos % 64;

    // 1. Fetch the word from the backing store
    let word = *self.limbs.get_unchecked(word_index);

    // 2. Shift and mask to extract the value
    (word >> bit_offset) & self.mask
}
```

Let's trace an access with `bit_width = 10` for the element at `index = 7`. The starting bit position is `7 * 10 = 70`. This maps to `word_index = 1` and `bit_offset = 6`. Our 10-bit integer begins at the 6th bit of the *second* `u64` word in our storage.

The right-shift `>>` operation moves the bits of the entire `u64` word to the right by `bit_offset` positions. This aligns the start of our desired 10-bit value with the least significant bit (LSB) of the word. The final step is to isolate our value. A pre-calculated `mask` (e.g., `0b1111111111` for 10 bits) is applied with a bitwise AND `&`. This zeroes out any high-order bits from the word, leaving just our target integer.


### Crossing Word Boundaries

The single-word logic we just showed is clean and fast, but it only works as long as `bit_offset + bit_width` is less than or equal to 64. This assumption breaks down as soon as an integer's bit representation needs to cross the boundary from one `u64` word into the next.

This is guaranteed to happen for any `bit_width` that isn't a power of two. With a 10-bit width, for example, the element at `index = 6` starts at bit position 60 (`word_index = 0`, `bit_offset = 60`). Its 10 bits will occupy bits 60-63 of the first word and bits 0-5 of the second. The simple right-shift-and-mask trick fails completely here.

To correctly decode the value, we have to read *two* consecutive `u64` words from memory and stitching together the parts of our integer.

The logic splits into two paths:

1.  **Fast Path:** The integer is fully contained within a single `u64`. We do one read.
2.  **Slow Path:** The integer spans two `u64`s. We do two reads, perform some bit-shifting acrobatics, and combine the results.

This turns our `get_unchecked` implementation into something more robust:

```rust
// A more robust get_unchecked, handling spanning values
unsafe fn get_unchecked(&self, index: usize) -> u64 {
    // ... (bit_pos, word_index, bit_offset calculation) ...

    let limbs = self.limbs.as_ref();

    if bit_offset + self.bit_width <= 64 {
        // Fast path: value is fully within one word
        (*limbs.get_unchecked(word_index) >> bit_offset) & self.mask
    } else {
        // Slow path: value spans two words
        let low_part = *limbs.get_unchecked(word_index) >> bit_offset;
        let high_part = *limbs.get_unchecked(word_index + 1) << (64 - bit_offset);
        (low_part | high_part) & self.mask
    }
}
```

The line `limbs.get_unchecked(word_index + 1)` introduces a critical safety concern: what if we're trying to read the very last element in our vector, and `word_index + 1` points past the end of our allocated buffer? This would be undefined behavior. To prevent this, the `FixedVec` builder always allocates one extra padding word at the end of its storage. This small memory cost guarantees that any read, even one that spans the final word boundary, will always land in valid, allocated memory.


## The `FixedVec` Architecture

The `get_unchecked` logic is built upon a generic architecture that extends beyond a simple `Vec<u64>`. The struct itself is defined as `FixedVec<T, W, E, B>`, and its behavior is dictated by four generic parameters that control its memory layout and type interactions.

```rust
pub struct FixedVec<T: Storable<W>, W: Word, E: Endianness, B: AsRef<[W]> = Vec<W>> {
    pub(crate) bits: B,
    pub(crate) bit_width: usize,
    pub(crate) mask: W,
    pub(crate) len: usize,
    pub(crate) _phantom: PhantomData<(T, W, E)>,
}
```

The first parameter, `T`, represents the **logical element type** that the user interacts with—for example, `i16` or `u32`. The second, `W`, is the **storage word**, constrained by our `Word` trait, which determines the primitive type of the backing buffer and the granularity of bitwise operations. This allows us to use `usize` for architecture-native performance or a fixed-size type like `u64` for predictable layouts. The third parameter, `E`, is the `dsi-bitstream::Endianness` (`LE` or `BE`), controlling the bit-level interpretation within each word. Finally, `B` is the **backing storage**, constrained by `AsRef<[W]>`. This parameter abstracts ownership, allowing `FixedVec` to function as either an owned container where `B = Vec<W>` or a zero-copy, borrowed view where `B = &[W]`.

The bridge between the logical type `T` and the physical storage `W` is the `Storable` trait.

```rust
pub trait Storable<W: Word>: Sized + Copy {
    fn into_word(self) -> W;
    fn from_word(word: W) -> Self;
}
```

This trait provides a bidirectional, lossless conversion. For unsigned types, this is a direct, lossless cast. For signed types, the implementation is more involved, transparently using ZigZag encoding to map signed integers to the unsigned domain of the storage word. The goal of this mapping is to represent signed integers with small absolute values (e.g., -1, 2) as small unsigned integers.

This is handled by the `ToNat` and `ToInt` traits from `dsi-bitstream`. The encoding logic in `to_nat` is particularly intersting:

```rust
// From dsi_bitstream::traits::ToNat
fn to_nat(self) -> Self::UnsignedInt {
    (self << 1).to_unsigned() ^ (self >> (Self::BITS - 1)).to_unsigned()
}
```

The expression `(self << 1)` shifts the value left, making space at the least significant bit (LSB). The second part, `(self >> (Self::BITS - 1))`, performs an arithmetic right shift. This creates a mask that is all zeros for a non-negative number and all ones for a negative number. The final XOR combines these, effectively interleaving the positive and negative integers into the unsigned space: 0 maps to 0, -1 to 1, 1 to 2, -2 to 3, and so on.

The decoding process reverses this transformation:

```rust
// From dsi_bitstream::traits::ToInt
fn to_int(self) -> Self::SignedInt {
    (self >> 1).to_signed() ^ (-(self & 1).to_signed())
}
```

Here, `(self >> 1)` shifts the interleaved value back into its approximate position. The term `-(self & 1)` leverages two's complement to create a mask from the LSB (the original sign bit): it becomes `0` for positive numbers and `-1` (all ones) for negative numbers. The final XOR with this mask either does nothing or flips the bits to correctly restore the two's complement representation.

By embedding this logic within the trait system, the user API remains clean and operates purely in terms of `T`. This design provides compile-time guarantees that any type used with `FixedVec` has a well-defined and correct storage representation, separating the user-facing type from the raw storage words and preventing logical errors. Furthermore, the abstraction over the backing buffer `B` is what enables zero-copy views, for instance when constructing an `FixedVec` over a memory-mapped file without allocating a new heap buffer.

### Mutability: Proxy Objects

A core feature of `std::vec::Vec` is mutable, index-based access via `&mut T`. This is fundamentally impossible for `FixedVec`. An element, such as a 10-bit integer, is not a discrete, byte-aligned entity in memory. It is a virtual value extracted from a bitstream, potentially spanning the boundary of two different `u64` words. It has no stable memory address, so a direct mutable reference cannot be formed.

To provide an ergonomic mutable API, we have to emulate the behavior of a mutable reference. We can achieve this through a **proxy object pattern**, implemented in a struct named `MutProxy`.

When we call a method like `at_mut(index)`, it does not return a reference. Instead, it constructs and returns a `MutProxy` instance. The proxy's lifecycle manages the entire modification process:

1.  **Construction:** The proxy is created. Its first action is to call the parent `FixedVec`'s internal `get` logic to read and decode the value at the specified `index`. This decoded value is stored as a temporary copy inside the proxy object itself.
2.  **Modification:** The `MutProxy` implements `Deref` and `DerefMut`, allowing the user to interact with the temporary copy as if it were the real value. Any modifications (`*proxy = new_value`, `*proxy += 1`) are applied to this local copy, not to the underlying bitstream.
3.  **Destruction:** When the `MutProxy` goes out of scope, its `Drop` implementation is executed. This is the critical step where the potentially modified value from the temporary copy is taken, re-encoded, and written back into the correct bit position in the parent `FixedVec`'s storage.

This is a classic copy-on-read, write-on-drop mechanism. It provides a safe and ergonomic abstraction for mutating non-addressable data, preserving the feel of direct manipulation while correctly handling the bit-level operations under the hood. The overhead is a single read at the start of the proxy's life and a single write at the end.

### An Unaligned Optimization

The slow path of our `get_unchecked` implementation requires two separate, aligned memory reads to handle values that span word boundaries. While correct, this is not the most efficient way to access memory. It involves two potential cache misses and requires several instructions to combine the results.

Modern CPUs, particularly on the x86-64 architecture, are highly optimized for unaligned memory access. A single unaligned read is often significantly faster than two aligned reads. This can lead us to a more aggressive optimization `get` function: `get_unaligned_unchecked`. Instead of calculating which *words* to read, we calculate the exact *byte* where our data begins and perform a single `read_unaligned` of a full `W` word from that position.

The logic translates the absolute bit position into a byte address and a residual bit offset within that byte.

```rust
pub unsafe fn get_unaligned_unchecked(&self, index: usize) -> T {
    // ... (for Little-Endian architectures) ...
    let bit_pos = index * self.bit_width;
    let byte_pos = bit_pos / 8;
    let bit_rem = bit_pos % 8;

    let limbs_ptr = self.as_limbs().as_ptr() as *const u8;

    // One unaligned read is often faster than two aligned reads.
    let word: W = (limbs_ptr.add(byte_pos) as *const W).read_unaligned();

    // The result is a word where our target is now at the LSB, plus garbage bits.
    let extracted_word = word >> bit_rem;

    // The mask isolates our final value.
    Storable::<W>::from_word(extracted_word & self.mask)
}
```

This operation is fast, but it introduces a significant safety concern. An unaligned read near the end of the buffer could attempt to access bytes beyond our allocated memory, resulting in undefined behavior. This is precisely why the `FixedVec` builder guarantees a padding word at the end of its storage. This padding ensures that even the most aggressive unaligned read near the vector's end will always access valid, allocated memory, making the optimization safe.

This optimization is implemented specifically for Little-Endian architectures. On these systems, the byte order allows for a simple right-shift to align the data after an unaligned read. The equivalent logic for Big-Endian is significantly more complex and often less performant than the two-read approach. For this reason, `get_unaligned_unchecked` on Big-Endian architectures falls back to the standard `get_unchecked` implementation. 

### Random Access Performance

We can now benchmark the latency of 1 million random `get_unchecked` operations on a vector containing 10 million elements. For each `bit_width`, the data is generated with a uniform random distribution in the range `[0, 2^bit_width)`.

Our baseline is the smallest standard `Vec<T>` capable of holding the data (`Vec<u8>` for `bit_width <= 8`, etc.). This provides a direct comparison against the optimal standard library implementation. We also include results from [`sux::BitFieldVec`], [`succinct::IntVector`], and [`simple-sds-sbwt::IntVector`] for context.

{% src/pages/bench-intvec/fixed_random_access_performance.html %}

The data shows that for `bit_width` values below 32, the `get_unaligned_unchecked` of our `FixedVec (Unaligned)` implementation exhibits lower latency than the corresponding `Vec<T>` baseline. This is a result of improved **cache locality**. A 64-byte L1 cache line can hold 64 elements from a `Vec<u8>`. With a `bit_width` of 4, the same cache line holds `(64 * 8) / 4 = 128` elements from our `FixedVec`. This increased data density improves the cache hit rate for random access patterns, and the latency reduction from avoiding DRAM access outweighs the instruction cost of the bitwise extraction.

The performance delta between `get_unaligned_unchecked` and `get_unchecked` also validates the unaligned access strategy. On x86-64, a single `read_unaligned` instruction is more efficient than the two dependent aligned reads required by the logic for spanning words. This results in better instruction pipeline utilization.

Consequently, for data with a small dynamic range that might otherwise be stored in a `Vec<u64>` for API convenience, our `FixedVec` is not only smaller but can be 2-3x faster for random reads due to the significant reduction in memory bandwidth requirements.

[`sux::BitFieldVec`]: https://docs.rs/sux/latest/sux/bits/bit_field_vec/index.html
[`succinct::IntVector`]: https://docs.rs/succinct/latest/succinct/trait.IntVec.html
[`simple-sds-sbwt::IntVector`]: https://docs.rs/simple-sds-sbwt/latest/simple_sds_sbwt/int_vector/struct.IntVector.html