---
author: Luca Lombardo
datetime: 2025-02-20
title: Engineering a Fixed-Width bit-packed Integer Vector in Rust
slug: compressed-intvec
featured: true
draft: false
tags:
  - Rust
  - Algorithms And Data Structures
ogImage: ""
description: Design and implementation of a memory-efficient, fixed-width bit-packed integer vector in Rust, achieving O(1) random access.
---

In Rust, `Vec<T>`, where `T` is a primitive integer type like `u64` or `i32`, is the standard for contiguous, heap-allocated arrays. However, its memory layout is fundamentally tied to the static size of `T`, leading to significant waste when the dynamic range of the stored values is smaller than the type's capacity.

This inefficiency is systemic. Consider storing the value `5` within a `Vec<u64>`. Its 8-byte in-memory representation is:

<div align="center">

`00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000101`

</div>

Only 3 bits are necessary to represent the value, leaving 61 bits as zero-padding. The same principle applies, albeit less dramatically, when storing `5` in a `u32` or `u16`. At scale, this overhead becomes prohibitive. A vector of one billion `u64` elements consumes `10^9 * std::mem::size_of::<u64>()`, or approximately 8 GB of memory, even if every element could fit within a single byte.

The canonical solution is bit packing, which aligns data end-to-end in a contiguous bitstream. However, this optimization has historically come at the cost of random access performance. The O(1) access guarantee of `Vec<T>` is predicated on simple pointer arithmetic: `address = base_address + index * std::mem::size_of::<T>()`. Tightly packing the bits invalidates this direct address calculation, seemingly forcing a trade-off between memory footprint and access latency.

This raises the central question that this post aims to answer: is it possible to design a data structure that decouples its memory layout from the static size of `T`, adapting instead to the data's true dynamic range, without sacrificing the O(1) random access that makes `Vec<T>` so effective?

## Fixed-width bit packing with arithmetic indexing

If the dynamic range of our data is known, we can define a fixed `bit_width` for every integer. For instance, if the maximum value in a dataset is `1000`, we know every number can be represented in 10 bits, since `2^10 = 1024`. Instead of allocating 64 bits per element, we can store them back-to-back in a contiguous bitstream, forming the core of what from now on we'll refer as `FixedVec`

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

The line `limbs.get_unchecked(word_index + 1)` introduces a critical safety concern: what if we're trying to read the very last element in our vector, and `word_index + 1` points past the end of our allocated buffer? This would be undefined behavior. To prevent this, we must allocate one extra padding word at the end of the storage. This small memory cost guarantees that any read, even one that spans the final word boundary, will always land in valid, allocated memory.

### Unaligned Reads

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

## The `FixedVec` Architecture

Now that we have a solid understanding of the access patterns, we need to think about the overall architecture of this structore. The `FixedVec` must be generic over several parameters that control its memory layout and type interactions:

```rust
pub struct FixedVec<T: Storable<W>, W: Word, E: Endianness, B: AsRef<W]> = Vec<W>> {
    pub(crate) bits: B,
    pub(crate) bit_width: usize,
    pub(crate) mask: W,
    pub(crate) len: usize,
    pub(crate) _phantom: PhantomData<(T, W, E)>,
}
```

Where `T` is the **logical element type** (`i16`, `u32`), `W` is the **storage word** (`u64`, `usize`), `E` is the `dsi-bitstream::Endianness`, and `B` is the **backing storage** (`Vec<W>` or `&[W]`). 

Now we need a robust trait system.

### Abstracting the storage word

The first step is to abstract the physical storage layer. The `W` parameter cannot be just any unsigned integer; it needs a specific set of capabilities. We can define these requirements in the `Word` trait.

```rust
pub trait Word:
    UnsignedInt + Bounded + ToPrimitive + dsi_bitstream::traits::Word
    + NumCast + Copy + Send + Sync + Debug + IntoAtomic + 'static
{
    const BITS: usize = std::mem::size_of::<Self>() * 8;
}
```

The numeric traits (`UnsignedInt`, `Bounded`, `NumCast`, `ToPrimitive`) are necessary for the arithmetic of offset and mask calculations. The `dsi_bitstream::traits::Word` bound allows us to integrate with its `BitReader` and `BitWriter` implementations, offloading the bitstream logic. `Send` and `Sync` are non-negotiable requirements for any data structure that might be used in a concurrent context. The `IntoAtomic` bound is particularly forward-looking: it establishes a compile-time link between a storage word like `u64` and its atomic counterpart, `AtomicU64`. We will use it later to build a thread safe, atomic version of `FixedVec`. Finally, the `const BITS` associated constant lets us write architecture-agnostic code that correctly adapts to `u32`, `u64`, or `usize` words without `cfg` flags.

### Bridging Types with the `Storable` Trait

With the physical storage layer defined, we need a formal contract to connect it to the user's logical type `T`. We can do this by creating the `Storable` trait, which defines a bidirectional, lossless conversion.

```rust
pub trait Storable<W: Word>: Sized + Copy {
    fn into_word(self) -> W;
    fn from_word(word: W) -> Self;
}
```

For unsigned types, the implementation is a direct cast. For signed types, however, we must map the `iN` domain to the `uN` domain required for bit-packing. A simple two's complement bitcast is unsuitable, as `i64(-1)` would become `u64::MAX`, a value requiring the maximum number of bits.

We can use **ZigZag encoding**, which maps integers with small absolute values to small unsigned integers. This is implemented via the `ToNat` and `ToInt` traits from `dsi-bitstream`. The core encoding logic in `to_nat` is:

```rust
// From dsi_bitstream::traits::ToNat
fn to_nat(self) -> Self::UnsignedInt {
    (self << 1).to_unsigned() ^ (self >> (Self::BITS - 1)).to_unsigned()
}
```

This operation works as follows: `(self << 1)` creates a space at the LSB. The term `(self >> (Self::BITS - 1))` is an arithmetic right shift, which generates a sign mask—all zeros for non-negative numbers, all ones for negative numbers. The final XOR uses this mask to interleave positive and negative integers: 0 becomes 0, -1 becomes 1, 1 becomes 2, -2 becomes 3, and so on.

The decoding reverses this transformation:

```rust
// From dsi_bitstream::traits::ToInt
fn to_int(self) -> Self::SignedInt {
    (self >> 1).to_signed() ^ (-(self & 1).to_signed())
}
```

Here, `(self >> 1)` shifts the value back. The term `-(self & 1)` creates a mask from the LSB (the original sign bit). In two's complement, this becomes `0` for even numbers (originally positive) and `-1` (all ones) for odd numbers (originally negative). The final XOR with this mask correctly restores the original two's complement representation.

By encapsulating this logic within the trait system, the main `FixedVec` implementation remains clean and agnostic to the signedness of the data it stores. This is a zero-cost abstraction that provides both safety and specialization at compile time.


## Constructing `FixedVec`: The Builder Pattern

Once the structure logic is in place, we have to design an ergonomic way to construct it. A simple `new()` function isn't sufficient because the vector's memory layout depends on parameters that must be determined *before* allocation, most critically the `bit_width`. This is a classic scenario for the builder pattern.

The central problem is that the optimal `bit_width` often depends on the data itself. We need a mechanism to specify the *strategy* for determining this width. We can create the `BitWidth` enum:

```rust
pub enum BitWidth {
    Minimal,
    PowerOfTwo,
    Explicit(usize),
}
```

This enum decouples the user's intent from the implementation details. `Minimal` signals a desire for maximum space efficiency, requiring an initial pass over the data to find the maximum value. `PowerOfTwo` signals a preference for a layout that might be more performant for certain operations, again requiring a data scan. `Explicit(n)` is the escape hatch for when the `bit_width` is known ahead of time, allowing us to skip the data scan entirely.

With this, the `FixedVecBuilder` could be designed as a state machine. It holds the chosen `BitWidth` strategy. The final `build()` method takes the input slice and executes the appropriate logic.

```rust
// A look at the builder's logic flow
pub fn build(self, input: &[T]) -> Result<FixedVec<...>, Error> {
    
    let final_bit_width = match self.bit_width_strategy {
        BitWidth::Explicit(n) => n,
        _ => {
            // For Minimal or PowerOfTwo, we first find the max value.
            let max_val = input.iter().map(|v| v.into_word()).max().unwrap_or(0);
            let min_bits = (64 - max_val.leading_zeros()).max(1) as usize;

            match self.bit_width_strategy {
                BitWidth::Minimal => min_bits,
                BitWidth::PowerOfTwo => min_bits.next_power_of_two(),
                _ => unreachable!(),
            }
        }
    };

    // ... (rest of the logic: allocate buffer, write data) ...
}
```

This design cleanly separates the configuration phase from the execution phase. The user can declaratively state their requirements, and the builder handles the implementation details, whether that involves a full data scan or a direct construction. For example:

```rust
use compressed_intvec::prelude::*;

let data: &[u32] = &[100, 200, 500]; // Max value 500 requires 9 bits

// The builder will scan the data, find max=500, calculate min_bits=9,
// and then round up to the next power of two.
let vec_pow2: UFixedVec<u32> = FixedVec::builder()
    .bit_width(BitWidth::PowerOfTwo)
    .build(data)
    .unwrap();
    
assert_eq!(vec_pow2.bit_width(), 16);
```

This approach makes the API expressive and robust, handling the different construction paths internally while exposing a simple, unified interface.


## Random Access Performance

We can now benchmark the latency of 1 million random access operations on a vector containing 10 million elements. For each `bit_width`, the data is generated with a uniform random distribution in the range `[0, 2^bit_width)`. The code for the benchmark is available here: [`bench-intvec`]

Our baseline is the smallest standard `Vec<T>` capable of holding the data (`Vec<u8>` for `bit_width <= 8`, etc.). This provides a direct comparison against the optimal standard library implementation. We also include results from [`sux::BitFieldVec`], [`succinct::IntVector`], and [`simple-sds-sbwt::IntVector`] for context.

{% src/pages/bench-intvec/fixed_random_access_performance.html %}

We can see that for `bit_width` values below 32, the `get_unaligned_unchecked` of our `FixedVec (Unaligned)` implementation exhibits lower latency than the corresponding `Vec<T>` baseline. This is a result of improved **cache locality**. A 64-byte L1 cache line can hold 64 elements from a `Vec<u8>`. With a `bit_width` of 4, the same cache line holds `(64 * 8) / 4 = 128` elements from our `FixedVec`. This increased data density improves the cache hit rate for random access patterns, and the latency reduction from avoiding DRAM access outweighs the instruction cost of the bitwise extraction. For values of `bit_width` above 32, the performance of `FixedVec` are very slightly worse than the `Vec<T>` baseline, as the cache locality advantage diminishes. However, the memory savings remain.

The performance delta between `get_unaligned_unchecked` and `get_unchecked` also validates the unaligned access strategy. On x86-64, a single `read_unaligned` instruction is more efficient than the two dependent aligned reads required by the logic for spanning words. This results in better instruction pipeline utilization.


[`sux::BitFieldVec`]: https://docs.rs/sux/latest/sux/bits/bit_field_vec/index.html
[`sux-rs`]: https://crates.io/crates/sux
[`succinct::IntVector`]: https://docs.rs/succinct/latest/succinct/trait.IntVec.html
[`simple-sds-sbwt::IntVector`]: https://docs.rs/simple-sds-sbwt/latest/simple_sds_sbwt/int_vector/struct.IntVector.html
[`bench-intvec`]: https://github.com/lukefleed/compressed-intvec/blob/master/benches/fixed/bench_random_access.rs


# Doing more than just reading

The design of `FixedVec` allows for more than just efficient reads. We can extend it to support mutation and even thread-safe (almost atomic) concurrent access.

### Mutability: Proxy Objects

A core feature of `std::vec::Vec` is mutable, index-based access via `&mut T`. This is fundamentally impossible for `FixedVec`. An element, such as a 10-bit integer, is not a discrete, byte-aligned entity in memory. It is a virtual value extracted from a bitstream, potentially spanning the boundary of two different `u64` words. It has no stable memory address, so a direct mutable reference cannot be formed.

To provide an ergonomic mutable API, we must emulate the behavior of a mutable reference. We achieve this through a proxy object pattern, implemented in a struct named `MutProxy`.

```rust
pub struct MutProxy<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    vec: &'a mut FixedVec<T, W, E, B>,
    index: usize,
    value: T, // A temporary, decoded copy of the element's value.
}
```

When we call a method like `at_mut(index)`, it does not return a reference. Instead, it constructs and returns a `MutProxy` instance. The proxy's lifecycle manages the entire modification process:

1.  **Construction:** The proxy is created. Its first action is to call the parent `FixedVec`'s internal `get` logic to read and decode the value at the specified `index`. This decoded value is stored as a temporary copy inside the proxy object itself.
2.  **Modification:** The `MutProxy` implements `Deref` and `DerefMut`, allowing the user to interact with the temporary copy as if it were the real value. Any modifications (`*proxy = new_value`, `*proxy += 1`) are applied to this local copy, not to the underlying bitstream.
3.  **Destruction:** When the `MutProxy` goes out of scope, its `Drop` implementation is executed. This is the critical step where the potentially modified value from the temporary copy is taken, re-encoded, and written back into the correct bit position in the parent `FixedVec`'s storage.

This is a classic copy-on-read, write-on-drop mechanism. It provides a safe and ergonomic abstraction for mutating non-addressable data, preserving the feel of direct manipulation while correctly handling the bit-level operations under the hood. The overhead is a single read at the start of the proxy's life and a single write at the end.

```rust
use compressed_intvec::fixed::{FixedVec, UFixedVec, BitWidth};

let data: &[u32] = &[10, 20, 30];
let mut vec: UFixedVec<u32> = FixedVec::builder()
    .bit_width(BitWidth::Explicit(7))
    .build(data)
    .unwrap();

// vec.at_mut(1) returns an Option<MutProxy<...>>
if let Some(mut proxy) = vec.at_mut(1) {
    // The DerefMut trait allows us to modify the proxy's internal copy.
    *proxy = 99;
} // The proxy is dropped here. Its Drop impl writes 99 back to the vec.

assert_eq!(vec.get(1), Some(99));
```

The overhead of this approach is a single read on the proxy's construction and a single write on its destruction, which is an acceptable trade-off for an ergonomic and safe mutable API.

### Zero-Copy Views with `FixedVecSlice`

Beyond single-element mutation, a `Vec`-like API needs to support slicing. Creating a `FixedVec` that borrows its data (`B = &[W]`) is the foundation for this, but we also need a dedicated slice type to represent a sub-region of another `FixedVec` without copying data. For this we can create `FixedVecSlice`.

The implementation is a classic "fat pointer" struct. It holds a reference to the parent `FixedVec` and a `Range<usize>` that defines the logical boundaries of the view.

```rust
// A zero-copy view into a contiguous portion of a FixedVec.
#[derive(Debug)]
pub struct FixedVecSlice<V> {
    pub(super) parent: V,
    pub(super) range: Range<usize>,
}
```

The generic parameter `V` is a reference to the parent vector. This allows the same `FixedVecSlice` struct to represent both immutable (`V = &FixedVec<...>`) and mutable (`V = &mut FixedVec<...>`) views.

We can implement all the operations on the slice by translating the slice-relative index into an absolute index in the parent vector. For example, we can easily implement `get_unchecked` with this delegation:

```rust
// Index translation within the slice's get_unchecked
pub unsafe fn get_unchecked(&self, index: usize) -> T {
    debug_assert!(index < self.len());
    // The index is relative to the slice, so we add the slice's start
    // offset to get the correct index in the parent vector.
    self.parent.get_unchecked(self.range.start + index)
}
```

This design ensures that there is no code duplication; the slice re-uses the access logic of the parent `FixedVec`.

#### Slicing and Mutability

For mutable slices (`V = &mut FixedVec<...>` ), we can provide mutable access to the slice's elements. The `at_mut` method on `FixedVecSlice` follows the same principle of index translation:

```rust
pub fn at_mut(&mut self, index: usize) -> Option<MutProxy<'_, T, W, E, B>> {
    if index >= self.len() {
        return None;
    }
    // The index is translated to the parent vector's coordinate space.
    Some(MutProxy::new(&mut self.parent, self.range.start + index))
}
```

A mutable slice borrows the parent `FixedVec` mutably. This means that while the slice exists, the parent vector cannot be accessed directly, upholding Rust's borrowing rules. A critical implementation detail is the `split_at_mut` method, which must produce two non-overlapping mutable slices from a single `&mut self`. This requires careful use of unsafe code to create two `&mut` references from a single one, which is safe only because we can prove to the compiler that the logical ranges they represent (`0..mid` and `mid..len`) are disjoint.

```rust
pub fn split_at_mut(&mut self, mid: usize) -> (FixedVecSlice<&mut Self>, FixedVecSlice<&mut Self>) {
    assert!(mid <= self.len, "mid > len in split_at_mut");
    // SAFETY: The two slices are guaranteed not to overlap.
    unsafe {
        let ptr = self as *mut Self;
        let left = FixedVecSlice::new(&mut *ptr, 0..mid);
        let right = FixedVecSlice::new(&mut *ptr, mid..self.len());
        (left, right)
    }
}
```

This combination of a generic slice struct and careful pointer manipulation allows us to build a rich, safe, and zero-copy API for both immutable and mutable views, mirroring the flexibility of Rust's native slice

## Thread-Safe Concurrent Access

The next step is to extend the `FixedVec` model to support concurrency. Our goal is to create a thread-safe variant, `AtomicFixedVec`, with an API that mirrors the behavior and guarantees of Rust's standard atomic types (`std::sync::atomic::AtomicU64`, etc.). This means providing methods like `load`, `store`, `swap`, and `fetch_add` that can be safely called from multiple threads.

This immediately presents a fundamental problem. Hardware-level atomic instructions operate on machine words: aligned, native-sized integers (`u8`, `u16`, `u32`, `u64`). Our data, however, is not structured this way. A logical element in `FixedVec` is a virtual entity, a sequence of bits that is not necessarily byte-aligned and, more critically, can span the boundary between two separate `u64` words in our backing storage.

How can we guarantee atomicity for an operation on a 10-bit integer that requires modifying the last 4 bits of one `u64` and the first 6 bits of the next? No single hardware instruction can perform such an operation atomically across two distinct memory locations. A naive implementation would lead to race conditions where one thread could observe a partially-written, corrupted value.

### Structure

Extending `FixedVec` to support concurrency requires a fundamental shift in its internal design. The first step is to change the backing storage. A `Vec<u64>` is not thread-safe for concurrent writes. We must replace it with a `Vec<AtomicU64>`. This ensures that every individual read and write to a 64-bit word is, at a minimum, atomic.

However, as we've established, a logical element can span two of these atomic words. A simple `Vec<AtomicU64>` does not solve the problem of atomically updating a value that crosses a word boundary. To handle this, we introduce a second component to our struct: a striped lock pool.

We can then define the `AtomicFixedVec` struct as follows:

```rust
pub struct AtomicFixedVec<T>
where
    T: Storable<u64>,
{
    // The backing store is now composed of atomic words.
    pub(crate) storage: Vec<AtomicU64>,
    // A pool of fine-grained locks to protect spanning-word operations.
    locks: Vec<Mutex<()>>,
    bit_width: usize,
    mask: u64,
    len: usize,
    _phantom: PhantomData<T>,
}
```

The `storage` field provides the atomic access at the word level. The `locks` field is an array of `parking_lot::Mutex` instances. This lock pool is the mechanism we will use to enforce atomicity for operations that must modify two adjacent words simultaneously.

With this structure in place, we can now design a hybrid concurrency model. Every operation must first determine if the target element is fully contained within a single `AtomicU64` or if it spans two. Based on this, it will dispatch to one of two paths: a high-performance, lock-free path for in-word elements, or a lock-based path that uses our striped lock pool for spanning elements.

### Lock-Free path for In-Word Elements

The high-throughput path in our hybrid model is for elements that are fully contained within a single `AtomicU64`. This condition, `bit_offset + bit_width <= 64`, is always met if the `bit_width` is a power of two (e.g., 2, 4, 8, 16, 32), making `BitWidth::PowerOfTwo` an explicit performance choice for write-heavy concurrent workloads. When this condition holds, we can perform a true lock-free atomic update using a **Compare-and-Swap (CAS) loop**.

A CAS operation is an atomic instruction provided by the hardware (e.g., `CMPXCHG` on x86-64) that attempts to write a new value to a memory location only if the current value at that location matches a given expected value. This provides the primitive for building more complex atomic operations without locks.

Let's try to implement an atomic `store` on a sub-word element. We cannot simply write the new value, as that would non-atomically overwrite adjacent elements in the same word. The operation must be a read-modify-write on the entire `AtomicU64`, where the "modify" step only alters the bits corresponding to our target element.

```rust
fn atomic_store(&self, index: usize, value: u64, order: Ordering) {
    let bit_pos = index * self.bit_width;
    let word_index = bit_pos / u64::BITS as usize;
    let bit_offset = bit_pos % u64::BITS as usize;

    // Lock-free path for single-word values.
    let atomic_word_ref = &self.storage[word_index];
    let store_mask = self.mask << bit_offset;
    let store_value = value << bit_offset;
    let mut old_word = atomic_word_ref.load(Ordering::Relaxed);
    loop {
        let new_word = (old_word & !store_mask) | store_value;
        match atomic_word_ref.compare_exchange_weak(
            old_word,
            new_word,
            order,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(x) => old_word = x,
        }
    }
```

We begin by calculating the `word_index` and `bit_offset`. We then prepare a `store_mask` to isolate the target bit range and a `store_value` with the new value already shifted into position. The core of the operation is the loop. We first perform a relaxed load to get the current state of the entire 64-bit word. Inside the loop, we compute `new_word` by using the mask to clear the target bits in our local copy (`old_word`) and then OR in the new value.

The critical instruction is `compare_exchange_weak`. It attempts to atomically replace the value at `atomic_word_ref` with `new_word`, but only if its current value is still equal to `old_word`. If another thread has modified the word in the meantime, the operation fails, returns the now-current value, and our loop continues with the updated `old_word`. We use the `weak` variant because it can be more performant on some architectures and is perfectly suitable within a retry loop.

This entire path is lock-free. Contention is managed at the hardware level by the CPU's cache coherency protocol, which ensures that only one core can successfully commit a CAS operation to a given cache line at a time. 

In the same way, we can implement other atomic operations like `atomic_rmw` (for `fetch_add`, `fetch_sub`, etc.) and `atomic_load` using similar CAS loops or direct atomic loads.

### Lock-Based Path for Spanning Elements

When an element crosses a word boundary (`bit_offset + bit_width > 64`), the lock-free CAS loop is no longer a viable strategy. An atomic update now requires modifying two separate `AtomicU64` words, and no single hardware instruction can do this as one indivisible transaction. To ensure atomicity, we must use a lock.

A single, global `Mutex` would serialize all concurrent writes, becoming an unacceptable performance bottleneck. We instead employ **lock striping**, a technique that partitions the lock coverage to reduce contention. `AtomicFixedVec` maintains a `Vec<parking_lot::Mutex<()>>` where the number of locks is a power of two, determined at construction time by a simple heuristic:

```rust
// Heuristic to determine the number of locks for striping.
let num_cores = std::thread::available_parallelism().map_or(MIN_LOCKS, |n| n.get());
let target_locks = (num_words / WORDS_PER_LOCK).max(1);
let num_locks = (target_locks.max(num_cores) * 2)
    .next_power_of_two()
    .min(MAX_LOCKS);
```

The logic aims to create enough locks to service the available hardware threads (`num_cores`) and to cover the data with a reasonable density (one lock per 64 words, `WORDS_PER_LOCK`). We take the maximum of these two, multiply by two as a simple overprovisioning factor, and round up to the next power of two to enable fast mapping via bitwise AND. The total number of locks is capped to prevent excessive memory consumption for the lock vector itself.

To perform an operation on a spanning element that touches `word_index` and `word_index + 1`, a thread acquires a lock by mapping the word index to a lock index. We use a fast bitwise AND for this mapping instead of a slower modulo operation: `let lock_index = word_index & (self.locks.len() - 1)`.

We can then partition the vector into a set of independent regions. A locked write to words `(i, i+1)` will not block a simultaneous locked write to words `(j, j+1)` or a lock-free write to word `k`, provided they map to different locks in the stripe. This maintains a high degree of parallelism. While one could theoretically design a more complex scheme where a lock protects a specific *pair* of words, implementing such a system with dynamic lock allocation and management in safe Rust would be an exercise in extreme complexity, likely involving raw pointers and manual memory management, for questionable performance gains. The striped array is a pragmatic and robust solution.

We can update the `atomic_store` function to include this lock-based path:

```rust
fn atomic_store(&self, index: usize, value: u64, order: Ordering) {
    let bit_pos = index * self.bit_width;
    let word_index = bit_pos / u64::BITS as usize;
    let bit_offset = bit_pos % u64::BITS as usize;

    if bit_offset + self.bit_width <= u64::BITS as usize {
        // Lock-free path for single-word values.
        let atomic_word_ref = &self.storage[word_index];
        let store_mask = self.mask << bit_offset;
        let store_value = value << bit_offset;
        let mut old_word = atomic_word_ref.load(Ordering::Relaxed);
        loop {
            let new_word = (old_word & !store_mask) | store_value;
            match atomic_word_ref.compare_exchange_weak(
                old_word,
                new_word,
                order,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => old_word = x,
            }
        }
    } else {
        // Locked path for values spanning two words.
        let lock_index = word_index & (self.locks.len() - 1);
        let _guard = self.locks[lock_index].lock();
        // The lock guarantees exclusive access to this multi-word operation.
        // We still use atomic operations inside to prevent races with the
        // lock-free path, which might be concurrently accessing one of 
        // these words.
        let low_word_ref = &self.storage[word_index];
        let high_word_ref = &self.storage[word_index + 1];

        // Modify the lower word.
        low_word_ref
            .fetch_update(order, Ordering::Relaxed, |mut w| {
                w &= !(u64::MAX << bit_offset);
                w |= value << bit_offset;
                Some(w)
            })
            .unwrap(); // Should not fail under lock.

        // Modify the higher word.
        let bits_in_high = (bit_offset + self.bit_width) - u64::BITS as usize;
        let high_mask = (1u64 << bits_in_high).wrapping_sub(1);
        high_word_ref
            .fetch_update(order, Ordering::Relaxed, |mut w| {
                w &= !high_mask;
                w |= value >> (u64::BITS as usize - bit_offset);
                Some(w)
            })
            .unwrap(); // Should not fail under lock.
    }
}
```

Once the lock is acquired, we have exclusive access for this two-word transaction. However, it's critical that we continue to use atomic operations like `fetch_update` to modify the words themselves. This is because another thread might be concurrently executing a *lock-free* operation on an adjacent, non-spanning element that happens to reside in `word_index` or `word_index + 1`. Using non-atomic writes inside the lock would create a data race with these lock-free operations.

Inside the critical section, we perform two separate atomic updates. For the lower word, we use `fetch_update` to clear the high bits and write the low bits of our value. We do the same for the higher word, calculating the `high_mask` to clear the low bits and writing the high bits of our value. The `unwrap()` calls are safe because, with the lock held, there can be no contention, so the update closure will never need to be retried and will never return `None`.

The lock provides the mutual exclusion that makes the two-word update appear atomic as a whole, while the continued use of atomics and the specified `Ordering` ensures that the final result is correctly propagated through the memory system and becomes visible to other threads according to the Rust memory model.