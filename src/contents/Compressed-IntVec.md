---
author: Luca Lombardo
datetime: 2025-08-24
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

---

In this post, we will explore the engineering challenges involved in implementing an efficient vector-like data structure in Rust that stores integers in a compressed, bit-packed format. We will focus on achieving O(1) random access performance while minimizing memory usage. We will try to mimic the ergonomics of Rust's standard `Vec<T>` as closely as possible, including support for mutable access and zero-copy slicing. Finally, we will extend the design to support thread-safe concurrent access with atomic operations.

* All the code can be found on github: [compressed-intvec](https://github.com/lukefleed/compressed-intvec)
* This is also published as a crate on crates.io: [compressed-intvec](https://crates.io/crates/compressed-intvec)

---

In Rust, `Vec<T>`, where `T` is a primitive integer type like `u64` or `i32`, is the standard for contiguous, heap-allocated arrays. However, its memory layout is fundamentally tied to the static size of `T`, leading to significant waste when the dynamic range of the stored values is smaller than the type's capacity.

This inefficiency is systemic. Consider storing the value `5` within a `Vec<u64>`. Its 8-byte in-memory representation is:

<div align="center">

`00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000101`

</div>

Only 3 bits are necessary to represent the value, leaving 61 bits as zero-padding. The same principle applies, albeit less dramatically, when storing `5` in a `u32` or `u16`. At scale, this overhead becomes prohibitive. A vector of one billion `u64` elements consumes `10^9 * std::mem::size_of::<u64>()`, or approximately 8 GB of memory, even if every element could fit within a single byte.

The canonical solution is bit packing, which aligns data end-to-end in a contiguous bitvector. However, this optimization has historically come at the cost of random access performance. The O(1) access guarantee of `Vec<T>` is predicated on simple pointer arithmetic: `address = base_address + index * std::mem::size_of::<T>()`. Tightly packing the bits invalidates this direct address calculation, seemingly forcing a trade-off between memory footprint and access latency.

This raises the central question that with this post we aim to answer: is it possible to design a data structure that decouples its memory layout from the static size of `T`, adapting instead to the data's true dynamic range, without sacrificing the O(1) random access that makes `Vec<T>` so effective?

## Fixed-width bit packing with arithmetic indexing

If the dynamic range of our data is known, we can define a fixed `bit_width` for every integer. For instance, if the maximum value in a dataset is `1000`, we know every number can be represented in 10 bits, since `2^10 = 1024`. Instead of allocating 64 bits per element, we can store them back-to-back in a contiguous bitvector, forming the core of what from now on we'll refer as `FixedVec`. We can immagine such data strucutre as a logical array of `N` integers, each `bit_width` bits wide, stored in a backing buffer of `u64` words.

```rust
struct FixedVec {
    limbs: Vec<u64>, // Backing storage
    bit_width: usize, // Number of bits per element
    len: usize, // Number of elements
    mask: u64, // Precomputed mask for extraction
}
```

Where the role of the `mask` field is to isolate the relevant bits during extraction. For a `bit_width` of 10, the mask would be `0b1111111111`.

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

The single-word access logic is fast, but it only works as long as `bit_offset + bit_width <= 64`. This assumption breaks down as soon as an integer's bit representation needs to cross the boundary from one `u64` word into the next. This is guaranteed to happen for any `bit_width` that is not a power of two. For example, with a 10-bit width, the element at `index = 6` starts at bit position 60. Its 10 bits will occupy bits 60-63 of the first word and bits 0-5 of the second. The simple right-shift-and-mask trick fails here.

To correctly decode the value, we must read *two* consecutive `u64` words and combine their bits. This splits our `get_unchecked` implementation into two paths. The first is the fast path we've already seen. The second is a new path for spanning values

To get the lower bits of the value, we read the first word and shift right, just as before. This leaves the upper bits of the word as garbage.

```rust
let low_part = *limbs.get_unchecked(word_index) >> bit_offset;
```

To get the upper bits of the value, we read the *next* word. The bits we need are at the beginning of this word, so we shift them left to align them correctly.

```rust
let high_part = *limbs.get_unchecked(word_index + 1) << (64 - bit_offset);
```

Finally, we combine the two parts with a bitwise OR and apply the mask to discard any remaining high-order bits from the `high_part`.

```rust
(low_part | high_part) & self.mask
```

The line `limbs.get_unchecked(word_index + 1)` introduces a safety concern: if we are reading the last element of the vector, `word_index + 1` could point past the end of our buffer, leading to undefined behavior. To prevent this, our builder must always allocate one extra padding word at the end of the storage. This small memory cost guarantees that any read, even one that spans the final word boundary, will always access valid, allocated memory.

Integrating these two paths gives us our final `get_unchecked` implementation:

```rust
pub unsafe fn get_unchecked(&self, index: usize) -> u64 {
    let bit_width = self.bit_width;
    let bit_pos = index * bit_width;
    let word_index = bit_pos / 64;
    let bit_offset = bit_pos % 64;

    let limbs = self.bits.as_ref();

    if bit_offset + bit_width <= 64 {
        // Fast path: value is fully within one word
        (*limbs.get_unchecked(word_index) >> bit_offset) & self.mask
    } else {
        // Slow path: value spans two words.
        let low_part = *limbs.get_unchecked(word_index) >> bit_offset;
        let high_part = *limbs.get_unchecked(word_index + 1) << (64 - bit_offset);
        (low_part | high_part) & self.mask
    }
}
```

### Faster Reads: Unaligned Access

Our `get_unchecked` implementation is correct, but the slow path for spanning values requires two separate, aligned memory reads. The instruction sequence for this method involves at least two load instructions, multiple shifts, and a bitwise OR. These instructions have data dependencies: the shifts cannot execute until the loads complete, and the OR cannot execute until both shifts are done. This dependency chain can limit the CPU's instruction-level parallelism and create pipeline stalls if the memory accesses miss the L1 cache.

Let's have a look at the machine code generated by this method. We can create a minimal binary with an `#[inline(never)]` function that calls `get_unchecked` on a known spanning index, then use `cargo asm` to inspect the disassembly.

```asm
; asm_test::aligned::get_aligned (spanning path)
.LBB14_2:
        ; let low = *limbs.get_unchecked(word_index) >> bit_offset;
        mov rsi, qword ptr [r8 + 8*rdx]      ; <<< LOAD #1 (low_part)

        ; let high = *limbs.get_unchecked(word_index + 1) << (64 - bit_offset);
        mov rdx, qword ptr [r8 + 8*rdx + 8]  ; <<< LOAD #2 (high_part)

        shr rsi, cl                          ; shift right of low_part
        shl rdx, cl                          ; shift left of high_part
        or rdx, rsi                          ; combine results
```

The instruction `mov rsi, qword ptr [r8 + 8*rdx]` is our first memory access. It loads a 64-bit value (`qword`) into the `rsi` register. The address is calculated using base-plus-index addressing: `r8` holds the base address of our `limbs` buffer, and `rdx` holds the `word_index`. This corresponds directly to `limbs[word_index]`.

Immediately following is `mov rdx, qword ptr [r8 + 8*rdx + 8]`. This is our second, distinct memory access. It loads the *next* 64-bit word from memory by adding an 8-byte offset to the previous address. This corresponds to `limbs[word_index + 1]`.

Only after both of these `mov` instructions complete can the CPU proceed. The `shr rsi, cl` instruction (shift right `rsi` by the count in `cl`) cannot execute until the first `mov` has placed a value in `rsi`. Similarly, `shl rdx, cl` depends on the second `mov`. Finally, `or rdx, rsi` depends on both shifts. 

This sequence of operations—loading two adjacent 64-bit words, shifting each, and combining them is a software implementation of what is, conceptually, a [128-bit barrel shifter](https://en.wikipedia.org/wiki/Barrel_shifter). We are selecting a 64-bit window from a virtual 128-bit integer formed by concatenating the two words from memory. There are clear data dependencies: the `shr` instruction depends on the first `mov` completing, the `shl` on the second `mov`, and the final `or` on both shifts. This creates a dependency chain that stalls the CPU's execution pipeline if either memory access is slow (e.g., a cache miss). This is the primary source of inefficiency in this path. We are forced to serialize two memory accesses and their subsequent arithmetic.

**Can we do better then this?** Potentially, yes. We can replace this multi-instruction sequence with something more direct by delegating the complexity to the hardware and performing a single unaligned memory read. Modern x86-64 CPUs handle this directly: when an unaligned load instruction is issued, the CPU's memory controller fetches the necessary cache lines and the load/store unit reassembles the bytes into the target register. This entire process is a single, optimized micro-operation. 

We can then implement a more aggressive access method: `get_unaligned_unchecked`. The strategy is to calculate the exact *byte* address where our data begins and perform a single, unaligned read of a full `W` word from that position.

The implementation first translates the absolute bit position into a byte address and a residual bit offset within that byte.

```rust
let bit_pos = index * self.bit_width;
let byte_pos = bit_pos / 8;
let bit_rem = bit_pos % 8;
```

With the byte-level address, we get a raw `*const u8` pointer to our storage and perform the unaligned read. The [read_unaligned](https://doc.rust-lang.org/std/ptr/fn.read_unaligned.html) intrinsic in Rust compiles down to a single machine instruction that the hardware can execute efficiently.

```rust
let limbs_ptr = self.as_limbs().as_ptr() as *const u8;
// This read may cross hardware word boundaries, but the CPU handles it.
let word: W = (limbs_ptr.add(byte_pos) as *const W).read_unaligned();
```

On a Little-Endian system, the loaded `word` now contains our target integer, but it's shifted by `bit_rem` positions. A simple right shift aligns our value to the LSB, and applying the mask isolates it.

```rust
let extracted_word = word >> bit_rem;
let final_value = extracted_word & self.mask;
```

This operation is safe only because we are supposing that our builder guarantees a padding word at the end of the storage buffer. This ensures that even an unaligned read starting in the last few bytes of the logical data area will not access unallocated memory. Combining all these steps, we get our final `get_unaligned_unchecked` implementation:

```rust
pub unsafe fn get_unaligned_unchecked(&self, index: usize) -> u64 {
    let bit_pos = index * self.bit_width;
    let byte_pos = bit_pos / 8;
    let bit_rem = bit_pos % 8;

    let limbs_ptr = self.as_limbs().as_ptr() as *const u8;
    // SAFETY: The builder guarantees an extra padding word at the end.
    let word = (limbs_ptr.add(byte_pos) as *const W).read_unaligned();
    let extracted_word = word >> bit_rem;
    extracted_word & self.mask
}
```

As before, we can inspect the generated assembly for this method when accessing a spanning index:

```asm
; asm_test::unaligned::get_unaligned
.LBB15_1:
        ; let bit_pos = index * self.bit_width;
        imul rcx, rax
        ; let byte_pos = bit_pos / 8;
        mov rax, rcx
        shr rax, 3
        ; self.bits.as_ref()
        mov rdx, qword ptr [rdi + 8]
        ; unsafe { crate::intrinsics::copy_nonoverlapping(src, dst, count) }
        mov rax, qword ptr [rdx + rax]       ; <<< SINGLE UNALIGNED LOAD
        ; self >> other
        and cl, 7
        shr rax, cl
        ; fn bitand(self, rhs: $t) -> $t { self & rhs }
        and rax, qword ptr [rdi + 32]
```

 The initial `imul` and `shr rax, 3` (a fast division by 8) correspond to the calculation of `byte_pos`. The instruction `mov rdx, qword ptr [rdi + 8]` loads the base address of our `limbs` buffer into the `rdx` register.

The instruction `mov rax, qword ptr [rdx + rax]` is our single unaligned load. The address `[rdx + rax]` is the sum of the buffer's base address and our calculated `byte_pos`. This `mov` instruction reads 8 bytes (a `qword`) directly from this potentially unaligned memory location into the `rax` register. We can see that as we hoped, the [read_unaligned](https://doc.rust-lang.org/std/ptr/fn.read_unaligned.html) intrinsic has been compiled down to a single hardware instruction.

The next instructions handle the extraction. The `and cl, 7` and `shr rax, cl` sequence corresponds to our `>> bit_rem`. `cl` holds the lower bits of the original `bit_pos` (our `bit_rem`), and the shift aligns our desired value to the LSB of the `rax` register. Finally, `and rax, qword ptr [rdi + 32]` applies the pre-calculated mask, which is stored at an offset from the `self` pointer in `rdi`.

### Random Access Performance

We can now benchmark the latency of 1 million random access operations on a vector containing 10 million elements. For each `bit_width`, we generate data with a uniform random distribution in the range `[0, 2^bit_width)`. The code for the benchmark is available here: [`bench-intvec`]

Our baseline is the smallest standard `Vec<T>` capable of holding the data (`Vec<u8>` for `bit_width <= 8`, etc.). This provides a direct comparison against the optimal standard library implementation. We also include results from [`sux::BitFieldVec`], [`succinct::IntVector`], and [`simple-sds-sbwt::IntVector`] for context.

<iframe src="/bench-intvec/fixed_random_access_performance.html" width="100%" height="550px" style="border: none;"></iframe>

We can see that for `bit_width` values below 32, the `get_unaligned_unchecked` of our `FixedVec` is almost always faster than the corresponding `Vec<T>` baseline. This is a result of improved cache locality. A 64-byte L1 cache line can hold 64 elements from a `Vec<u8>`. With a `bit_width` of 4, the same cache line holds `(64 * 8) / 4 = 128` elements from our `FixedVec`. This increased data density improves the cache hit rate for random access patterns, and the latency reduction from avoiding DRAM access outweighs the instruction cost of the bitwise extraction. For values of `bit_width` above 32, the performance of `FixedVec` are very slightly worse than the `Vec<T>` baseline, as the cache locality advantage diminishes. However, the memory savings remain.

The performance delta between `get_unaligned_unchecked` and `get_unchecked` confirms the unaligned access strategy discussed before: a single `read_unaligned` instruction is more efficient than the two dependent aligned reads required by the logic for spanning words.

We can see that the only other crate that comes close to our performance is [`sux::BitFieldVec`], by Sebastiano Vigna. The other two crates, [`succinct::IntVector`] and [`simple-sds-sbwt::IntVector`], are significantly slower (note that the Y-axis is logarithmic.

[`sux::BitFieldVec`]: https://docs.rs/sux/latest/sux/bits/bit_field_vec/index.html
[`sux-rs`]: https://crates.io/crates/sux
[`succinct::IntVector`]: https://docs.rs/succinct/latest/succinct/trait.IntVec.html
[`simple-sds-sbwt::IntVector`]: https://docs.rs/simple-sds-sbwt/latest/simple_sds_sbwt/int_vector/struct.IntVector.html
[`bench-intvec`]: https://github.com/lukefleed/compressed-intvec/blob/master/benches/fixed/bench_random_access.rs

## The Architecture

With the access patterns defined, we need to think about the overall architecture of this data structure. A solution hardcoded to `u64` would lack the flexibility to adapt to different use cases. We need a structure that is generic over the its principal components: the logical type, the physical storage type, the bit-level ordering, and ownership. We can define a struct that is generic over these four parameters:

```rust
pub struct FixedVec<T: Storable<W>, W: Word, E: Endianness, B: AsRef<[W]> = Vec<W>> {
    bits: B,
    bit_width: usize,
    mask: W,
    len: usize,
}
```

Where:

`T` is the **logical element type**, the type as seen by the user of the API (e.g., `i16`, `u32`). By abstracting `T`, we divide the user-facing type from the internal storage representation.

`W` is the **physical storage word**, which must implement our `Word` trait. It defines the primitive unsigned integer (`u32`, `u64`, `usize`) of the backing buffer and sets the granularity for all bitwise operations. 

`E` requires the `dsi-bitstream::Endianness` trait, allowing us to specify either Little-Endian (`LE`) or Big-Endian (`BE`) byte order. 

`B`, which must implement `AsRef<[W]>`, represents the **backing storage**. This abstraction over ownership allows `FixedVec` to be either an owned container where `B = Vec<W>`, or a zero-copy, borrowed view where `B = &[W]`. This makes it possible to, for example, construct a `FixedVec` directly over a memory-mapped slice without any heap allocation.

In this way, the compiler monomorphizes the struct and its methods for each concrete instantiation (e.g., `FixedVec<i16, u64, LE, Vec<u64>>`), resulting in specialized code with no runtime overhead from the generic abstractions.

### The `Word` Trait

The first step is to abstract the physical storage layer. The `W` parameter must be a primitive unsigned integer type that supports bitwise operations. We can define a `Word` trait that captures these requirements:

```rust
pub trait Word:
    UnsignedInt + Bounded + ToPrimitive + dsi_bitstream::traits::Word
    + NumCast + Copy + Send + Sync + Debug + IntoAtomic + 'static
{
    const BITS: usize = std::mem::size_of::<Self>() * 8;
}
```

The numeric traits (`UnsignedInt`, `Bounded`, `NumCast`, `ToPrimitive`) are necessary for the arithmetic of offset and mask calculations. The `dsi_bitstream::traits::Word` bound allows us to integrate with its `BitReader` and `BitWriter` implementations, offloading the bitstream logic. `Send` and `Sync` are non-negotiable requirements for any data structure that might be used in a concurrent context. The `IntoAtomic` bound is particularly forward-looking: it establishes a compile-time link between a storage word like `u64` and its atomic counterpart, `AtomicU64`. We will use it later to build a thread safe, atomic version of `FixedVec`. Finally, the `const BITS` associated constant lets us write architecture-agnostic code that correctly adapts to `u32`, `u64`, or `usize` words without `cfg` flags.

### The `Storable` Trait

With the physical storage layer defined, we need a formal contract to connect it to the user's logical type `T`. We can do this by creating the `Storable` trait, which defines a bidirectional, lossless conversion.

```rust
pub trait Storable<W: Word>: Sized + Copy {
    fn into_word(self) -> W;
    fn from_word(word: W) -> Self;
}
```

For unsigned types, the implementation is a direct cast. For signed types, however, we must map the `iN` domain to the `uN` domain required for bit-packing. A simple two's complement bitcast is unsuitable, as `i64(-1)` would become `u64::MAX`, a value requiring the maximum number of bits. We need a better mapping that preserves small absolute values.

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

With this logic within the trait system, the main `FixedVec` implementation remains clean and agnostic to the signedness of the data it stores. 

## The Builder Pattern

Once the structure logic is in place, we have to design an ergonomic way to construct it. A simple `new()` function isn't sufficient because the vector's memory layout depends on parameters that must be determined *before* allocation, most critically the `bit_width`. This is a classic scenario for a builder pattern.


The central problem is that the optimal `bit_width` often depends on the data itself. We need a mechanism to specify the *strategy* for determining this width. We can create the `BitWidth` enum:

```rust
pub enum BitWidth {
    Minimal,
    PowerOfTwo,
    Explicit(usize),
}
```



With this enum, the user can choose between three strategies:

- `Minimal`: The builder scans the input data to find the maximum value, then calculates the minimum number of bits required to represent it. This is the most space-efficient option but requires a full pass over the data.
- `PowerOfTwo`: Similar to `Minimal`, but rounds the bit width up to the next power of two. This can simplify certain bitwise operations and align better with hardware word sizes, at the cost of some additional space.
- `Explicit(n)`: The user provides a fixed bit width. This avoids the data scan but requires the user to ensure that all values fit within the specified width.

> **Note:** Yes, I could have also made three different build functions: `new_with_minimal_bit_width()`, `new_with_power_of_two_bit_width()`, and `new_with_explicit_bit_width(n)`. However, this would lead to a combinatorial explosion if we later wanted to add more configuration options. The builder pattern scales better.

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

> `UFixedVec<T>` is a type alias for `FixedVec<T, u64, LE, Vec<u64>>`, the most common instantiation.


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

### Zero-Copy Views

A `Vec`-like API needs to support slicing. Creating a `FixedVec` that borrows its data (`B = &[W]`) is the first step for this, but we also need a dedicated slice type to represent a sub-region of another `FixedVec` without copying data. For this we can create `FixedVecSlice`.

We can implement this as a classic "fat pointer" struct. It holds a reference to the parent `FixedVec` and a `Range<usize>` that defines the logical boundaries of the view.

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

In this way there is no code duplication; the slice re-uses the access logic of the parent `FixedVec`.

### Slicing and Mutability

For mutable slices (`V = &mut FixedVec<...>` ), we can provide mutable access to the slice's elements. We can easily implement the `at_mut` method on `FixedVecSlice` using the same principle of index translation:

```rust
pub fn at_mut(&mut self, index: usize) -> Option<MutProxy<'_, T, W, E, B>> {
    if index >= self.len() {
        return None;
    }
    // The index is translated to the parent vector's coordinate space.
    Some(MutProxy::new(&mut self.parent, self.range.start + index))
}
```

A mutable slice borrows the parent `FixedVec` mutably. This means that while the slice exists, the parent vector cannot be accessed directly due to Rust's borrowing rules. Let's consider the following situation: we may need to split a mutable slice into two non-overlapping mutable slices. This is common for example in algorithms that operate on sub-regions of an array. However, implementing such a method requires to use some unsafe code. The method, let's say `split_at_mut`, must produce two `&mut` references from a single one. In order to be safe, we must prove to the compiler that the logical ranges they represent (`0..mid` and `mid..len`) are disjoint.

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

This combination of a generic slice struct and careful pointer manipulation allows us to build a rich, safe, and zero-copy API for both immutable and mutable views, mirroring Rust's native slice

# Thread-Safe Concurrent Access

The next step is to extend the `FixedVec` model to support concurrency. Our goal is to create a thread-safe variant, `AtomicFixedVec`, with an API that mirrors the behavior and guarantees of Rust's standard atomic types (`std::sync::atomic::AtomicU64`, etc.). This means providing methods like `load`, `store`, `swap`, and `fetch_add` that can be safely called from multiple threads.

However, there is a fundamental problem. Hardware-level atomic instructions operate on machine words: aligned, native-sized integers (`u8`, `u16`, `u32`, `u64`). Our data, however, is not structured this way. A logical element in `FixedVec` is a virtual entity, a sequence of bits that is not necessarily byte-aligned and, more critically, can span the boundary between two separate `u64` words in our backing storage.

How can we guarantee atomicity for an operation on a 10-bit integer that requires modifying the last 4 bits of one `u64` and the first 6 bits of the next? No single hardware instruction can perform such an operation atomically across two distinct memory locations. A naive implementation would lead to race conditions where one thread could observe a partially-written, corrupted value.

### Structure

To make `FixedVec` support concurrency, we have to make some changes in its internal design. The first step is to change the backing storage. A `Vec<T>` is not thread-safe for concurrent writes. We can replace it with a `Vec<AtomicU64>`. This ensures that every individual read and write to a 64-bit word is, at a minimum, atomic.

However, as we've established, a logical element can span two of these atomic words. A simple `Vec<AtomicU64>` does not solve the problem of atomically updating a value that crosses a word boundary. To handle this, we can add a second component to our struct: a striped lock pool. This is a vector of `parking_lot::Mutex` instances, where the number of locks is a power of two. Each lock protects a subset of the `AtomicU64` words in our storage. When an operation needs to modify two adjacent words, it will acquire the appropriate locks to ensure exclusive access. 

We can then define the `AtomicFixedVec` struct as follows:

```rust
pub struct AtomicFixedVec<T>
where
    T: Storable<u64>,
{
    // The backing store is now composed of atomic words.
    storage: Vec<AtomicU64>,
    // A pool of fine-grained locks to protect spanning-word operations.
    locks: Vec<Mutex<()>>,
    bit_width: usize,
    mask: u64,
    len: usize,
}
```

 This lock pool is the mechanism we will use to enforce atomicity for operations that must modify two adjacent words simultaneously.

With this structure in place, we can now design a hybrid concurrency model. Every operation must first determine if the target element is fully contained within a single `AtomicU64` or if it spans two. Based on this, it will dispatch to one of two paths: a high-performance, lock-free path for in-word elements, or a lock-based path that uses our striped lock pool for spanning elements.

### Lock-Free path for In-Word Elements

The high-throughput path in our hybrid model is for elements that are fully contained within a single `AtomicU64`. This condition, `bit_offset + bit_width <= 64`, is always met if the `bit_width` is a power of two (e.g., 2, 4, 8, 16, 32), making `BitWidth::PowerOfTwo` an explicit performance choice for write-heavy concurrent workloads. When this condition holds, we can perform a true lock-free atomic update using a **Compare-and-Swap (CAS) loop**.

A CAS operation is an atomic instruction provided by the hardware (e.g., `CMPXCHG` on x86-64) that attempts to write a new value to a memory location only if the current value at that location matches a given expected value. This provides the primitive for building more complex atomic operations without locks.

Let's try to implement an atomic `store` on a sub-word element. We cannot simply write the new value, as that would non-atomically overwrite adjacent elements in the same word. The operation must be a read-modify-write on the entire `AtomicU64`, where the "modify" step only alters the bits corresponding to our target element.

We begin by calculating the `word_index` and `bit_offset`. 

```rust
let bit_pos = index * self.bit_width;
let word_index = bit_pos / u64::BITS as usize;
let bit_offset = bit_pos % u64::BITS as usize;
```

We then prepare a `store_mask` to isolate the target bit range and a `store_value` with the new value already shifted into position. 

```rust
let store_mask = self.mask << bit_offset;
let store_value = value << bit_offset;
```

The core of the operation is the loop. We first perform a relaxed load to get the current state of the entire 64-bit word. Inside the loop, we compute `new_word` by using the mask to clear the target bits in our local copy (`old_word`) and then OR in the new value.

```rust
let atomic_word_ref = &self.storage[word_index];
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

The critical instruction is `compare_exchange_weak`. It attempts to atomically replace the value at `atomic_word_ref` with `new_word`, but only if its current value is still equal to `old_word`. If another thread has modified the word in the meantime, the operation fails, returns the now-current value, and our loop continues with the updated `old_word`. We use the `weak` variant because it can be more performant on some architectures and is perfectly suitable within a retry loop. Putting everything together, we obtain the following `atomic_store` implementation for the lock-free path:

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

This entire path is lock-free. Contention is managed at the hardware level by the CPU's cache coherency protocol, which ensures that only one core can successfully commit a CAS operation to a given cache line at a time. 

In the same way, we can implement other atomic operations like `atomic_rmw` (for `fetch_add`, `fetch_sub`, etc.) and `atomic_load` using similar CAS loops or direct atomic loads.

### Lock-Based Path for Spanning Elements

Now let's move to the more complex case when an element spans two `AtomicU64` words. Here the lock-free CAS loop is no longer enough. An atomic update would require to modify two separate `AtomicU64` words, and no single hardware instruction can do this as one indivisible transaction. To ensure atomicity, we must use a lock.

A single, global `Mutex` would serialize all concurrent writes, becoming an unacceptable performance bottleneck.  We can instead employ **lock striping**, a concurrency pattern that partitions the data structure to reduce the scope of contention. The core idea is to maintain an array of locks, and map different regions of our data to different locks. We can add to our `AtomicFixedVec` a `Vec<parking_lot::Mutex<()>>`. To perform an operation on a spanning element that touches `word_index` and `word_index + 1`, a thread first acquires a specific lock from this pool. The mapping is done by hashing the word index to a lock index. Since the number of locks is a power of two, we can use a fast bitwise AND for this mapping instead of a slower modulo operation: `let lock_index = word_index & (self.locks.len() - 1)`.

In this way we are partitioning the vector into a set of independent regions. A locked write to words `(i, i+1)` will not block a simultaneous locked write to words `(j, j+1)` or a lock-free write to word `k`, provided they map to different locks in the stripe. We now need a heuristic to determine, at construction time, how many locks to create. Ideally we would like to balance contention reduction against memory overhead. This could be a valuable option:

```rust
// Heuristic to determine the number of locks for striping.
let num_cores = std::thread::available_parallelism().map_or(MIN_LOCKS, |n| n.get());
let target_locks = (num_words / WORDS_PER_LOCK).max(1);
let num_locks = (target_locks.max(num_cores) * 2)
    .next_power_of_two()
    .min(MAX_LOCKS);
```

The logic aims to create enough locks to service the available hardware threads (`num_cores`) and to cover the data with a reasonable density (one lock per 64 words, `WORDS_PER_LOCK`). We take the maximum of these two, multiply by two as a simple overprovisioning factor, and round up to the next power of two to enable fast mapping via bitwise AND. The total number of locks is capped to prevent excessive memory consumption for the lock vector itself.

> **In this the best way?** I don't know, probably not. For instance, a *seqlock* could theoretically offer higher performance for read-heavy workloads. A writer would increment a version counter, perform the non-atomic two-word write, and increment it again. Readers would check for a consistent version number to validate that their read was not interrupted. However, this pattern is fundamentally incompatible with Rust's safety guarantees. A reader in a seqlock might observe a "torn read" (a state where one word has been updated but the other has not). This constitutes a data race, which safe Rust's memory model is designed to prevent. A correct seqlock implementation requires a ton of `unsafe` code, volatile reads, and careful memory fencing. Given this constrains (and that I am not an expert in lock-free programming), we can stick to the simpler yet performant implementation bases on `Mutex`. At least, power of two `bit_width` values can avoid the spanning case entirely.

With the lock striping mechanism cleared, we can now complete the implementation for our `atomic_store` method. The first step is to acquire the correct lock.

```rust
let lock_index = word_index & (self.locks.len() - 1);
let _guard = self.locks[lock_index].lock();
```

Once the lock is acquired, we have exclusive access for this two-word transaction. However, it's critical that we continue to use atomic operations to modify the words themselves. This is because another thread might be concurrently executing a *lock-free* operation on an adjacent, non-spanning element that happens to reside in `word_index` or `word_index + 1`. Using non-atomic writes inside the lock would create a data race with those lock-free operations.

We therefore use `fetch_update`, another CAS-based atomic, to modify each of the two words. For the lower word, we clear the high bits starting from our `bit_offset` and OR in the low bits of our new value.

```rust
let low_word_ref = &self.storage[word_index];
let high_word_ref = &self.storage[word_index + 1];

// Modify the lower word.
low_word_ref
    .fetch_update(order, Ordering::Relaxed, |mut w| {
        w &= !(u64::MAX << bit_offset);
        w |= value << bit_offset;
        Some(w)
    })
    .unwrap(); // This unwrap is safe: with the lock held, there is no contention.
```

Next, we do the same for the higher word, calculating a `high_mask` to clear the low bits and writing the remaining high bits of our value.

```rust
let bits_in_high = (bit_offset + self.bit_width) - u64::BITS as usize;
let high_mask = (1u64 << bits_in_high).wrapping_sub(1);
high_word_ref
    .fetch_update(order, Ordering::Relaxed, |mut w| {
        w &= !high_mask;
        w |= value >> (u64::BITS as usize - bit_offset);
        Some(w)
    })
    .unwrap(); // Also safe.
```

The lock provides the mutual exclusion that makes the two-word update appear atomic as a whole. The continued use of atomics with the user-specified `Ordering` ensures that the final result is correctly propagated through the memory system and becomes visible to other threads according to the Rust memory model.

Putting it all together, the final, complete `atomic_store` method handles both the lock-free and locked paths.

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

The branching logic `if bit_offset + self.bit_width <= 64` is a simple, constant-time check that determines which path to take. The condition is highly predictable for the CPU's branch predictor, as the `bit_offset` follows a simple arithmetic progression based on the index. This minimizes pipeline stalls. More importantly, this structure enables significant compiler optimization. When we use `AtomicFixedVec` with a `bit_width` that is a power of two, the branch condition can often be resolved at compile time. With Link-Time Optimization (LTO), the compiler can prove that the `else` branch for the locked path is unreachable. This allows for dead code elimination, producing a specialized function containing *only* the lock-free CAS loop.

#### Benchmarking the `store` operation

We now want to measure the performance of our `atomic_store` implementation under concurred load. We want to know how the throughput of the `store` operation scales as we increase thread count and contention. We can create two benchmark scenarios, one with a bit-width that is a power of two, and one with a non-power-of-two bit-width. Both simulate "diffuse contention": a pool of threads performs a high volume of atomic `store` operations to random indices within a shared vector of 10k 64-bit elements. Each thread is assigned 100k operations. This randomness ensures that while direct contention on the same element is rare, there will be frequent collisions on the same `AtomicU64` words, especially as thread count increases.

In the benchmark, each thread with `thread_id` simply writes its own ID to the vector for each operation:

```rust
vec[index].store(thread_id, Ordering::SeqCst);
```

We compare our `AtomicFixedVec` against two other implementations:
1.  A `Vec<AtomicU16>`: This serves as our "ideal" baseline.
2.  [`sux::AtomicBitFieldVec`]: A similar implementation of another library. This comparison however is not perfectly equivalent for bit-widths that are not powers of two. As per their documentation, `sux` does not guarantee atomicity for values that span word boundaries, which can lead to "torn writes." Our `AtomicFixedVec` is designed to prevent this class of data race through its lock striping mechanism. The performance cost of this correctness guarantee is precisely what we aim to measure.

The first benchmark measure the performance of our **lock-free path**. We configure all vectors with a `bit_width` of 16. Because 16 is a power of two and evenly divides the 64-bit word size, every element is guaranteed to be fully contained within a single `u64` word. This is the best-case scenario for bit-packed atomics. It ensures that all `store` operations can be performed with lock-free CAS loops.

<iframe src="/bench-intvec/atomic_scaling_lock_free_diffuse.html" width="100%" height="550px" style="border: none;"></iframe>

As we can see, all three implementations scale well with increasing thread count. The throughput of `AtomicFixedVec` is very close to that of `sux::AtomicBitFieldVec`, which is expected since both are using similar lock-free CAS loops for this scenario. Both bit-packed vectors have a noticeable dip at two threads, likely due to initial cache coherency overhead, but then scale up effectively with the core count.

The second benchmark tries to stress the **locked path**. We configure the vectors with a `bit_width` of 15. This non-power-of-two width guarantees that a predictable fraction of writes will cross word boundaries. In our case, `(15 + 63) % 64` spanning cases out of 64 offsets, so roughly `14/64` or ~22% will require locking. This forces our `AtomicFixedVec` to use its lock striping mechanism for those spanning writes. In contrast, `sux` will proceed without locking, risking data races but avoiding locking overhead. 

<iframe src="/bench-intvec/atomic_scaling_locked_path_diffuse.html" width="100%" height="550px" style="border: none;"></iframe>

This benchmark shows the real cost of correctness. Our `AtomicFixedVec` now shows lower throughput and poorer scaling compared to both the baseline and `sux`. Every write operation that crosses a word boundary (approximately 22% of them in this test) must acquire a lock, execute its two atomic updates, and release the lock. While with lock striping we prevent a single point of serialization, the overhead of the locking protocol itself, especially under contention from multiple threads, is non-trivial. In contrast, `sux` maintains higher throughput by avoiding locks entirely, but at the cost of potentially observing torn writes.

[`sux::AtomicBitFieldVec`]: https://docs.rs/sux/latest/sux/bits/bit_field_vec/struct.AtomicBitFieldVec.html

### Read-Modify-Write Operations

With the hybrid atomicity model defined, the next step is to build a robust API. Instead of re-implementing the complex hybrid logic for every atomic operation, we can implement it once in a single, powerful primitive: `compare_exchange`. All other Read-Modify-Write (RMW) operations can then be built on top of this primitive.

`compare_exchange` attempts to store a `new` value into a location if and only if the value currently at that location matches an expected `current` value. This operation is the fundamental building block for lock-free algorithms.

#### The Lock-Free Path: A CAS Loop on a Sub-Word

For elements that are fully contained within a single `AtomicU64`, we can implement `compare_exchange` with a CAS loop. However, the logic is more complex than a simple store. We aren't comparing the entire 64-bit word; we are comparing a specific `bit_width` slice within it.

The process is as follows. First, we load the entire 64-bit word. Then, we extract the specific bits that correspond to our logical element and compare them against the user-provided `current` value.

```rust
// Inside the lock-free path of atomic_compare_exchange
let mut old_word = atomic_word_ref.load(failure);
loop {
    let old_val_extracted = (old_word >> bit_offset) & self.mask;
    if old_val_extracted != current {
        return Err(old_val_extracted);
    }
    // ...
```

If this check fails, it means another thread has modified the element since our initial load. We can immediately return an `Err` with the value we just read, satisfying the contract of `compare_exchange`.

If the check succeeds, we proceed to the atomic write phase. We prepare a `new_word` by masking and ORing in the `new` value, just as we did for `atomic_store`. Then we attempt the hardware-level `compare_exchange_weak`.

```rust
// ... continuing the loop ...
let new_word = (old_word & !store_mask) | new_value_shifted;
match atomic_word_ref.compare_exchange_weak(old_word, new_word, success, failure) {
    Ok(_) => return Ok(current),
    Err(x) => old_word = x,
}
```

If the `compare_exchange_weak` succeeds, it means no other thread modified the 64-bit word between our initial `load` and this instruction. Our update is successful. If it fails, another thread (possibly operating on an *adjacent* element in the same word) has modified the word. We update our local `old_word` with the new value from memory and retry the entire loop.

#### The Locked Path

For elements that span two words, we need to use the lock striping mechanism to ensure the operation is transactional. First, we must acquire the appropriate lock, giving us exclusive access to the two-word region.

Once the lock is held, the logic is straightforward:

1.  **Read:** We perform an atomic read of the spanning value. This is itself a locked operation to ensure we get a consistent view.
2.  **Compare:** We compare this value against the user-provided `current`. If they don't match, we release the lock and immediately return `Err`.
3.  **Write:** If they do match, we call our existing `atomic_store` method to write the `new` value. `atomic_store` will re-acquire the same lock to perform its two-word write.
4.  **Return:** We release the lock and return `Ok`.

Because the entire read-compare-write sequence is protected by the same mutex, we guarantee that no other thread can interfere. Combining these two paths gives us the complete `atomic_compare_exchange` method, which can then be used to implement all other atomic operations in terms of it.

```rust
fn atomic_compare_exchange(
    &self,
    index: usize,
    current: u64,
    new: u64,
    success: Ordering,
    failure: Ordering,
) -> Result<u64, u64> {
    let bit_pos = index * self.bit_width;
    let word_index = bit_pos / u64::BITS as usize;
    let bit_offset = bit_pos % u64::BITS as usize;

    if bit_offset + self.bit_width <= u64::BITS as usize {
        // Lock-free path
        let atomic_word_ref = &self.storage[word_index];
        let store_mask = self.mask << bit_offset;
        let new_value_shifted = new << bit_offset;
        let mut old_word = atomic_word_ref.load(failure);
        loop {
            let old_val_extracted = (old_word >> bit_offset) & self.mask;
            if old_val_extracted != current {
                return Err(old_val_extracted);
            }
            let new_word = (old_word & !store_mask) | new_value_shifted;
            match atomic_word_ref.compare_exchange_weak(old_word, new_word, success, failure) {
                Ok(_) => return Ok(current),
                Err(x) => old_word = x,
            }
        }
    } else {
        // Locked path
        let lock_index = word_index & (self.locks.len() - 1);
        let _guard = self.locks[lock_index].lock();
        let old_val = self.atomic_load(index, failure);
        if old_val != current {
            return Err(old_val);
        }
        self.atomic_store(index, new, success);
        Ok(current)
    }
}
```
With `atomic_compare_exchange` providing the core atomicity primitive, we can now construct the high-level Read-Modify-Write (RMW) API. All RMW operations, such as `fetch_add` or `fetch_max`, share an identical algorithmic structure: a loop that repeatedly reads a value, computes a new value, and attempts to commit it with `compare_exchange`. Re-implementing this loop for every operation would be redundant.

Instead, we can abstract this pattern into a single generic method, `atomic_rmw`, that is parameterized not just over values, but over the *operation itself*. This is where Rust generics and closures come into play. The signature of `atomic_rmw` can be defined as follows:

```rust
fn atomic_rmw<F>(&self, index: usize, val: T, order: Ordering, op: F) -> T
where
    F: Fn(T, T) -> T
{
    // ... implementation ...
}
```
*Note: The actual implementation uses `impl Fn(...)` for a more ergonomic syntax, but the principle is the same.*

The `op` parameter is a generic type `F` constrained by the `Fn(T, T) -> T` trait. This means `op` can be any closure (or function pointer) that takes two values of our logical type `T` and returns a new `T`. This allows us to inject any binary operation—addition, bitwise AND, max, etc. directly into the RMW logic.

To implement `atomic_rmw`, we begin with a relaxed load to get an initial `current` value. Inside the loop, it invokes the `op` closure with the `current` value and the user-provided `val` to compute the `new` value.

```rust
let mut current = self.load(index, Ordering::Relaxed);
loop {
    let new = op(current, val);
    // ...
```

The core of the loop is the call to `atomic_compare_exchange`. This attempts to commit the transaction.

```rust
match self.compare_exchange(index, current, new, order, Ordering::Relaxed) {
    Ok(old) => return old,
    Err(actual) => current = actual,
}
```

If the operation succeeds, we return the old value, satisfying the RMW contract. If it fails due to contention, `compare_exchange` returns the `actual` value now in memory. We update our local `current` and the loop repeats, re-applying the `op` closure on the newer value. The complete `atomic_rmw` method looks like this:

```rust
fn atomic_rmw(&self, index: usize, val: T, order: Ordering, op: impl Fn(T, T) -> T) -> T {
    let mut current = self.load(index, Ordering::Relaxed);
    loop {
        let new = op(current, val);
        match self.compare_exchange(index, current, new, order, Ordering::Relaxed) {
            Ok(old) => return old,
            Err(actual) => current = actual,
        }
    }
}
```

By parameterizing over the operation, we have completely decoupled the complex, low-level concurrency logic from the simple arithmetic or bitwise logic of each specific RMW operation. This allows the compiler to monomorphize and inline the `op` closure for each call site, resulting in specialized and efficient machine code for each RMW variant, giving us a zero-cost abstraction.

Now implementing specific RMW methods becomes a straightforward exercise in composition. Each specific operation is now a simple non-looping wrapper that calls `atomic_rmw` and provides a closure defining the desired computation. For example, `fetch_add` is implemented by passing a closure that performs a wrapping addition.

```rust
pub fn fetch_add(&self, index: usize, val: T, order: Ordering) -> T {
    self.atomic_rmw(index, val, order, |a, b| a.wrapping_add(&b))
}
```

The same pattern applies to all other RMW operations. A bitwise `fetch_and` provides a closure with the `&` operator, and `fetch_max` provides one that calls the `max` method.

```rust
pub fn fetch_sub(&self, index: usize, val: T, order: Ordering) -> T {
    self.atomic_rmw(index, val, order, |a, b| a.wrapping_sub(&b))
}

pub fn fetch_and(&self, index: usize, val: T, order: Ordering) -> T {
    self.atomic_rmw(index, val, order, |a, b| a & b)
}

pub fn fetch_or(&self, index: usize, val: T, order: Ordering) -> T {
    self.atomic_rmw(index, val, order, |a, b| a | b)
}

pub fn fetch_xor(&self, index: usize, val: T, order: Ordering) -> T {
    self.atomic_rmw(index, val, order, |a, b| a ^ b)
}

pub fn fetch_max(&self, index: usize, val: T, order: Ordering) -> T {
    self.atomic_rmw(index, val, order, |a, b| a.max(b))
}

pub fn fetch_min(&self, index: usize, val: T, order: Ordering) -> T {
    self.atomic_rmw(index, val, order, |a, b| a.min(b))
}
```