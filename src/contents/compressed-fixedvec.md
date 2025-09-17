---
author: Luca Lombardo
datetime: 2025-08-24
title: Engineering a Fixed-Width bit-packed Integer Vector in Rust
slug: compressed-fixedvec
featured: true
university: false
posts: true
draft: false
tags:
  - Rust
  - Succinct Data Structures
ogImage: "/assets/fixedvec.png"
description: Design and implementation of a memory-efficient, fixed-width bit-packed integer vector in Rust, with extremely fast random access.
---

If you've ever worked with massive integer datasets, you know that memory usage can quickly become a bottleneck. While developing succinct data structures, I found myself needing to store large arrays of integers—values with no monotonicity or other exploitable patterns, that I knew came from a universe much smaller than their type's theoretical capacity.

In this post, we will explore the engineering challenges involved in implementing an efficient vector-like data structure in Rust that stores integers in a compressed, bit-packed format. We will focus on achieving O(1) random access performance while minimizing memory usage. We will try to mimic the ergonomics of Rust's standard `Vec<T>` as closely as possible, including support for mutable access and zero-copy slicing.


* All the code can be found on github: [compressed-intvec](https://github.com/lukefleed/compressed-intvec)
* This is also published as a crate on crates.io: [compressed-intvec](https://crates.io/crates/compressed-intvec)

---

# Memory Waste in Standard Vectors

In Rust, the contract of a `Vec<T>` (where `T` is a primitive integer type like `u64` or `i32`) is simple: O(1) random access in exchange for a memory layout that is tied to the static size of `T`. This is a good trade-off, until it isn't. When the dynamic range of the stored values is significantly smaller than the type's capacity, this memory layout leads to substantial waste.

Consider storing the value `5` within a `Vec<u64>`. Its 8-byte in-memory representation is:

<div align="center">

`00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000101`

</div>

Only 3 bits are necessary to represent the value, leaving 61 bits as zero-padding. The same principle applies, albeit less dramatically, when storing `5` in a `u32` or `u16`. At scale, this overhead becomes prohibitive. A vector of one billion `u64` elements consumes `10^9 * std::mem::size_of::<u64>()`, or approximately 8 GB of memory, even if every element could fit within a single byte.

The canonical solution is bit packing, which aligns data end-to-end in a contiguous bitvector. However, this optimization has historically come at the cost of random access performance. The O(1) access guarantee of `Vec<T>` is predicated on simple pointer arithmetic: `address = base_address + index * std::mem::size_of::<T>()`. Tightly packing the bits invalidates this direct address calculation, seemingly forcing a trade-off between memory footprint and access latency.

This raises the central question that with this post we aim to answer: is it possible to design a data structure that decouples its memory layout from the static size of `T`, adapting instead to the data's true dynamic range, without sacrificing the O(1) random access that makes `Vec<T>` so effective?

# Packing and Accessing Bits

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
let bit_pos = index * bit_width; // Absolute bit position of the value
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


## Crossing Word Boundaries

The single-word access logic is fast, but it only works as long as `bit_offset + bit_width <= 64`. This assumption breaks down as soon as an integer's bit representation needs to cross the boundary from one `u64` word into the next. This is guaranteed to happen for any `bit_width` that is not a power of two. For example, with a 10-bit width, the element at `index = 6` starts at bit position 60. Its 10 bits will occupy bits 60-63 of the first word and bits 0-5 of the second. The simple right-shift-and-mask trick fails here.

![crossing word boundaries](/assets/fixedvec.svg)

To correctly decode the value, we must read *two* consecutive `u64` words and combine their bits. This splits our `get_unchecked` implementation into two paths. The first is the fast path we've already seen. The second is a new path for spanning values.

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

The line `limbs.get_unchecked(word_index + 1)` introduces a safety concern: if we are reading the last element of the vector, `word_index + 1` could point past the end of our buffer, leading to undefined behavior. To prevent this, our builder must always allocate one extra padding word at the end of the storage.

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

## Faster Reads: Unaligned Access

Our `get_unchecked` implementation is correct, but the slow path for spanning values requires two separate, aligned memory reads. The instruction sequence for this method involves at least two load instructions, multiple shifts, and a bitwise OR. These instructions have data dependencies: the shifts cannot execute until the loads complete, and the OR cannot execute until both shifts are done. This dependency chain can limit the CPU's instruction-level parallelism and create pipeline stalls if the memory accesses miss the L1 cache.

Let's have a look at the machine code generated by this method. We can create a minimal binary with an `#[inline(never)]` function that calls `get_unchecked` on a known spanning index, then use [`cargo asm`](https://crates.io/crates/cargo-asm) to inspect the disassembly.

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

This sequence of operations—loading two adjacent 64-bit words, shifting each, and combining them is a software implementation of what is, conceptually, a [128-bit barrel shifter](https://en.wikipedia.org/wiki/Barrel_shifter). We are selecting a 64-bit window from a virtual 128-bit integer formed by concatenating the two words from memory.

**Can we do better then this?** Potentially, yes. We can replace this multi-instruction sequence with something more direct by delegating the complexity to the hardware and performing a single unaligned memory read. Modern x86-64 CPUs handle this directly: when an unaligned load instruction is issued, the CPU's memory controller fetches the necessary cache lines and the load/store unit reassembles the bytes into the target register. This entire process is a single, optimized micro-operation.

We can try to implement a more aggressive access method. The strategy is to calculate the exact *byte* address where our data begins and perform a single, unaligned read of a full `W` word from that position.

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

This operation is safe only because, as said before, we are supposing that our builder guarantees a padding word at the end of the storage buffer. Combining all these steps, we get our final implementation:

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

Let's have a look at the generated machine code for this new method when accessing an index that spans words:

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

## Random Access Performance

We can now benchmark the latency of 1 million random access operations on a vector containing 10 million elements. For each `bit_width`, we generate data with a uniform random distribution in the range `[0, 2^bit_width)`. The code for the benchmark is available here: [`bench-intvec`]

Our baseline is the smallest standard `Vec<T>` capable of holding the data (`Vec<u8>` for `bit_width <= 8`, etc.). We also include results from [`sux`], [`succinct`], and [`simple-sds-sbwt`] for context. I am not aware of any other Rust crates that implement fixed-width bit-packed integer vectors, so if you know of any, please let me know!

<iframe src="/bench-intvec/fixed_random_access_performance.html" width="100%" height="550px" style="border: none;"></iframe>

We can see that for `bit_width` values below 32, the `get_unaligned_unchecked` of our `FixedVec` is almost always faster than the corresponding `Vec<T>` baseline. This is a result of improved cache locality. A 64-byte L1 cache line can hold 64 elements from a `Vec<u8>`. With a `bit_width` of 4, the same cache line holds `(64 * 8) / 4 = 128` elements from our `FixedVec`. This increased density improves the cache hit rate for random access patterns, and the latency reduction from avoiding DRAM access outweighs the instruction cost of the bitwise extraction. For values of `bit_width` above 32, the performance of `FixedVec` are *very slightly* worse than the `Vec<T>` baseline, as the cache locality advantage diminishes. However, the memory savings remain.

The performance delta between `get_unaligned_unchecked` and `get_unchecked` confirms the unaligned access strategy discussed before: a single `read_unaligned` instruction is more efficient than the two dependent aligned reads required by the logic for spanning words.

We can see that the implementation of [`sux`] is almost on par with ours. The other two crates, [`succinct`] and [`simple-sds-sbwt`], are significantly slower (note that the Y-axis is logarithmic). Tho, it's worth noting that neither of these last two crates provides unchecked or unaligned access methods, so their implementations are inherently more conservative.

[`sux`]: https://docs.rs/sux/latest/sux/bits/bit_field_vec/index.html
[`succinct`]: https://docs.rs/succinct/latest/succinct/trait.IntVec.html
[`simple-sds-sbwt`]: https://docs.rs/simple-sds-sbwt/latest/simple_sds_sbwt/int_vector/struct.IntVector.html
[`bench-intvec`]: https://github.com/lukefleed/compressed-intvec/blob/master/benches/fixed/bench_random_access.rs

## Iterating Over Values

The most common operation on any `Vec`-like structure is, after all, a simple `for` loop. The simplest way to implement `iter()` would be to just wrap `get()` in a loop:

```rust
// A naive, inefficient iterator
for i in 0..vec.len() {
    let value = vec.get(i);
    // ... do something with value
}
```

This works, but it's terribly inefficient. Every single call to `get(i)` independently recalculates the `word_index` and `bit_offset` from scratch. We're throwing away valuable state, our current position in the bitstream, on every iteration, forcing the CPU to perform redundant multiplications and divisions.

We can think then about a *stateful* iterator. It should operate directly on the bitvector, maintaining its own position. Instead of thinking in terms of logical indices, it should think in terms of a "bit window", a local `u64` register that holds the current chunk of bits being processed.

The idea is simple: the iterator loads one `u64` word from the backing store into its window. It then satisfies `next()` calls by decoding values directly from this in-register window. Only when the window runs out of bits does it need to go back to memory for the next `u64` word. This amortizes the cost of memory access over many `next()` calls.

For forward iteration, the state is minimal:

```rust
struct FixedVecIter<'a, ...> {
    // ...
    front_window: u64,
    front_bits_in_window: usize,
    front_word_index: usize,
    // ...
}
```

The `next()` method first checks if the current `front_window` has enough bits to satisfy the request. If `self.front_bits_in_window >= bit_width`, it's the fast path: a simple shift and mask on a register, which is incredibly fast.

```rust
// Inside next(), fast path:
if self.front_bits_in_window >= bit_width {
    let value = self.front_window & self.mask;
    self.front_window >>= bit_width;
    self.front_bits_in_window -= bit_width;
    return Some(value);
}
```

If the window is running low on bits, we hit the slower path. The next value spans the boundary between our current window and the next `u64` word in memory. We must read the next word, combine its bits with the remaining bits in our current window, and then extract the value. This is the same logic as the spanning-word `get()`, but it's performed incrementally.

### Double-Ended Iteration

But I want my iterator to be bidirectional! Well, then we need to ensure it implements [`DoubleEndedIterator`](https://doc.rust-lang.org/std/iter/trait.DoubleEndedIterator.html) and supports [`next_back()`](https://doc.rust-lang.org/std/iter/trait.DoubleEndedIterator.html#tymethod.next_back). This throws a wrench in our simple stateful model. A single window and cursor can only move in one direction.

The solution is to maintain two independent sets of state: one for the front and one for the back. The `FixedVecIter` needs to track two windows, two bit counters, and two word indices.

```rust
struct FixedVecIter<'a, ...> {
    // ...
    front_index: usize,
    back_index: usize,

    // State for forward iteration
    front_window: u64,
    front_bits_in_window: usize,
    front_word_index: usize,

    // State for backward iteration
    back_window: u64,
    back_bits_in_window: usize,
    back_word_index: usize,
    // ...
}
```

Initializing the front is easy: we load `limbs[0]` into `front_window`. The back is more complex. We must calculate the exact word index and the number of valid bits in the *last* word that contains data. This requires a bit of arithmetic to handle cases where the data doesn't perfectly fill the final word.

The `next()` method consumes from the `front_window`, advancing the front state. The `next_back()` method consumes from the `back_window`, advancing the back state. The iterator is exhausted when `front_index` meets `back_index`.

> The full implementation can be found in the iter module of the [library](https://github.com/lukefleed/compressed-intvec/blob/master/src/fixed/iter.rs)

# The other half of the problem: writing bits

We have solved the read problem, but we may also want to modify values in place. A method like `set(index, value)` seems simple, but it opens up the same can of worms as `get`, just in reverse. We can't just write the new value; we have to do so without clobbering the adjacent, unrelated data packed into the same `u64` word.

Just like with reading, the logic splits into two paths. The "fast path" handles values that are fully contained within a single `u64`. Here, we can't just overwrite the word. We first need to clear out the bits for the element we're replacing and then merge in the new value.

## In-Word Write

Our goal here is to update a `bit_width`-sized slice of a `u64` word while leaving the other bits untouched. This operation must be a read-modify-write sequence to avoid corrupting adjacent elements. The most efficient way to implement this is to load the entire word into a register, perform all bitwise modifications locally, and then write the final result back to memory in a single store operation.

First, we load the word from our backing `limbs` slice.

```rust
let mut word = *limbs.get_unchecked(word_index);
```

Next, we need to create a "hole" in our local copy where the new value will go. We do this by creating a mask that has ones *only* in the bit positions we want to modify, and then inverting it to create a clearing mask.

```rust
// For a value at bit_offset, the mask must also be shifted.
let clear_mask = !(self.mask << bit_offset);
// Applying this mask zeroes out the target bits in our register copy.
word &= clear_mask;
```

With the target bits zeroed, we can merge our new value. The value is first shifted left by `bit_offset` to align it correctly within the 64-bit word. Then, a bitwise OR merges it into the "hole" we just created.

```rust
// Shift the new value into position and merge it.
word |= value_w << bit_offset;
```

Finally, with the modifications complete, we write the updated word from the register back to memory in a single operation.

```rust
*limbs.get_unchecked_mut(word_index) = word;
```
This entire sequence of one read, two bitwise operations in-register, one write is the canonical and most efficient way to perform a sub-word update.

## Spanning Write

Now for the hard part: writing a value that crosses a word boundary. This operation must modify two separate `u64` words in our backing store. It's the inverse of the spanning read. We need to split our `value_w` into a low part and a high part and write each to the correct word, minimizing memory accesses.

To operate on two distinct memory locations, `limbs[word_index]` and `limbs[word_index + 1]`, we first need mutable access to both. In a safe, hot path like this, we can use `split_at_mut_unchecked` to bypass Rust's borrow checker bounds checks, as we have already guaranteed through our logic that both indices are valid.

```rust
// SAFETY: We know word_index and word_index + 1 are valid.
let (left, right) = limbs.split_at_mut_unchecked(word_index + 1);
```

Our strategy is to read both words into registers, perform all bitwise logic locally, and then write both modified words back to memory. This minimizes the time we hold mutable references and can improve performance.

First, we handle the `low_word`. We need to replace its high bits (from `bit_offset` onwards) with the low bits of our new value. The most direct way is to create a mask for the bits we want to *keep*. The expression `(1 << bit_offset) - 1` is a bit-twiddling trick to generate a mask with `bit_offset` ones at the least significant end.

```rust
let mut low_word_val = *left.get_unchecked(word_index);

// Create a mask to preserve the low `bit_offset` bits of the word.
let low_mask = (1u64 << bit_offset).wrapping_sub(1);
low_word_val &= low_mask;
```

With the target bits zeroed out, we merge in the low part of our new value. A left shift aligns it correctly, and the high bits of `value_w` are naturally shifted out of the register.

```rust
// Merge in the low part of our new value.
low_word_val |= value_w << bit_offset;
*left.get_unchecked_mut(word_index) = low_word_val;
```

Next, we handle the `high_word` in a symmetrical fashion. We need to write the remaining high bits of `value_w` into the low-order bits of this second word. First, we calculate how many bits of our value actually belong in the first word:

```rust
let remaining_bits_in_first_word = 64 - bit_offset;
```

Now, we read the second word and create a mask to clear the low-order bits where our data will be written. With the operation `self.mask >> remaining_bits_in_first_word` we can determine how many bits of our value spill into the second word, creating a mask for them. Inverting this gives us a mask to *preserve* the existing high-order bits of the `high_word`.

```rust
let mut high_word_val = *right.get_unchecked(0);

// Clear the low bits of the second word that will be overwritten.
high_word_val &= !(self.mask >> remaining_bits_in_first_word);
```

Finally, we isolate the high part of `value_w` by right-shifting it by the number of bits we already wrote, and merge it into the cleared space.

```rust
// Merge in the high part of our new value.
high_word_val |= value_w >> remaining_bits_in_first_word;
*right.get_unchecked_mut(0) = high_word_val;
```

## Random Write Performance

As with reads, we can benchmark the latency of 1 million random write operations on a vector containing 10 million elements. The code for the benchmark is available here: [`bench-intvec-writes`].

<iframe src="/bench-intvec/random_write_performance.html" width="100%" height="550px" style="border: none;"></iframe>

Here, the `Vec<T>` baseline is the clear winner across almost all bit-widths. This isn't surprising. A `set` operation in a `Vec<T>` compiles down to a single `MOV` instruction with a simple addressing mode (`[base + index * element_size]`). It's about as fast as the hardware allows.

As for the reads, the performance of our `FixedVec` is almost identical to that of [`sux`]. The other two crates, [`succinct`] and [`simple-sds-sbwt`], are again slower. It's worth noting that also for the writes, neither of these last two crates provides unchecked methods.

> For the 64-bit width case, I honestly have no idea what is going on with [`sux`] being so much faster than everything else, even then `Vec<u64>`! Mybe some weird compiler optimization? If you have any insight, please let me know.

# The Architecture

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

## `Word` Trait: The Physical Storage Layer

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

## `Storable` Trait: The Logical Type Layer

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

## Mutability: Proxy Objects

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

## Mutable Slices

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

# Next Step: Concurrency

So, where does this leave us? We started with a simple goal: to build a vector that doesn't waste memory for small integers. We discovered that a single unaligned read could be the key to outperforming a standard Vec in random access workloads. The conventional wisdom is that compression costs CPU cycles, but this proves it's not always true. Sometimes, by trading a few more instructions for better cache locality, we can get the best of both worlds: **less memory and more speed**.

We've built a fast, memory-efficient, and ergonomic vector. But our work is only half-done. Everything we've discussed so far falls apart in a multi-threaded world. How do you atomically modify a value that doesn't even exist as a single unit in memory? We'll tackle that challenge in the next post
