---
author: Luca Lombardo
pubDatetime: 2025-09-16T00:00:00Z
title: Building a Thread-Safe, Bit-Packed Atomic Vector in Rust
slug: thread-safe-fixedvec
featured: true
draft: true
tags:
  - Rust
  - Succinct Data Structures
description: Engineering a thread-safe, bit-packed integer vector in Rust, using a hybrid model of lock-free and lock-based concurrency to handle operations that span hardware word boundaries.
---

This is the second post of a series on how to build a memory-efficient vector in Rust. All this work is open source and is currently being used for research projects in succinct data structures.

* All the code can be found on github: [compressed-intvec](https://github.com/lukefleed/compressed-intvec)
* This is also published as a crate on crates.io: [compressed-intvec](https://crates.io/crates/compressed-intvec)

## Table of Contents

In the [first post]([link-to-part-1](https://lukefleed.xyz/posts/compressed-fixedvec/)), we built `FixedVec`, a bit-packed integer vector that gives us O(1) random access while using a fraction of the memory of a standard `Vec<T>`. We worked through the core problem of reading bit-packed data, and saw how a single unaligned memory read could let us efficiently handle values that span across 64-bit word boundaries. We ended up with a solid, fast data structure complete with a clean, ergonomic API.

So, we have a great single-threaded vector. But what happens when we need to share it between threads? The natural next step is to build an `AtomicFixedVec` with an API that feels like Rust's standard atomics: `load`, `store`, `fetch_add`, and so on.

As soon as we try, we hit a wall. The atomic instructions our CPUs give us work on nicely aligned, word-sized chunks of memory. But our data doesn't live like that. A 10-bit integer in our vector is just a virtual concept, a sequence of bits that might start in the middle of one `u64` and end in another.

This leads to the fundamental problem: how can we possibly implement an atomic `fetch_add` on a value that's physically split across two different `AtomicU64`s? There's no single hardware instruction that can atomically modify two memory locations at once. A naive attempt would create torn writes and all sorts of nasty data races.

In this post, we're going to walk through the engineering behind `AtomicFixedVec` and solve this problem. The solution is a hybrid model that combines lock-free, high-performance CAS loops for the common case with a fine-grained locking mechanism to correctly and safely handle the values that span word boundaries.

# Structure

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

## Lock-Free path for In-Word Elements

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

## Lock-Based Path for Spanning Elements

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

# Benchmarking the `store` operation

We now want to measure the performance of our `atomic_store` implementation under concurred load. We want to know how the throughput of the `store` operation scales as we increase thread count and contention. We can create two benchmark scenarios, one with a bit-width that is a power of two, and one with a non-power-of-two bit-width. Both simulate "diffuse contention": a pool of threads performs a high volume of atomic `store` operations to random indices within a shared vector of 10k 64-bit elements. Each thread is assigned 100k operations. This randomness ensures that while direct contention on the same element is rare, there will be frequent collisions on the same `AtomicU64` words, especially as thread count increases.

In the benchmark, each thread with `thread_id` simply writes its own ID to the vector for each operation:

```rust
vec[index].store(thread_id, Ordering::SeqCst);
```

We compare our `AtomicFixedVec` against two other implementations:
1.  A `Vec<AtomicU16>`: This serves as our "ideal" baseline.
2.  [`sux::AtomicBitFieldVec`]: A similar implementation of another library. This comparison however is not perfectly equivalent for bit-widths that are not powers of two. As per their documentation, `sux` does not guarantee atomicity for values that span word boundaries, which can lead to "torn writes." Our `AtomicFixedVec` is designed to prevent this class of data race through its lock striping mechanism. The performance cost of this correctness guarantee is precisely what we aim to measure.

The first benchmark measure the performance of our **lock-free path**. We configure all vectors with a `bit_width` of 16. Because 16 is a power of two and evenly divides the 64-bit word size, every element is guaranteed to be fully contained within a single `u64` word. This is the best-case scenario for bit-packed atomics. It ensures that all `store` operations can be performed with lock-free CAS loops.

<iframe src="/bench-intvec/atomic_scaling_lock_free_diffuse/" width="100%" height="550px" style="border: none;"></iframe>

As we can see, all three implementations scale well with increasing thread count. The throughput of `AtomicFixedVec` is very close to that of `sux::AtomicBitFieldVec`, which is expected since both are using similar lock-free CAS loops for this scenario. Both bit-packed vectors have a noticeable dip at two threads, likely due to initial cache coherency overhead, but then scale up effectively with the core count.

The second benchmark tries to stress the **locked path**. We configure the vectors with a `bit_width` of 15. This non-power-of-two width guarantees that a predictable fraction of writes will cross word boundaries. In our case, `(15 + 63) % 64` spanning cases out of 64 offsets, so roughly `14/64` or ~22% will require locking. This forces our `AtomicFixedVec` to use its lock striping mechanism for those spanning writes. In contrast, `sux` will proceed without locking, risking data races but avoiding locking overhead.

<iframe src="/bench-intvec/atomic_scaling_locked_path_diffuse/" width="100%" height="550px" style="border: none;"></iframe>

This benchmark shows the real cost of correctness. Our `AtomicFixedVec` now shows lower throughput and poorer scaling compared to both the baseline and `sux`. Every write operation that crosses a word boundary (approximately 22% of them in this test) must acquire a lock, execute its two atomic updates, and release the lock. While with lock striping we prevent a single point of serialization, the overhead of the locking protocol itself, especially under contention from multiple threads, is non-trivial. In contrast, `sux` maintains higher throughput by avoiding locks entirely, but at the cost of potentially observing torn writes.

[`sux::AtomicBitFieldVec`]: https://docs.rs/sux/latest/sux/bits/bit_field_vec/struct.AtomicBitFieldVec.html

# Read-Modify-Write Operations

With the hybrid atomicity model defined, the next step is to build a robust API. Instead of re-implementing the complex hybrid logic for every atomic operation, we can implement it once in a single, powerful primitive: `compare_exchange`. All other Read-Modify-Write (RMW) operations can then be built on top of this primitive.

`compare_exchange` attempts to store a `new` value into a location if and only if the value currently at that location matches an expected `current` value. This operation is the fundamental building block for lock-free algorithms.

## The Lock-Free Path: A CAS Loop on a Sub-Word

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

## The Locked Path

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
