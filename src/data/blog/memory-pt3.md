---
author: Luca Lombardo
pubDatetime: 2025-12-29T00:00:00.000Z
title: "Who Owns the Memory? Part 3: How Big Is your Type?"
slug: who-owns-the-memory-pt3
featured: false
draft: false
tags:
  - Rust
  - Programming
description: "Type layout, fat pointers, and the cost of polymorphism."
---

This is the third part of a series exploring how C, C++, and Rust manage memory at a low level. In [Part I](https://lukefleed.xyz/posts/who-owns-the-memory-pt1/) we examined how memory is organized at the hardware level. [Part II](https://lukefleed.xyz/posts/who-owns-the-memory-pt2/) explored ownership and lifetime, showing how the three languages answer the question of who is responsible for freeing memory and when access to that memory is valid.

Part III turns to _representation_: how abstract types become concrete bit patterns, and how polymorphism, the ability to write code that operates on many types, is implemented.

You can discuss this article on [Lobsters](https://lobste.rs/s/gykpyi/who_owns_memory_part_3_how_big_is_your_type),  Reddit ([r/rust](https://www.reddit.com/r/rust/comments/1q4qe9x/who_owns_the_memory_part_3_how_big_is_your_type/) and [r/programming](https://www.reddit.com/r/programming/comments/1q4qdxa/who_owns_the_memory_part_3_how_big_is_your_type/)) and [Hacker News](https://news.ycombinator.com/item?id=46500833)

## Table of Contents

## Type Layout and Memory Representation

In Part I we established that every type has a size and an alignment, and that compilers insert padding to satisfy alignment constraints. We showed how a poorly ordered struct in C could waste 8 bytes of padding where a well-ordered one used only 2. But we sidestepped a question: who decides the order?

In C and C++, the answer is straightforward: we do. The compiler lays out fields in declaration order, inserting padding as the alignment algorithm dictates. This predictability is essential for binary compatibility, memory-mapped I/O, and network protocols where byte offsets must match external specifications. It is also a constraint: we bear responsibility for field ordering, and a careless declaration can bloat a frequently-allocated struct.

Rust makes a different choice. By default, the compiler reserves the right to reorder fields.

### The C Layout Algorithm

C specifies a deterministic algorithm for struct layout. Start with a current offset of zero. For each field in declaration order: add padding until the offset is a multiple of the field's alignment, record the field's offset, advance by the field's size. Finally, round the struct's total size up to its alignment.

```c
struct Example {
    char a;      // offset 0, size 1
    // 7 bytes padding (align to 8 for double)
    double b;    // offset 8, size 8
    char c;      // offset 16, size 1
    // 7 bytes padding (struct size must be multiple of 8)
};
// sizeof(struct Example) == 24
```

The algorithm is mechanical. Given field types and their order, the layout is fully determined. This property is what makes C the lingua franca of FFI: any language that implements the same algorithm can share data structures with C code.

C++ inherits this layout for standard-layout types. The Itanium ABI, which governs most non-Windows C++ implementations, extends the algorithm to handle base classes, virtual functions, and virtual inheritance. For our purposes, a C++ struct without inheritance, virtual functions, or access specifiers mixing fields follows the same layout as C.

The `[[no_unique_address]]` attribute (C++20) allows the compiler to overlap a data member with another if the member has no members of its own. This enables empty base optimization for member objects:

```cpp
struct Empty {};

struct WithoutAttribute {
    Empty e;
    int x;
};
// sizeof(WithoutAttribute) == 8 on typical platforms
// (Empty requires 1 byte in C++, padded to 4 for int alignment)

struct WithAttribute {
    [[no_unique_address]] Empty e;
    int x;
};
// sizeof(WithAttribute) == 4
```

Without the attribute, `Empty` occupies one byte (C++ mandates that different objects have different addresses, so even empty classes have nonzero size). With the attribute, `e` can share storage with padding or with `x` itself, reducing the struct to just `sizeof(int)`.

### repr(Rust): The Compiler's Prerogative

Rust's default representation, `repr(Rust)`, makes minimal guarantees. The specification states only:

1. Fields are properly aligned.
2. Fields do not overlap (except for ZSTs, which may share addresses).
3. The struct's alignment is at least the maximum alignment of its fields.

There is no guarantee about field order. The compiler may reorder fields to minimize padding, and different compilations of the same source can produce different layouts. The same generic struct instantiated with different type parameters will typically have different field orderings.

Consider:

```rust
struct Foo<T, U> {
    count: u16,
    data1: T,
    data2: U,
}
```

For `Foo<u32, u16>`, an efficient layout places `count` and `data2` (both 2 bytes) adjacent, followed by `data1` (4 bytes):

```rust
// Possible layout of Foo<u32, u16>: size 8, align 4
// count:  offset 0, size 2
// data2:  offset 2, size 2
// data1:  offset 4, size 4
```

For `Foo<u16, u32>`, the same reordering achieves the same efficiency:

```rust
// Possible layout of Foo<u16, u32>: size 8, align 4
// count:  offset 0, size 2
// data1:  offset 2, size 2
// data2:  offset 4, size 4
```

If Rust preserved declaration order, `Foo<u32, u16>` would require padding after `count`:

```rust
// Declaration-order layout of Foo<u32, u16>: size 12, align 4
// count:  offset 0, size 2
// _pad:   offset 2, size 2
// data1:  offset 4, size 4
// data2:  offset 8, size 2
// _pad:   offset 10, size 2
```

The reordering saves 4 bytes per instance without any annotation or programmer attention.

The trade-off is unpredictability. We cannot assume field offsets. We cannot cast a `*const Foo<A, B>` to a byte pointer and read at a known offset to extract a field. We cannot send a `repr(Rust)` struct across a network or write it to a file and expect another compilation (or even the same compilation with different flags) to interpret it correctly.

### repr(C): Predictable Layout for FFI

When we need C-compatible layout, we annotate the type with `#[repr(C)]`:

```rust
#[repr(C)]
struct ThreeInts {
    first: i16,
    second: i8,
    third: i32,
}
```

With `repr(C)`, Rust applies the C layout algorithm. Fields appear in declaration order. Padding follows the standard rules. The resulting layout is compatible with a C struct declared with the same field types and order:

```c
struct ThreeInts {
    int16_t first;
    int8_t second;
    int32_t third;
};
```

`repr(C)` is necessary for FFI correctness. It is also useful when we need stable layout for unsafe code that relies on field offsets, or when serializing data to a known binary format. The trade-off is that we accept whatever padding the declaration order implies.

For enums, `repr(C)` produces a layout compatible with C unions plus a tag. The exact representation depends on whether the enum has fields:

```rust
#[repr(C)]
enum MyEnum {
    A(u32),
    B(f32, u64),
    C { x: u32, y: u8 },
    D,
}
```

This is laid out as a `repr(C)` union of `repr(C)` structs, where each struct begins with a discriminant of the platform's C `int` size:

```rust
// Equivalent repr(C) layout:
#[repr(C)]
union MyEnumRepr {
    a: MyVariantA,
    b: MyVariantB,
    c: MyVariantC,
    d: MyVariantD,
}

#[repr(C)]
struct MyVariantA { tag: MyEnumDiscriminant, value: u32 }

#[repr(C)]
struct MyVariantB { tag: MyEnumDiscriminant, value0: f32, value1: u64 }

#[repr(C)]
struct MyVariantC { tag: MyEnumDiscriminant, x: u32, y: u8 }

#[repr(C)]
struct MyVariantD { tag: MyEnumDiscriminant }

#[repr(C)]
enum MyEnumDiscriminant { A, B, C, D }
```

The discriminant size for a fieldless `repr(C)` enum matches the C ABI's enum size, which is implementation-defined. On most platforms, this is `int` (4 bytes), though some ABIs use smaller types for small enums.

### Primitive Representations for Enums

When we need precise control over the discriminant, we can specify an integer type:

```rust
#[repr(u8)]
enum Opcode {
    Nop = 0,
    Load = 1,
    Store = 2,
    // ... up to 255 variants
}
```

This enum occupies exactly 1 byte. The discriminant is stored as a `u8`. This is essential for binary protocols where the tag must fit a specific field width.

For enums with fields, `repr(u8)` or similar sets the discriminant size but still uses C-style layout for the variant data:

```rust
#[repr(u8)]
enum Packet {
    Ping,
    Data([u8; 64]),
    Error(u32),
}
// size: 72 bytes (1 byte tag + 7 padding + 64 data)
// The discriminant is guaranteed to be 1 byte
```

Combining `repr(C)` and a primitive representation, like `#[repr(C, u8)]`, specifies both C layout and a specific discriminant type.

Adding an explicit `repr` to an enum with fields has a consequence: it *suppresses niche optimization*. This becomes clear after we discuss zero-sized types.

### repr(packed): Eliminating Padding

Sometimes we need minimum size regardless of alignment. Network packet headers, binary file formats, and memory-mapped hardware registers often require tightly packed data. `repr(packed)` removes inter-field padding:

```rust
#[repr(packed)]
struct PackedExample {
    a: u8,
    b: u32,
    c: u8,
}
// size: 6 bytes (1 + 4 + 1), no padding
// alignment: 1 byte
```

Compare with the default layout, which would require 12 bytes (1 + 3 padding + 4 + 1 + 3 padding). The packed version is half the size.

The penalty is that fields may be misaligned. On x86-64, misaligned loads incur a performance penalty. On stricter architectures like ARM without unaligned access support or older SPARC, they cause hardware exceptions. Even on tolerant architectures, misaligned atomics are often incorrect.

Rust addresses part of this problem by making it *illegal to create a reference to a misaligned field*:

```rust
#[repr(packed)]
struct Packed {
    a: u8,
    b: u32,  // may be at offset 1, misaligned
}

let p = Packed { a: 1, b: 2 };
let r: &u32 = &p.b;  // ERROR: reference to packed field
```

The compiler rejects this because a `&u32` must point to a 4-byte-aligned address, but `p.b` may not be. The workaround is to copy the value first:

```rust
let value = p.b;         // copies the unaligned bytes into a local
let r: &u32 = &value;    // value is properly aligned on the stack
```

Or use raw pointer operations with explicit unaligned reads:

```rust
let ptr: *const u32 = std::ptr::addr_of!(p.b);
let value = unsafe { ptr.read_unaligned() };
```

`repr(packed(n))` generalizes this by setting the maximum field alignment to `n`. Fields with natural alignment less than `n` are laid out normally; fields with natural alignment greater than `n` are treated as if their alignment were `n`. This allows partial packing, trading some space for better access patterns.

For FFI work, combining `repr(C, packed)` gives C-compatible layout with minimal size. This matches `#pragma pack(1)` in many C compilers.

### repr(align): Preventing False Sharing

While `repr(packed)` reduces alignment, `repr(align(n))` increases it. The attribute forces the type to have alignment of at least `n` bytes, where `n` must be a power of two.

Why would we want *more* alignment than necessary? The answer lies in cache architecture. On x86-64, the L1 cache operates on 64-byte cache lines. When one core writes to an address, the entire cache line containing that address is invalidated in all other cores' caches. This is the MESI protocol (or variants like MESIF, MOESI) maintaining coherence.

Consider two atomic counters that different threads increment concurrently:

```rust
use std::sync::atomic::{AtomicU64, Ordering};

struct Counters {
    a: AtomicU64,  // offset 0, 8 bytes
    b: AtomicU64,  // offset 8, 8 bytes
}
```

Both counters fit in a single 64-byte cache line. When thread 1 increments `a`, it invalidates thread 2's cache line. When thread 2 increments `b`, it invalidates thread 1's cache line. Neither thread is accessing the other's data, yet they are constantly invalidating each other's caches. This is *false sharing*, and it can degrade performance by an order of magnitude on contended workloads.

The fix is to ensure each counter occupies its own cache line:

```rust
use std::sync::atomic::{AtomicU64, Ordering};

#[repr(align(64))]
struct CacheAligned(AtomicU64);

struct Counters {
    a: CacheAligned,  // offset 0, padded to 64 bytes
    b: CacheAligned,  // offset 64, padded to 64 bytes
}
```

Now `sizeof::<Counters>()` is 128 bytes instead of 16, but `a` and `b` cannot share a cache line. Each thread's writes affect only its own cache line, and the coherence protocol stops bouncing the line between cores.

The trade-off is memory consumption. Padding a 8-byte counter to 64 bytes is an 8x increase. For a handful of hot counters in a concurrent data structure, this is negligible. For an array of thousands of counters, it becomes prohibitive. The choice depends on access patterns: if threads access disjoint counters, alignment helps. If threads frequently access the same counter, the contention is real, not false, and alignment does nothing.

`repr(align)` can combine with other representations. `#[repr(C, align(64))]` gives C-compatible field ordering with cache-line alignment. The align modifier applies to the struct as a whole, not to individual fields, so all fields remain at their natural offsets within the enlarged, aligned struct.

### repr(transparent): Zero-Cost Wrappers

Sometimes we want a new type that has identical layout to an existing type. The newtype pattern wraps a single field:

```rust
struct Meters(f64);
struct Seconds(f64);
```

With `repr(Rust)`, these types are not guaranteed to have the same layout or ABI as `f64`. A function returning `Meters` might use different registers than a function returning `f64`, even though the underlying data is identical.

`repr(transparent)` guarantees layout and ABI identity:

```rust
#[repr(transparent)]
struct Meters(f64);

#[repr(transparent)]
struct Seconds(f64);
```

Now `Meters` has exactly the same size, alignment, and function-call ABI as `f64`. We can pass a `Meters` where C expects a `double`. We can transmute between `Meters` and `f64` (in either direction) without undefined behavior.

`repr(transparent)` requires that the type has exactly one field with non-zero size, and may have any number of zero-sized fields with alignment 1 (like `PhantomData<T>`).

This enables patterns like:

```rust
use std::marker::PhantomData;

#[repr(transparent)]
struct Id<T>(u64, PhantomData<T>);
```

`Id<User>` and `Id<Post>` are distinct types at compile time but have identical layout to `u64`. The `PhantomData` participates in the type system (for variance and drop checking) but not in the layout.

### Zero-Sized Types

Rust permits types with size zero:

```rust
struct Nothing;           // no fields
struct AlsoNothing { }    // empty struct
struct MoreNothing(());   // contains unit type
struct AndNothing([u8; 0]); // zero-length array
```

All of these have size 0 and alignment 1 (the minimum). They occupy no memory. An array `[Nothing; 1000000]` has size 0.

ZSTs become useful in generic contexts. A `HashMap<K, V>` stores keys and values. What if we only care about keys? We could duplicate the code as `HashSet<K>`, or we could define:

```rust
type HashSet<K> = HashMap<K, ()>;
```

Because `()` is a ZST, `HashMap<K, ()>` stores no value data. The compiler eliminates loads, stores, and allocations for the values. We get a set implementation from a map implementation with no runtime overhead.

The standard library's `HashSet` is implemented exactly this way:

```rust
pub struct HashSet<T, S> {
    map: HashMap<T, (), S>,
}
```

Unsafe code must be careful with ZSTs. Pointer arithmetic on `*const T` where `T` is zero-sized is a no-op: `ptr.add(1)` returns `ptr` unchanged. This breaks the assumption that advancing a pointer produces a different address. Additionally, most allocators do not accept zero-size requests, so `Box::new(ZST)` uses special handling rather than calling the allocator.

One subtle property: references to ZSTs must be non-null and properly aligned (alignment 1 means any address is valid), but dereferencing them is defined to read zero bytes. A reference to a ZST at address 0x1 is valid; a reference at address 0 (null) is undefined behavior even though no actual memory access occurs.

### Empty Types

Zero-sized types can be instantiated. We can create a value of type `()` or `struct Nothing;` and pass it around. Empty types go further: they cannot be instantiated at all.

```rust
enum Void {}
```

An enum with no variants has no valid values. We cannot construct a `Void` because there is no variant to construct. The type exists at the type level but can never exist at the value level.

This enables type-level reasoning about impossibility. Consider a trait for data sources that might fail:

```rust
trait DataSource {
    type Error;
    fn fetch(&self) -> Result<Data, Self::Error>;
}
```

Most implementations have a meaningful error type: network sources fail with I/O errors, parsers fail with syntax errors. But some sources are infallible. An in-memory cache cannot fail to read its own contents:

```rust
enum Infallible {}

struct MemoryCache { data: Data }

impl DataSource for MemoryCache {
    type Error = Infallible;
    fn fetch(&self) -> Result<Data, Infallible> {
        Ok(self.data.clone())
    }
}
```

The `Error = Infallible` communicates at the type level that `fetch` cannot return `Err`. Callers can use an irrefutable pattern:

```rust
fn use_cache(cache: &MemoryCache) {
    let Ok(data) = cache.fetch();  // no Err case to handle
    process(data);
}
```

The compiler optimizes based on this knowledge. `Result<T, Infallible>` has the same layout as `T` because the `Err` variant cannot exist and requires no discriminant:

```rust
use std::mem::size_of;
use std::convert::Infallible;

assert_eq!(size_of::<Result<u64, Infallible>>(), 8);  // same as u64
```

The standard library provides `std::convert::Infallible` for this purpose. It is defined as an empty enum and used throughout the standard library to indicate operations that cannot fail.

Raw pointers to empty types are valid to construct but dereferencing them is undefined behavior. There is no value to read, so the dereference cannot produce a valid result. This makes empty types unsuitable for representing C's `void*`. The recommended approach for opaque C pointers is `*const ()` or a newtype wrapper around it, which can be safely dereferenced to read zero bytes.

### Niche Optimization

An enum must somehow record which variant it currently holds. The naive approach stores a *discriminant*, a small integer that identifies the variant, alongside the variant's data. For an enum with four variants, we need at least 2 bits to distinguish them; in practice, the discriminant occupies 1, 2, or 4 bytes depending on the number of variants and alignment constraints.

Consider `Option<T>`, which has two variants: `Some(T)` and `None`. The naive layout stores a discriminant plus the `T`:

```rust
// Naive Option<u64> layout:
// discriminant: 1 byte (0 = None, 1 = Some)
// padding: 7 bytes (to align u64)
// value: 8 bytes
// total: 16 bytes
```

This is wasteful when `T` is something like `&u64`. A reference is 8 bytes, so `Option<&u64>` would be 16 bytes: 8 for the pointer, plus padding and discriminant overhead. But here is the insight that enables optimization: *a reference can never be null*. Rust guarantees that `&T` always points to a valid `T`. The bit pattern consisting of all zeros, the null pointer, can never represent a valid reference.

The compiler exploits this. For `Option<&T>`, there is no separate discriminant. The `Some` variant stores the pointer as-is. The `None` variant is represented by the null pointer. Pattern matching becomes a null check:

```rust
// Actual Option<&u64> layout:
// pointer: 8 bytes
// total: 8 bytes
// None is represented as null (0x0000000000000000)
// Some(&x) is represented as the address of x
```

This is called *niche optimization*. A *niche* is a bit pattern that a type guarantees it will never hold. The compiler uses niches to encode enum discriminants without allocating additional space.

Which types have niches? Any type that forbids certain bit patterns:

References (`&T`, `&mut T`) forbid null. `NonNull<T>` exists specifically to be a non-null pointer. The `NonZeroU32` type (and its siblings `NonZeroU8`, `NonZeroI64`, etc.) forbid zero. `Box<T>` contains a non-null pointer internally. Function pointers forbid null. `bool` only permits 0 and 1, so 254 other byte values are niches.

The optimization composes. `Option<Box<T>>` uses null for `None`. What about `Option<Option<Box<T>>>`? The outer `Option` needs a niche to represent its `None`, and the inner `Option` already used null for *its* `None`. Can we distinguish outer-`None` from `Some(None)`?

On x86-64 with 48-bit virtual addresses, pointers have constraints beyond non-nullness. The upper 16 bits must be a sign extension of bit 47. Most user-space pointers have the top bits all zeros (canonical lower-half addresses). The compiler can use a non-canonical address like `0x0000000000000001` (misaligned for any type with alignment > 1) to represent additional `None` variants:

```rust
use std::mem::size_of;

assert_eq!(size_of::<Box<i32>>(), 8);
assert_eq!(size_of::<Option<Box<i32>>>(), 8);
assert_eq!(size_of::<Option<Option<Box<i32>>>>(), 8);
```

All three types fit in 8 bytes. The compiler found two niches in the pointer: null for the inner `None`, and another invalid address for the outer `None`.

For types without niches, the discriminant requires additional space:

```rust
use std::mem::size_of;

assert_eq!(size_of::<u64>(), 8);
assert_eq!(size_of::<Option<u64>>(), 16);
```

Every bit pattern is a valid `u64`, so there is no niche. The compiler must store a separate discriminant, and alignment padding expands the total to 16 bytes.

Now we can return to the point we deferred earlier: why does `#[repr(u8)]` on an enum suppress niche optimization? The `repr(u8)` attribute guarantees that the discriminant is stored as an explicit `u8` at a known location. This is a layout guarantee for FFI and binary serialization. If the compiler used niche optimization, there would be no explicit discriminant byte; the variant would be encoded in the payload's bit pattern. These two requirements are incompatible. Explicit discriminant layout and niche optimization are mutually exclusive:

```rust
use std::mem::size_of;

enum WithNiche { Some(Box<i32>), None }

#[repr(u8)]
enum WithoutNiche { Some(Box<i32>), None }

assert_eq!(size_of::<WithNiche>(), 8);      // niche used, no discriminant
assert_eq!(size_of::<WithoutNiche>(), 16);  // explicit u8 discriminant + padding + pointer
```

The `repr(u8)` forces a 1-byte discriminant to exist, which with 7 bytes of padding and the 8-byte pointer yields 16 bytes total.

### Visualizing Layouts

Rust provides `std::mem::size_of::<T>()` and `std::mem::align_of::<T>()` for inspecting type properties at runtime (they are const functions, so compile-time evaluation is also possible). For detailed layout information, the `-Zprint-type-sizes` flag on nightly rustc shows field ordering, padding, and discriminant placement:

```
RUSTFLAGS=-Zprint-type-sizes cargo +nightly build --release
```

For an enum like:

```rust
enum E {
    A,
    B(i32),
    C(u64, u8, u64, u8),
    D(Vec<u32>),
}
```

The output shows:

```
print-type-size type: `E`: 32 bytes, alignment: 8 bytes
print-type-size     discriminant: 1 bytes
print-type-size     variant `D`: 31 bytes
print-type-size         padding: 7 bytes
print-type-size         field `.0`: 24 bytes, alignment: 8 bytes
print-type-size     variant `C`: 23 bytes
print-type-size         field `.1`: 1 bytes
print-type-size         field `.3`: 1 bytes
print-type-size         padding: 5 bytes
print-type-size         field `.0`: 8 bytes, alignment: 8 bytes
print-type-size         field `.2`: 8 bytes
```

The compiler has reordered variant `C`'s fields to place the `u8`s before the padding, minimizing wasted space. The discriminant is only 1 byte despite four variants (2 bits suffice, but alignment constraints mean 1 byte is the minimum addressable unit).

## Fat Pointers and Dynamically Sized Types

We saw how `repr(Rust)` gives the compiler freedom to reorder fields because it knows each field's size and alignment at compile time. But this assumption does not always hold. Some types have sizes that can only be determined at runtime, and the way Rust handles them differs sharply from C++.

### The Problem with C Arrays

In C, when we pass an array to a function, the type system loses information.

```c
void process(int arr[], size_t len);

int main(void) {
    int data[10] = {0};
    process(data, 10);  // must pass length separately
}
```

The parameter `int arr[]` is identical to `int *arr`. The array *decays* to a pointer, and the length vanishes from the type. Nothing in the language connects the pointer to its length. If we pass the wrong length, we get buffer overflows. If we forget to pass it entirely, the function cannot know where the array ends.

This decay happens implicitly. The expression `data` in the call site has type `int[10]`, but by the time it reaches `process`, it is just `int*`. The compiler erases information that the programmer must then track manually.

### Dynamically Sized Types in Rust

Rust solves this with *dynamically sized types* (DSTs), types whose size is not known at compile time. The language has three built-in DST categories.

The slice type `[T]` represents a contiguous sequence of `T` values of unknown length. Unlike `[T; N]`, which has compile-time-known size `N * size_of::<T>()`, the type `[T]` could represent any number of elements. A `[u8]` might be 10 bytes or 10,000 bytes.

The string slice `str` is semantically a `[u8]` with a validity invariant requiring UTF-8 encoding. It shares the same dynamically-sized nature.

Trait objects `dyn Trait` represent values of unknown concrete type that implement `Trait`. A `dyn Display` might be a `String` at 24 bytes, an `i32` at 4 bytes, or a custom type of arbitrary size. The concrete type is erased; only the interface remains.

However, all these types cannot exist directly on the stack or as struct fields (except as the last field). We cannot write `let x: [u8];` because the compiler cannot determine how much stack space to allocate. DSTs exist only behind pointers.

### Wide Pointers

A pointer to a sized type is a single machine word, 8 bytes on x86-64. A pointer to a DST must carry additional information, making it twice as wide.

For slices, a `&[T]` stores a pointer to the first element paired with the number of elements.

```rust
use std::mem::size_of;

assert_eq!(size_of::<&u8>(), 8);        // thin pointer
assert_eq!(size_of::<&[u8]>(), 16);     // wide pointer
assert_eq!(size_of::<&str>(), 16);      // same representation as &[u8]
```

The slice reference carries both pointer and length:

```rust
let arr = [1i32, 2, 3, 4, 5];
let slice: &[i32] = &arr[1..4];

// The slice is (pointer to arr[1], length 3)
let ptr = slice.as_ptr();
let len = slice.len();

assert_eq!(len, 3);
assert_eq!(unsafe { *ptr }, 2);  // first element of slice
```

The wide pointer solves the C problem. When we pass a `&[T]`, the length travels with the pointer. There is no way to separate them, pass the wrong length or forget it.

The coercion from `&[i32; 5]` to `&[i32]` is an *unsizing coercion*. The compiler takes the thin pointer to the array, combines it with the statically-known length, and produces a wide pointer. This happens implicitly at coercion sites, including `let` bindings with explicit types, function arguments, and return values.

### Trait Object Representation

For trait objects, the wide pointer contains different metadata. A `&dyn Trait` stores a pointer to the concrete value paired with a pointer to a *vtable* (virtual method table).

```rust
use std::mem::size_of;

trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}

impl Drawable for i32 {
    fn draw(&self) { println!("{}", self); }
    fn area(&self) -> f64 { 0.0 }
}

assert_eq!(size_of::<&i32>(), 8);
assert_eq!(size_of::<&dyn Drawable>(), 16);
```

The vtable is a static data structure generated by the compiler for each (concrete type, trait) pair. It contains function pointers for every method in the trait, plus metadata required for memory management.

For a trait `Drawable` with methods `draw` and `area`, the vtable looks roughly like this:

```
+0:   drop_in_place::<ConcreteType>
+8:   size_of::<ConcreteType>
+16:  align_of::<ConcreteType>
+24:  Drawable::draw for ConcreteType
+32:  Drawable::area for ConcreteType
```

The size and alignment entries are essential for dropping boxed trait objects. When we call `drop` on a `Box<dyn Drawable>`, the runtime must know how many bytes to deallocate and what alignment the allocator expects.

### Virtual Dispatch in Assembly

When we call a method on a trait object, the compiler generates an indirect call through the vtable. Consider this function:

```rust
fn call_draw(obj: &dyn Drawable) {
    obj.draw();
}
```

On x86-64, this compiles to something like:

```asm
call_draw:
    ; rdi = data pointer (obj.data)
    ; rsi = vtable pointer (obj.vtable)
    mov     rax, [rsi + 24]    ; load draw function pointer from vtable
    mov     rdi, rdi           ; data pointer becomes first argument (self)
    jmp     rax                ; tail call to draw implementation
```

The vtable lookup adds one memory indirection compared to a direct call. More significantly, the indirect call through a register prevents the CPU from predicting the branch target until the vtable load completes. Modern CPUs have indirect branch predictors, but they are less effective than direct branch prediction.

Compare this to a generic function with static dispatch:

```rust
fn call_draw_static<T: Drawable>(obj: &T) {
    obj.draw();
}
```

Here the compiler monomorphizes the function for each concrete `T`, producing a direct call:

```asm
call_draw_static_for_i32:
    jmp     <i32 as Drawable>::draw
```

No vtable lookup, no indirect branch. The call target is known at compile time, enabling inlining. If `draw` is small, the compiler can inline it entirely, eliminating the call overhead and enabling further optimizations across the inlined code.

### Performance Implications of Dynamic Dispatch

The overhead of virtual dispatch is not just the vtable lookup. The indirect call has cascading effects on the CPU pipeline.

Modern CPUs predict branch targets to keep the instruction pipeline full. For direct calls, the target is encoded in the instruction itself, and the predictor can fetch the target instructions immediately. For indirect calls through a register, the CPU must wait for the register value to be computed (the vtable load), then use an indirect branch predictor to guess the target. Indirect branch predictors maintain a table of recent indirect branch targets, but they are less accurate than direct branch prediction, especially when a call site dispatches to many different implementations.

When the predictor guesses wrong, the pipeline must be flushed and restarted from the correct target. On modern x86-64, this costs roughly 15-20 cycles. If a tight loop calls a trait object method that alternates between two implementations, the predictor may never stabilize, incurring the misprediction penalty on every other iteration.

The vtable itself must be in cache for the lookup to be fast. A vtable is small (typically 32-64 bytes for a trait with a few methods), but if we iterate over a heterogeneous collection of trait objects, each object may point to a different vtable. With many distinct implementations, the vtables compete for cache space. The first access to each vtable is a cache miss, adding 100+ cycles of memory latency.

Inlining is the most significant loss. When the compiler inlines a function, it can see both caller and callee code simultaneously. This enables constant propagation, dead code elimination, loop fusion, and SIMD vectorization across the boundary. In the general case, none of this is possible through a vtable; the compiler cannot see through the indirection, so each call becomes an optimization barrier.

In C++, modern compilers can sometimes *devirtualize* virtual calls. If the concrete type is visible at the call site (immediately after construction, or when the class is marked `final`), Clang/GCC may replace the indirect call with a direct one. With LTO and `-fwhole-program-vtables`, C++ compilers can devirtualize when only one implementation exists program-wide.

Rust's situation is less favorable. As of 2025, rustc does not provide the metadata LLVM needs for whole-program devirtualization. Even when only a single type implements a trait in the entire binary, trait object calls remain indirect. LLVM can devirtualize in trivial cases where the concrete type is immediately visible (creating a trait object and calling a method in the same basic block), but this is rare in practice. The tracking issue [rust-lang/rust#68262](https://github.com/rust-lang/rust/issues/68262) has seen little progress. For now, if we need devirtualization in Rust, we can use generics or enums, the compiler will not rescue trait objects for us.

Consider a loop summing areas. Here we must be careful: the static and dynamic versions solve *different* problems.

```rust
// Static dispatch: all elements must be the same concrete type
fn sum_areas_static<T: Drawable>(shapes: &[T]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// Dynamic dispatch: elements can be different concrete types
fn sum_areas_dynamic(shapes: &[&dyn Drawable]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}
```

The static version requires a homogeneous slice, all `Circle`s or all `Rectangle`s, never mixed. The dynamic version accepts heterogeneous collections. These are not interchangeable; we choose based on whether we need polymorphism over a closed or open set of types.

For the homogeneous case, the compiler can inline the entire loop body, unroll the loop, and potentially vectorize with SIMD. For the heterogeneous case, each `area()` call is a function pointer load, an indirect call, and a return. The loop cannot be vectorized because the compiler cannot prove anything about what `area()` does.

When the set of types is closed and known at compile time, Rust offers a third approach that combines the benefits of both: enum dispatch.

```rust
enum Shape {
    Circle(Circle),
    Rectangle(Rectangle),
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            Shape::Circle(c) => c.area(),
            Shape::Rectangle(r) => r.area(),
        }
    }
}

fn sum_areas_enum(shapes: &[Shape]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}
```

The enum approach accepts a heterogeneous collection (circles and rectangles mixed), but the dispatch is a compile-time match expression rather than a vtable lookup. The compiler can inline each branch, and modern CPUs predict `match` arms more reliably than indirect calls. The slice is homogeneous at the type level (`&[Shape]`), so it is cache-friendly, no pointer chasing, no scattered vtables.

The trade-off is extensibility. Adding a new shape to an enum requires modifying the enum definition and every `match`. With trait objects, new types can implement the trait without touching existing code. Enums are closed; traits are open.

The rule of thumb is to use generics with trait bounds for performance-critical code paths where all elements share a concrete type. Use enums when the type set is closed and we need heterogeneous collections with predictable dispatch. Reserve trait objects for open extensibility where the flexibility is worth the cost, or for reducing compile times and binary size when performance is not critical.

### The Vtable Location Trade-off

Rust and C++ made opposite design decisions about where to store the vtable pointer.

In C++, polymorphic objects embed a *vptr* directly in the object:

```cpp
class Drawable {
public:
    virtual void draw() = 0;
    virtual double area() = 0;
    virtual ~Drawable() = default;
    int x;
};

// Memory layout of a Drawable subclass instance:
// +0:  vptr (8 bytes, points to vtable)
// +8:  x (4 bytes)
// +12: padding (4 bytes)
// Total: 16 bytes
```

Every instance of a polymorphic class carries the vptr. A `Drawable*` or `Drawable&` is 8 bytes, but each object is 8 bytes larger than it would be without virtual functions.

Rust places the vtable pointer in the reference, not the object:

```rust
struct Circle {
    radius: f64,
}

impl Drawable for Circle {
    fn draw(&self) { /* ... */ }
    fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius }
}

// Memory layout of Circle:
// +0: radius (8 bytes)
// Total: 8 bytes (no vptr)

// Memory layout of &dyn Drawable pointing to Circle:
// +0: data pointer (8 bytes)
// +8: vtable pointer (8 bytes)
// Total: 16 bytes
```

The trade-off is memory allocation versus reference size. With a million objects and one reference each, C++ uses 8 MB for vptrs embedded in objects; Rust uses 8 MB for vtable pointers in references. With a million objects and ten references each, C++ still uses 8 MB; Rust uses 80 MB.

But Rust's design enables something C++ cannot do. The same object can be viewed through multiple trait lenses simultaneously:

```rust
use std::fmt::Debug;

let circle = Circle { radius: 1.0 };

let drawable: &dyn Drawable = &circle;
let debug: &dyn Debug = &circle;
let any: &dyn std::any::Any = &circle;
```

Each reference carries its own vtable pointer, pointing to different vtables. The `circle` object itself is unchanged. In C++, the vptr embedded in the object determines its dynamic type at construction time. A `Drawable*` and a `Base*` pointing to the same object use the same embedded vptr (possibly adjusted for multiple inheritance).

### Dyn Compatibility

Not every trait can be used as a trait object. The rules for *dyn compatibility* (formerly called object safety) restrict which traits work with dynamic dispatch.

A dyn-compatible trait must not require `Self: Sized`. Requiring `Sized` would contradict the purpose of trait objects, which exist to handle values of unknown size. The trait must have no associated constants, since constants require knowing the concrete type at compile time. It must have no associated types with generic parameters, which would require instantiation at compile time.

All methods must be dispatchable or explicitly non-dispatchable. A dispatchable method must have a receiver (`&self`, `&mut self`, `Box<Self>`, etc.), must not return `Self` by value, must not take `Self` by value as a parameter, and must not have type parameters.

```rust
// Dyn compatible
trait Compatible {
    fn method(&self);
    fn returns_ref(&self) -> &str;
}

// NOT dyn compatible
trait Incompatible {
    fn returns_self(&self) -> Self;        // Self in return position
    fn takes_self(&self, other: Self);     // Self as parameter
    fn generic<T>(&self, x: T);            // type parameter
    const VALUE: i32;                       // associated constant
}
```

Dynamic dispatch requires a vtable entry for each method, and a vtable entry is a function pointer with a fixed signature. If a method returns `Self`, the return type depends on the concrete type, which is erased. If a method is generic over `T`, there would need to be infinitely many vtable entries, one for each `T`.

Methods that violate these rules can still exist on dyn-compatible traits if they have a `where Self: Sized` bound, making them unavailable through trait objects but callable on concrete types:

```rust
trait PartiallyCompatible {
    fn dispatchable(&self);

    fn not_dispatchable(&self) -> Self where Self: Sized;
}

// This works:
let obj: &dyn PartiallyCompatible = &some_value;
obj.dispatchable();

// This does not compile:
// obj.not_dispatchable();
```

### Auto Traits and Lifetime Bounds

Trait objects can include auto traits and lifetime bounds. Unlike regular traits, where only one non-auto trait is allowed, auto traits can be added freely:

```rust
use std::fmt::Debug;

// All valid trait object types:
fn takes_debug(x: &dyn Debug) {}
fn takes_debug_send(x: &dyn Debug + Send) {}
fn takes_debug_send_sync(x: &(dyn Debug + Send + Sync)) {}
fn takes_debug_static(x: &(dyn Debug + 'static)) {}
```

The auto traits (`Send`, `Sync`, `Unpin`, etc.) do not add methods to the vtable. They are marker traits checked at compile time when the trait object is created. If we try to create a `&dyn Debug + Send` from a type that is not `Send`, the compiler rejects it:

```rust
use std::rc::Rc;
use std::fmt::Debug;

let rc: Rc<i32> = Rc::new(42);
// Error: Rc<i32> is not Send
// let obj: &(dyn Debug + Send) = &*rc;
```

Lifetime bounds constrain how long references inside the trait object can live. A `dyn Trait + 'static` contains no non-`'static` references. A `dyn Trait + 'a` may contain references that live at least as long as `'a`. The default lifetime depends on context and follows elision rules.

### Supertraits in the Vtable

When a trait has supertraits, the vtable includes method entries for the supertrait methods:

```rust
trait Shape {
    fn area(&self) -> f64;
}

trait Circle: Shape {
    fn radius(&self) -> f64;
}
```

The vtable for `dyn Circle` contains entries for both `area` and `radius`. When we call a supertrait method on a trait object, it goes through the same vtable lookup:

```rust
fn print_area(c: &dyn Circle) {
    // This works because Shape::area is in the Circle vtable
    println!("Area: {}", c.area());
}
```

Trait object upcasting allows converting `&dyn Circle` to `&dyn Shape`. The compiler generates a different vtable for `dyn Shape` that contains only `area`, and the conversion substitutes the vtable pointer:

```rust
fn use_as_shape(c: &dyn Circle) {
    let shape: &dyn Shape = c;  // upcasting coercion
    println!("Area: {}", shape.area());
}
```

### The ?Sized Bound

By default, all type parameters have an implicit `Sized` bound:

```rust
fn foo<T>(x: T) { }
// is equivalent to:
fn foo<T: Sized>(x: T) { }
```

This makes sense. To pass `x` by value, the compiler must know its size to copy it onto the stack. But sometimes we want to accept DSTs through references. The `?Sized` bound relaxes the `Sized` requirement:

```rust
fn process<T: ?Sized>(x: &T) {
    // T might be a DST
    // x is a wide pointer if T is unsized
}

// Now this works:
process::<[u8]>(&[1, 2, 3][..]);
process::<str>("hello");
process::<dyn std::fmt::Debug>(&42);
```

The standard library uses `?Sized` extensively. The signature of `std::borrow::Borrow` is:

```rust
pub trait Borrow<Borrowed: ?Sized> {
    fn borrow(&self) -> &Borrowed;
}
```

This allows `String` to implement `Borrow<str>`, returning `&str`, a reference to a DST.

In trait definitions, `Self` has an implicit `Self: ?Sized` bound, the opposite of type parameters. This allows traits to be implemented for DSTs:

```rust
trait MyTrait {
    fn method(&self);
}

impl MyTrait for str {
    fn method(&self) {
        println!("length: {}", self.len());
    }
}
```

If traits required `Self: Sized` by default, we could not implement traits for `str`, `[T]`, or `dyn OtherTrait`.

### Custom DSTs

A struct can contain a DST as its last field, making the struct itself a DST:

```rust
struct MySlice {
    header: u32,
    data: [u8],  // DST as last field
}
```

`MySlice` is now a DST. We cannot create it directly on the stack. The only supported construction method is through unsizing coercion from a sized variant:

```rust
struct MySized<const N: usize> {
    header: u32,
    data: [u8; N],
}

fn main() {
    let sized = MySized::<4> { header: 42, data: [1, 2, 3, 4] };
    let dynamic: &MySlice = unsafe {
        // Requires ptr::from_raw_parts or careful transmutation
        std::mem::transmute::<&MySized<4>, &MySlice>(&sized)
    };

    assert_eq!(dynamic.header, 42);
    assert_eq!(dynamic.data.len(), 4);
}
```

This is awkward and requires unsafe code. The `ptr::from_raw_parts` API (stabilized in Rust 1.79) provides a safer way to construct custom DST pointers, but the ergonomics remain poor. Most code uses the built-in DST types (`[T]`, `str`, `dyn Trait`) rather than defining custom ones.

To construct a DST, we must provide the metadata (length or vtable pointer) that completes the type. But the language provides no syntax for this at the value level. The standard library handles DST construction internally through careful unsafe code and compiler magic for types like `str` and `[T]`.

### Invalid Metadata is Undefined Behavior

The metadata in a wide pointer is not merely informational. The compiler trusts it for safety-critical operations. Providing invalid metadata is undefined behavior.

For slices, the length must not cause the slice to extend beyond its allocation:

```rust
let arr = [1, 2, 3];
let ptr = arr.as_ptr();
// UB: length 1000 exceeds the allocation
let bad_slice: &[i32] = unsafe {
    std::slice::from_raw_parts(ptr, 1000)
};
```

For trait objects, the vtable must be a valid vtable for the trait that matches the actual dynamic type of the pointed-to object:

```rust
// UB: null is not a valid vtable
let data_ptr: *const () = &42i32 as *const i32 as *const ();
let vtable_ptr: *const () = std::ptr::null();
let bad: &dyn std::fmt::Debug = unsafe {
    std::mem::transmute((data_ptr, vtable_ptr))
};
```

The Rust Reference explicitly lists invalid wide pointer metadata as undefined behavior. Calling a method through a corrupted vtable pointer could jump to arbitrary code. Using an invalid slice length could read or write out of bounds.

## Polymorphism: Monomorphization vs Vtables

We saw how Rust represents pointers to dynamically sized types. A `&dyn Draw` carries both a data pointer and a vtable pointer, 16 bytes that enable runtime method dispatch. But this raises a question we have not yet answered: why does Rust need two polymorphism mechanisms at all? Templates and generics already let us write code that works across types. Why introduce vtables?

When we write a function that operates on "any type implementing trait X," the compiler must decide how to generate code for it. Two strategies exist. The compiler can stamp out a separate copy of the function for each concrete type that actually gets used, a process called *monomorphization*. Alternatively, the compiler can generate a single copy of the function that dispatches method calls through a table of function pointers at runtime. Both C++ and Rust support both strategies. C, lacking native generics, provides only workarounds that approximate each approach with varying degrees of type safety.

Monomorphization eliminates runtime indirection entirely; every call is direct, every function body can be inlined, the optimizer sees through abstraction boundaries. But the binary contains a separate copy of the generic code for every type instantiation, which bloats both compile time and binary size. Dynamic dispatch through vtables produces a single copy of the code regardless of how many implementing types exist, but every method call requires loading a function pointer from memory and jumping through it, which the CPU cannot predict well and which prevents inlining.

### C Without Generics

C has no built-in parametric polymorphism. We cannot write a function that operates on "any comparable type" and have the compiler generate type-safe, specialized versions. Historically, C programmers used three workarounds.

Preprocessor macros perform textual substitution before the compiler ever sees the code:

```c
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int x = MAX(3, 5);           // expands to ((3) > (5) ? (3) : (5))
double y = MAX(1.5, 2.5);    // expands to ((1.5) > (2.5) ? (1.5) : (2.5))
```

Each use site expands to type-specific code, achieving something like monomorphization. But macros operate outside the type system. The preprocessor has no concept of what `a` and `b` are; it pastes text. This leads to the classic double-evaluation trap:

```c
#define SQUARE(x) ((x) * (x))
int a = 5;
int b = SQUARE(a++);  // expands to ((a++) * (a++)), undefined behavior
```

The argument is evaluated twice. If it has side effects, the program's behavior becomes undefined. The macro cannot evaluate its argument once and bind it to a local; macros do not have local variables.

C11 introduced `_Generic`, which provides compile-time type dispatch:

```c
#define abs(x) _Generic((x),    \
    int: abs_int,               \
    long: abs_long,             \
    double: fabs,               \
    default: abs_int)(x)

int abs_int(int x) { return x < 0 ? -x : x; }
long abs_long(long x) { return x < 0 ? -x : x; }
```

The `_Generic` keyword examines the type of its first argument and selects the corresponding expression. This is better than macros: the selection happens within the type system, and each branch is a proper function that evaluates its argument once. But we must enumerate every supported type explicitly and write separate implementations for each. We have not reduced code duplication; we have centralized dispatch.

For dynamic polymorphism, we can use function pointers with `void*`:

```c
typedef int (*comparator)(const void*, const void*);

void qsort(void* base, size_t nmemb, size_t size, comparator cmp);

int compare_int(const void* a, const void* b) {
    return *(const int*)a - *(const int*)b;
}

int arr[] = {5, 2, 8, 1};
qsort(arr, 4, sizeof(int), compare_int);
```

The standard library's `qsort` treats the array as raw bytes and accepts a function pointer for comparison. The type information is erased: the comparator receives `void*` and must cast internally. Nothing prevents passing `compare_int` to sort an array of `double`. The compiler cannot verify correctness. If the programmer gets it wrong, the program silently produces garbage or crashes.

### C++ Templates

C++ templates let us define families of functions and classes parameterized over types:

```cpp
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

int x = max(3, 5);           // instantiates max<int>
double y = max(1.5, 2.5);    // instantiates max<double>
```

When the compiler encounters `max(3, 5)`, it deduces `T = int` and generates a specialized function `max<int>`. A separate `max<double>` gets generated for the second call. Each instantiation is compiled independently, producing code identical to what we would write by hand. There is no runtime overhead.

Templates use what is sometimes called *duck typing*: instantiation succeeds when the operations used in the template body are valid for the concrete type; otherwise, the compiler emits an error. The problem is that errors emerge from deep within the template instantiation, often producing notoriously verbose diagnostics that obscure the root cause. The template's requirements are implicit; we discover at instantiation time whether a type satisfies them.

This implicit checking enables SFINAE (Substitution Failure Is Not An Error), a mechanism where invalid template substitutions silently remove candidates from the overload set rather than causing hard errors. Before C++20, constraining templates required arcane metaprogramming with `std::enable_if` and type traits:

```cpp
#include <type_traits>

template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
absolute(T x) {
    return x < 0 ? -x : x;
}
```

The `enable_if` machinery conditionally makes the return type valid or invalid depending on the type trait. If `T` is not integral, the substitution fails, and this overload is removed from consideration. The code works but is dense with machinery that obscures intent.

C++20 introduced concepts, which make constraints explicit:

```cpp
#include <concepts>

template<typename T>
concept Comparable = requires(T a, T b) {
    { a < b } -> std::convertible_to<bool>;
    { a > b } -> std::convertible_to<bool>;
};

template<Comparable T>
T max(T a, T b) {
    return (a > b) ? a : b;
}
```

The `Comparable` concept declares what operations `T` must support. The template states its constraint explicitly in the signature. If we instantiate `max` with a non-comparable type, the error message refers directly to the violated concept rather than to some failed substitution buried in the template body.

Regardless of how constraints are expressed, templates monomorphize. For large templates like `std::sort` or `std::unordered_map`, binary bloat becomes pretty significant. We can mitigate this effect with explicit instantiation, where we declare in a single translation unit which instantiations to generate:

```cpp
// header
template<typename T>
void process(T x);

// source
template<typename T>
void process(T x) { /* implementation */ }

template void process<int>(int);
template void process<double>(double);
```

Other translation units can use `process<int>` without triggering instantiation; they link against the pre-generated code. This reduces compile time and binary size at the cost of flexibility.

### C++ Virtual Functions

C++ supports runtime polymorphism through virtual functions. A class with at least one virtual function is *polymorphic*:

```cpp
class Shape {
public:
    virtual double area() const = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
    double radius;
public:
    Circle(double r) : radius(r) {}
    double area() const override { return 3.14159 * radius * radius; }
};

class Rectangle : public Shape {
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const override { return width * height; }
};
```

When we call a virtual function through a base class pointer or reference, the actual function invoked depends on the dynamic type of the object:

```cpp
void print_area(const Shape& s) {
    std::cout << s.area() << "\n";  // virtual dispatch
}

Circle c(1.0);
Rectangle r(2.0, 3.0);
print_area(c);  // calls Circle::area
print_area(r);  // calls Rectangle::area
```

The compiler cannot know at compile time which implementation to call. The decision is deferred to runtime.

To implement this, the compiler inserts a hidden pointer (the *vptr*) into every object of a polymorphic class. The vptr points to a static table (the *vtable*) shared by all objects of the same dynamic type. The Itanium C++ ABI, used by GCC, Clang, and most non-Windows compilers, specifies the vtable layout precisely.

The vtable contains several components, laid out at specific offsets from an *address point*. Components before the address point (at negative offsets) include virtual call offsets for adjusting `this` pointers in multiple inheritance scenarios, virtual base offsets for locating virtual base subobjects, and the *offset-to-top*, a `ptrdiff_t` giving the displacement from this vtable pointer location to the top of the complete object (used by `dynamic_cast<void*>`). At the address point sits the RTTI pointer, pointing to type information for runtime type identification. After the address point come the virtual function pointers themselves, in declaration order.

For a simple class hierarchy without multiple inheritance, the layout simplifies. A `Circle` object in memory looks like:

```
Circle object (16 bytes on x86-64):
+0:   vptr (8 bytes, points to Circle's vtable)
+8:   radius (8 bytes, double)

Circle's vtable:
-16:  offset-to-top (0)
-8:   RTTI pointer
 0:   &Circle::~Circle() (complete destructor)
+8:   &Circle::~Circle() (deleting destructor)
+16:  &Circle::area()
```

The vptr in every `Circle` instance points to offset 0 of this vtable, the address point. When we call `s.area()` where `s` is a `Shape&`, the compiler generates:

```asm
; rdi = pointer to Shape object
mov     rax, [rdi]          ; load vptr from object
mov     rax, [rax + 16]     ; load area() pointer from vtable
call    rax                 ; indirect call
```

Two memory loads occur on every virtual call: one to fetch the vptr from the object, one to fetch the function pointer from the vtable. More significantly, the `call rax` is an indirect branch. The CPU's branch predictor must guess the target without knowing it until the register value is computed. If a call site invokes many different implementations (iterating over a heterogeneous container of shapes), the predictor may thrash, causing pipeline stalls.

The optimizer cannot inline through a virtual call. It does not know which function will be invoked, so it cannot substitute the function body at the call site. This blocks constant propagation, dead code elimination, and other interprocedural optimizations.

The advantage is code size. There is exactly one `print_area` function, regardless of how many shape types exist. The vtable adds per-class overhead, not per-use overhead. For large class hierarchies, this can dramatically reduce binary size compared to templated alternatives.

### Rust Generics

Rust generics follow the monomorphization strategy, like C++ templates, but with a critical difference: constraints are declared upfront.

```rust
fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

let x = max(3, 5);           // instantiates max::<i32>
let y = max(1.5, 2.5);       // instantiates max::<f64>
```

The bound `T: PartialOrd` states that `T` must implement the `PartialOrd` trait. The compiler checks this at the call site: calling `max` with a type that does not implement `PartialOrd` produces an error that directly states the unsatisfied bound.

Inside the function body, we can only use operations that `PartialOrd` guarantees. Attempting to call methods not provided by the bound fails immediately, not at instantiation:

```rust
fn broken<T: PartialOrd>(a: T, b: T) -> T {
    println!("{}", a);  // error: T doesn't implement Display
    if a > b { a } else { b }
}
```

Rust checks the generic function against its declared bounds before instantiation. This differs from C++ templates, where the body is tentatively compiled against each concrete type, with errors emerging during instantiation.

The monomorphization process itself works similarly. When the Rust compiler encounters generic function calls, it records the concrete types. During code generation, the *monomorphization collector* traverses the call graph to identify all required instantiations. Each generic function paired with each set of concrete type arguments becomes a distinct *mono item* that gets compiled to machine code.

The collector partitions mono items into *Codegen Units* (CGUs). For incremental compilation, the partitioner creates separate CGUs for stable non-generic code and for monomorphized instances. When only generic instantiations change, the stable CGU can be reused.

The same binary size concerns apply. A generic function used with many types produces many copies. The `cargo llvm-lines` tool shows which functions contribute most to generated LLVM IR. In large codebases, common utility functions like `Option::map` and `Result::map_err` can get instantiated hundreds of times, dominating code size.

The standard mitigation is the *inner function pattern*: move the bulk of the logic into a non-generic inner function, leaving a thin generic wrapper:

```rust
pub fn read<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    fn inner(path: &Path) -> io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut bytes = Vec::with_capacity(size as usize);
        io::default_read_to_end(&mut file, &mut bytes)?;
        Ok(bytes)
    }
    inner(path.as_ref())
}
```

The outer generic function calls `as_ref()` to convert `P` to `&Path`, then delegates to `inner`. Now `inner` compiles once regardless of how many path types are used. The outer wrapper is tiny, its monomorphization overhead minimal.

### Rust Trait Objects

When monomorphization costs are prohibitive, Rust offers trait objects. We already covered the representation: `&dyn Trait` is a wide pointer containing a data pointer and a vtable pointer. Calling a method loads the function pointer from the vtable and invokes it indirectly.

The key difference from C++ is where the vtable pointer lives. In C++, the vptr is embedded in the object. A `Circle` contains its own vptr; the `Circle` type is inherently polymorphic. In Rust, the vtable pointer is in the reference, not the object. A plain `Circle` struct has no vptr. The vtable pointer appears only when we view the `Circle` through a trait object:

```rust
let c = Circle { radius: 1.0 };
let shape: &dyn Shape = &c;  // wide pointer created here
```

The `Circle` struct occupies only the bytes its fields require. The 8-byte vtable pointer is added when we create the `&dyn Shape`, not when we create the `Circle`. An object can be viewed through multiple different trait objects, each with its own vtable, without the object itself containing any vtable pointers.

This design means Rust structs remain trivially copyable (if their fields are) even when they implement traits with virtual methods. In C++, adding a single virtual function to a class makes it non-trivially copyable and increases its size by the vptr. In Rust, implementing a trait never changes a struct's layout.

### Choosing Between Strategies

Static dispatch through monomorphization is appropriate when the hot path demands maximum performance, when the set of types is small and known, when inlining across the generic boundary matters, and when compile time and binary size are acceptable costs.

Dynamic dispatch through vtables is appropriate when the concrete type cannot be known until runtime (plugin systems, user-defined types loaded dynamically), when binary size is constrained, when compile time must be minimized, and when the call overhead is negligible compared to the work the method performs.

There are common patterns that combine both. For example, we can expose a generic public API and then convert to a trait object internally to avoid instantiation explosion:

```rust
pub fn process<W: Write>(writer: W) {
    process_dyn(&mut writer as &mut dyn Write)
}

fn process_dyn(writer: &mut dyn Write) {
    // large implementation, compiled once
}
```

The public API accepts any `Write` implementor. Internally, we immediately convert to a trait object. `process_dyn` compiles only once. The cost is one virtual dispatch per method call within `process_dyn`, but the binary contains only one copy of the implementation.

### Under the Hood

Consider incrementing a counter through both dispatch strategies. Let's start with static dispatch:

```rust
trait Counter {
    fn increment(&mut self);
}

struct Simple(u64);
impl Counter for Simple {
    fn increment(&mut self) { self.0 += 1; }
}

fn inc_static<T: Counter>(c: &mut T) {
    c.increment();
}
```

For `inc_static::<Simple>`, monomorphization produces:

```asm
inc_static_Simple:
    add     qword ptr [rdi], 1
    ret
```

The entire method call is inlined to a single `add` instruction. The trait abstraction has zero runtime cost.

Now consider dynamic dispatch:

```rust
fn inc_dynamic(c: &mut dyn Counter) {
    c.increment();
}
```

The compiler generates:

```asm
inc_dynamic:
    mov     rax, [rsi + 24]    ; load increment from vtable
    mov     rdi, rdi           ; data pointer (already in rdi)
    jmp     rax                ; tail call through vtable
```

The function loads the method pointer from the vtable and jumps to it. The actual increment happens in the target function, which cannot be inlined here. For a single increment, the difference is trivial. In a tight loop incrementing millions of times, the static version avoids the vtable load and indirect branch on every iteration.

The static version also enables further optimization. If the compiler can prove the counter is never observed between increments, it can batch them. If the counter value is known, it can constant-fold. None of this is possible through the vtable indirection.

## Closures and Captures

We have seen two strategies for polymorphism: monomorphization produces specialized code for each concrete type, while vtables enable a single function to operate on values of unknown type through indirect dispatch. Both mechanisms deal with *code* that varies. But what happens when we need a function that carries *state*?

Consider a sorting function that accepts a comparison predicate. The predicate must know *how* to compare, which is pure code. But suppose we want to sort by distance from some reference point. Now the predicate needs access to the reference point's coordinates, data that exists outside the function itself. The predicate is no longer pure code; it is code plus environment.

This is the closure problem. A closure *closes over* variables from its enclosing scope, capturing them for later use. The three languages approach this problem with characteristic differences. C lacks closures entirely and requires manual workarounds. C++ introduced lambda expressions that desugar to anonymous structs with an overloaded call operator. Rust closures work similarly but integrate with the ownership system, with the `Fn`, `FnMut`, and `FnOnce` traits encoding how the closure interacts with its captured state.

### C: Function Pointers and the Context Pattern

C has function pointers, not closures. A function pointer is an address of executable code; it contains no data beyond the address itself.

```c
int compare_ints(const void *a, const void *b) {
    int x = *(const int*)a;
    int y = *(const int*)b;
    return (x > y) - (x < y);
}

qsort(array, n, sizeof(int), compare_ints);
```

This works when comparison needs no external state. When it does, C libraries adopt a convention: pass a `void*` context alongside the function pointer, and the callback receives this context as an additional argument.

```c
struct DistanceContext {
    double ref_x, ref_y;
};

int compare_by_distance(const void *a, const void *b, void *ctx) {
    const struct Point *pa = a;
    const struct Point *pb = b;
    const struct DistanceContext *c = ctx;

    double da = hypot(pa->x - c->ref_x, pa->y - c->ref_y);
    double db = hypot(pb->x - c->ref_x, pb->y - c->ref_y);
    return (da > db) - (da < db);
}

// Usage requires a sorting function that accepts context
struct DistanceContext ctx = { .ref_x = 0.0, .ref_y = 0.0 };
qsort_r(points, n, sizeof(struct Point), compare_by_distance, &ctx);
```

The `qsort_r` variant (POSIX, not standard C) threads the context through to the comparator. The pattern is universal in C callback APIs: a function pointer paired with a `void*` that the library passes back untouched.

The `void*` erases type information; nothing prevents us from passing a `DistanceContext*` to a callback expecting something else. The compiler cannot verify that the context pointer remains valid when the callback executes. If the callback outlives the context's stack frame, we have a dangling pointer. The burden falls entirely on us.

### C++ Lambdas

C++11 introduced lambda expressions, syntactic sugar for anonymous function objects. A lambda like

```cpp
auto ref_x = 0.0, ref_y = 0.0;
auto compare = [ref_x, ref_y](const Point& a, const Point& b) {
    double da = std::hypot(a.x - ref_x, a.y - ref_y);
    double db = std::hypot(b.x - ref_x, b.y - ref_y);
    return da < db;
};
```

desugars to something equivalent to

```cpp
struct __lambda_1 {
    double ref_x;
    double ref_y;

    bool operator()(const Point& a, const Point& b) const {
        double da = std::hypot(a.x - ref_x, a.y - ref_y);
        double db = std::hypot(b.x - ref_x, b.y - ref_y);
        return da < db;
    }
};

__lambda_1 compare{ref_x, ref_y};
```

The capture list specifies what enters the closure and how. `[x]` captures `x` by value (copies it into the struct). `[&x]` captures by reference (the struct holds a reference). `[=]` captures everything used by value. `[&]` captures everything by reference. `[x, &y]` mixes modes.

Each lambda has a unique anonymous type that we cannot name. We must use `auto` for local variables or templates for function parameters:

```cpp
template<typename F>
void use_callback(F&& f) {
    f();
}
```

Alternatively, `std::function<R(Args...)>` provides type erasure, wrapping any callable with a matching signature into a uniform type at the cost of heap allocation and virtual dispatch.

The capture mode determines the closure's size. A lambda capturing two `double`s by value occupies 16 bytes (plus alignment). A lambda capturing by reference stores pointers, 8 bytes each on x86-64. A lambda capturing nothing is stateless; the standard guarantees that captureless lambdas can convert to plain function pointers:

```cpp
int (*fp)(int, int) = [](int a, int b) { return a + b; };
```

C++ lambdas are mutable by default if they capture by value with `mutable`:

```cpp
int counter = 0;
auto increment = [counter]() mutable { return ++counter; };
// Each call modifies the lambda's internal copy of counter
```

Without `mutable`, the call operator is `const` and cannot modify captured values. The default reflects the common case where closures are passed to algorithms and called multiple times; mutating captured state would be surprising.

### Rust Closures

Rust closures follow the same structural principle: a closure is an anonymous struct containing captured values, with a method implementing the call. But the details differ in ways that matter for safety.

```rust
let ref_x = 0.0_f64;
let ref_y = 0.0_f64;
let compare = |a: &Point, b: &Point| {
    let da = ((a.x - ref_x).powi(2) + (a.y - ref_y).powi(2)).sqrt();
    let db = ((b.x - ref_x).powi(2) + (b.y - ref_y).powi(2)).sqrt();
    da.partial_cmp(&db).unwrap()
};
```

The compiler generates a struct like

```rust
struct __closure_1<'a> {
    ref_x: &'a f64,
    ref_y: &'a f64,
}

impl<'a> FnOnce<(&Point, &Point)> for __closure_1<'a> {
    type Output = std::cmp::Ordering;
    extern "rust-call" fn call_once(self, args: (&Point, &Point)) -> Self::Output {
        let (a, b) = args;
        let da = ((a.x - *self.ref_x).powi(2) + (a.y - *self.ref_y).powi(2)).sqrt();
        let db = ((b.x - *self.ref_x).powi(2) + (b.y - *self.ref_y).powi(2)).sqrt();
        da.partial_cmp(&db).unwrap()
    }
}

impl<'a> FnMut<(&Point, &Point)> for __closure_1<'a> {
    extern "rust-call" fn call_mut(&mut self, args: (&Point, &Point)) -> Self::Output {
        self.call_once(args)
    }
}

impl<'a> Fn<(&Point, &Point)> for __closure_1<'a> {
    extern "rust-call" fn call(&self, args: (&Point, &Point)) -> Self::Output {
        self.call_once(args)
    }
}
```

Unlike C++, Rust does not require explicit capture annotations. The compiler infers the capture mode from how variables are used in the closure body. If we only read a variable, it captures by shared reference. If we mutate it, it captures by mutable reference. If we move out of it or if the variable does not implement `Copy`, it captures by value.

```rust
let s = String::from("hello");

// Captures s by shared reference (only reads)
let c1 = || println!("{}", s);

// Captures s by mutable reference (mutates)
let mut s = String::from("hello");
let c2 = || s.push_str(" world");

// Captures s by value (moves out)
let s = String::from("hello");
let c3 = || drop(s);
```

The `move` keyword overrides inference, forcing all captures to be by value:

```rust
let s = String::from("hello");
let c = move || println!("{}", s);
// s is no longer accessible here; it was moved into the closure
```

This is essential when the closure must outlive the scope where it was created, as when spawning a thread:

```rust
let data = vec![1, 2, 3];
std::thread::spawn(move || {
    // data is owned by this closure, transferred to the new thread
    println!("{:?}", data);
});
```

Without `move`, the closure would capture `data` by reference. But `data` lives on the spawning thread's stack, which may be deallocated before the spawned thread runs. The compiler rejects this:

```rust
let data = vec![1, 2, 3];
std::thread::spawn(|| {
    println!("{:?}", data);  // ERROR: closure may outlive borrowed value
});
```

The `move` keyword forces ownership transfer, ensuring the closure owns `data` and can safely take it to another thread.

### The Fn Trait Hierarchy

The three traits `Fn`, `FnMut`, and `FnOnce` form a hierarchy that describes how a closure can be called, not how it captures.

`FnOnce` requires ownership of the closure to call it. The method signature is `fn call_once(self, args: Args) -> Output`. After calling a `FnOnce`, the closure is consumed. Every closure implements `FnOnce` because every closure can be called at least once.

`FnMut` requires mutable access to the closure. The signature is `fn call_mut(&mut self, args: Args) -> Output`. A closure implements `FnMut` if calling it does not require consuming any captured values. It may mutate its captures, but it does not move out of them.

`Fn` requires only shared access. The signature is `fn call(&self, args: Args) -> Output`. A closure implements `Fn` if calling it neither consumes nor mutates any captured values.

The hierarchy is `Fn: FnMut: FnOnce`. A closure implementing `Fn` automatically implements `FnMut` and `FnOnce`. The traits encode *what the closure does when called*, not *how it captured* its environment.

This can be tricky if we think in terms of C++ lambdas. For example, a `move` closure can still implement `Fn`:

```rust
let x = 42;
let c = move || x;  // captures x by value (copies, since i32: Copy)
// c implements Fn because calling it only reads the captured x
```

Conversely, a closure that captures by reference but mutates the referent implements only `FnMut`:

```rust
let mut counter = 0;
let mut increment = || { counter += 1; };
// increment implements FnMut (mutates through captured &mut)
// does NOT implement Fn
```

And a closure that moves out of a captured value implements only `FnOnce`:

```rust
let s = String::from("hello");
let consume = move || drop(s);
// consume implements FnOnce only (consumes captured s)
// does NOT implement FnMut or Fn
```

Functions accepting closures declare which trait they require. `Iterator::for_each` takes `FnMut` because it calls the closure multiple times but does not need concurrent shared access. `Iterator::map` takes `FnMut` for the same reason. `thread::spawn` takes `FnOnce` because the closure runs exactly once in the spawned thread.

```rust
fn call_twice<F: FnMut()>(mut f: F) {
    f();
    f();
}

fn call_once<F: FnOnce()>(f: F) {
    f();
}
```

### Closure Traits and Send/Sync

Closures inherit `Send` and `Sync` from their captures. If all captured values are `Send`, the closure is `Send`. If all values captured by shared reference are `Sync`, and all values captured by mutable reference, copy, or move are `Send`, then the closure is `Send`. These rules match normal struct composition.

```rust
use std::rc::Rc;

let rc = Rc::new(42);
let closure = move || println!("{}", rc);
// closure is NOT Send because Rc is not Send
// std::thread::spawn(closure);  // ERROR
```

```rust
use std::sync::Arc;

let arc = Arc::new(42);
let closure = move || println!("{}", arc);
// closure IS Send because Arc is Send
std::thread::spawn(closure);  // OK
```

A closure is `Clone` or `Copy` if it does not capture by mutable reference and all its captures are `Clone` or `Copy`:

```rust
let x = 42;
let closure = move || x;
// closure is Copy because it captures only Copy types by value

let y = closure;  // copy
let z = closure;  // still valid
```

These automatic trait derivations mean closures integrate with Rust's concurrency primitives without special handling. A `move` closure capturing `Arc<Mutex<T>>` is `Send`, can be shipped to another thread, and provides safe interior mutability through the mutex. The type system composes.
