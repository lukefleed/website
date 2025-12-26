---
author: Luca Lombardo
pubDatetime: 2025-12-25T00:00:00.000Z
title: "Who Owns the Memory? Part 3: "
slug: who-owns-the-memory-pt3
featured: false
draft: true
tags:
  - Rust
  - Programming
description: " "
---

## Table of Contents

## Type Layout and Memory Representation

In Part I we established that every type has a size and an alignment, and that compilers insert padding to satisfy alignment constraints. We showed how a poorly ordered struct in C could waste 8 bytes of padding where a well-ordered one used only 2. But we sidestepped a question: who decides the order?

In C and C++, the answer is straightforward: we do. The compiler lays out fields in declaration order, inserting padding as the alignment algorithm dictates. This predictability is essential for binary compatibility, memory-mapped I/O, and network protocols where byte offsets must match external specifications. It is also a constraint: the programmer bears responsibility for field ordering, and a careless declaration can bloat a frequently-allocated struct.

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

There is no guarantee about field order. The compiler may reorder fields to minimize padding, and different compilations of the same source may produce different layouts. The same generic struct instantiated with different type parameters may have different field orderings.

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
struct MyVariantA { tag: u32, value: u32 }

#[repr(C)]
struct MyVariantB { tag: u32, _pad: u32, value0: f32, _pad2: u32, value1: u64 }
// ... and so on
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

Adding an explicit `repr` to an enum with fields has a consequence that surprises many: it *suppresses niche optimization*. We have not yet explained what niche optimization is, so this statement may seem cryptic. We will return to it after discussing zero-sized types, where the concept will make more sense.

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

The standard library's `HashSet` is *literally* implemented this way:

```rust
pub struct HashSet<T, S = RandomState> {
    base: base::HashSet<T, S>,
}

// where base::HashSet is:
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

This might seem useless, but it enables type-level reasoning about impossibility. Consider a function that parses input and might fail:

```rust
fn parse(input: &str) -> Result<Data, ParseError> { /* ... */ }
```

Now consider a function that validates already-parsed data. The validation cannot fail because the data has already been parsed:

```rust
fn validate(data: &Data) -> Result<(), Void> {
    // validation logic that always succeeds
    Ok(())
}
```

The return type `Result<(), Void>` communicates that the `Err` variant is impossible. The compiler knows this. When we pattern match on the result, we do not need to handle `Err`:

```rust
let Ok(()) = validate(&data);  // irrefutable pattern, no Err case needed
```

The compiler optimizes based on this knowledge. `Result<T, Void>` has the same layout as `T` because the `Err` variant cannot exist and requires no discriminant. The representation is identical:

```rust
use std::mem::size_of;

enum Void {}

assert_eq!(size_of::<Result<u64, Void>>(), 8);  // same as u64
assert_eq!(size_of::<u64>(), 8);
```

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

We can examine the actual representation using `std::ptr::metadata`:

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

Inlining is the most significant loss. When the compiler inlines a function, it can see both caller and callee code simultaneously. This enables constant propagation, dead code elimination, loop fusion, and SIMD vectorization across the boundary. None of this is possible through a vtable. The compiler cannot see through the indirection at compile time, so each call is an optimization barrier.

Consider a loop summing areas:

```rust
// Static dispatch: the compiler can inline, vectorize, and unroll
fn sum_areas_static<T: Drawable>(shapes: &[T]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// Dynamic dispatch: each area() call goes through vtable
fn sum_areas_dynamic(shapes: &[&dyn Drawable]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}
```

With static dispatch, if `T` is `Circle` and `area()` is a simple computation, the compiler can inline the entire loop body, unroll the loop, and potentially vectorize with SIMD. With dynamic dispatch, each `area()` call is a function pointer load, an indirect call, and a return. The loop cannot be vectorized because the compiler cannot prove anything about what `area()` does.

The rule of thumb: use generics with trait bounds for performance-critical code paths. Reserve trait objects for heterogeneous collections where the flexibility is worth the cost, or for reducing compile times and binary size when performance is not critical.

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

The previous section examined how Rust represents trait objects at runtime, with wide pointers carrying vtable metadata. But trait objects are only one half of Rust's polymorphism story. To understand why two mechanisms exist and when to choose each, we need to examine the fundamental tension that every systems language must resolve: writing code that operates on multiple types without sacrificing performance.

### The Core Trade-off

Consider a function that finds the maximum element in a slice. We want this to work for `i32`, `f64`, `String`, and any other type that can be compared. Writing separate implementations for each type is tedious and error-prone. We need *parametric polymorphism*, code parameterized over types.

Two implementation strategies exist. The compiler can generate a specialized copy of the code for each concrete type used, a process called *monomorphization*. Alternatively, the compiler can generate a single copy of the code that operates through an indirection layer, dispatching to type-specific implementations at runtime via *vtables*. Both C++ and Rust support both strategies. C, lacking native generics, relies on workarounds that approximate each approach.

Monomorphization eliminates runtime indirection but increases binary size and compile time. Dynamic dispatch keeps code size small but introduces branch prediction overhead and prevents inlining. Understanding when each trade-off applies requires examining how each language implements both strategies.

### C: Life Without Generics

C has no built-in parametric polymorphism. We historically used three workarounds, each with significant limitations.

**Preprocessor macros** perform textual substitution before compilation. We can write a type-agnostic max function as a macro:

```c
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int x = MAX(3, 5);           // expands to ((3) > (5) ? (3) : (5))
double y = MAX(1.5, 2.5);    // expands to ((1.5) > (2.5) ? (1.5) : (2.5))
```

This achieves something like monomorphization, since each use site expands to type-specific code. But macros operate outside the type system. The preprocessor has no idea what types `a` and `b` are; it pastes text. This leads to classic pitfalls:

```c
#define SQUARE(x) ((x) * (x))
int a = 5;
int b = SQUARE(a++);  // expands to ((a++) * (a++)), UB due to double increment
```

The macro evaluates its argument twice, causing undefined behavior when the argument has side effects.

**`_Generic` selection** (C11) provides compile-time type dispatch:

```c
#define abs(x) _Generic((x),    \
    int: abs_int,               \
    long: abs_long,             \
    double: fabs,               \
    default: abs_int)(x)

int abs_int(int x) { return x < 0 ? -x : x; }
long abs_long(long x) { return x < 0 ? -x : x; }
```

The `_Generic` keyword examines the type of its first argument and selects the corresponding expression. This is more principled than macros, since the selection happens within the type system. But it requires manually listing every supported type and writing separate implementations for each. We have not reduced code duplication; we have centralized the dispatch.

**Function pointers with `void*`** approximate dynamic dispatch:

```c
typedef int (*comparator)(const void*, const void*);

void qsort(void* base, size_t nmemb, size_t size, comparator cmp);

int compare_int(const void* a, const void* b) {
    return *(const int*)a - *(const int*)b;
}

int arr[] = {5, 2, 8, 1};
qsort(arr, 4, sizeof(int), compare_int);
```

The standard library's `qsort` operates on arbitrary types by treating the array as raw bytes (`void*`) and accepting a comparator function pointer. The actual comparison logic lives in a type-specific callback. This works, but sacrifices type safety: nothing prevents passing a `compare_int` function to sort an array of `double`. The compiler cannot verify correctness.

None of these approaches satisfies. Macros lack type safety and hygiene. `_Generic` requires exhaustive enumeration of types. Function pointers with `void*` sacrifice compile-time checking entirely. C++ templates and Rust generics were designed to solve these problems.

### C++ Templates: Monomorphization with Duck Typing

C++ templates define families of functions or classes parameterized over types:

```cpp
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

int x = max(3, 5);           // instantiates max<int>
double y = max(1.5, 2.5);    // instantiates max<double>
```

When the compiler encounters `max(3, 5)`, it deduces that `T = int` and generates a specialized function `max<int>`. When it encounters `max(1.5, 2.5)`, it generates `max<double>`. Each instantiation is a separate function in the final binary.

This is monomorphization: the generic template is transformed into multiple concrete implementations. The generated code is identical to what we would write by hand. There is no runtime overhead; each call to `max<int>` is a direct call to a function that compares two `int` values.

**Duck typing and SFINAE.** C++ templates use what is sometimes called *duck typing*: the template body is compiled against the concrete type, and if the operations in the body are valid for that type, the instantiation succeeds. If not, the compiler emits an error.

```cpp
template<typename T>
void print(T x) {
    std::cout << x;  // requires operator<< for T
}

print(42);            // OK: int has operator<<
print(std::vector<int>{1,2,3});  // error: no operator<< for vector
```

The error message emerges from deep within the template instantiation, often producing notoriously verbose output that obscures the actual problem. The root cause is that template requirements are implicit: we discover at instantiation time whether the operations are valid, not at definition time.

This implicit checking enables a technique called SFINAE (Substitution Failure Is Not An Error). When the compiler tries to instantiate a template and the substitution fails, it does not immediately produce an error; it removes that template from the overload set. This allows *template metaprogramming* where we select between implementations based on type properties:

```cpp
#include <type_traits>

// Enabled only for integral types
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
absolute(T x) {
    return x < 0 ? -x : x;
}

// Enabled only for floating-point types
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
absolute(T x) {
    return std::fabs(x);
}
```

The `std::enable_if` machinery conditionally makes the return type valid or invalid depending on the type trait. When instantiating `absolute<int>`, the first overload succeeds (integral type), so the second is discarded via SFINAE. When instantiating `absolute<double>`, the second succeeds.

SFINAE is powerful but arcane. The code is dense with template machinery that obscures the actual logic. Error messages remain poor because the constraints are expressed indirectly through type manipulations.

**C++20 Concepts** address this by making constraints explicit:

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

The `Comparable` concept declares what operations a type must support. The template explicitly states its constraint: `T` must satisfy `Comparable`. If we try to instantiate `max` with a type that lacks comparison operators, the error message directly refers to the violated concept, not to failed substitution deep in the template body.

Concepts bring C++ templates closer to Rust's trait bounds. The constraint is declared upfront, checked at instantiation against the concept definition, and produces clear error messages.

**Template costs.** Monomorphization has compile-time and binary-size costs. The compiler must parse and instantiate templates separately in every translation unit that uses them. This is why C++ template code traditionally lives in headers: the definition must be visible wherever instantiation occurs.

Each distinct instantiation produces a separate copy of the generated code. A template used with 50 different types produces 50 copies of the function in the final binary. For large templates like `std::sort` or `std::unordered_map`, this causes significant binary bloat.

The standard mitigation is *explicit instantiation*, where we declare which instantiations to generate in a single translation unit:

```cpp
// header: my_template.h
template<typename T>
void process(T x);

// source: my_template.cpp
template<typename T>
void process(T x) { /* implementation */ }

// explicit instantiations
template void process<int>(int);
template void process<double>(double);
```

Other translation units can now use `process<int>` and `process<double>` without triggering instantiation; they link against the pre-generated code. This reduces compile time and binary size at the cost of flexibility (only the explicitly instantiated types are available).

### C++ Virtual Functions: The Itanium ABI

C++ also supports runtime polymorphism through virtual functions. A class with at least one virtual function is *polymorphic*:

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

When we call a virtual function through a pointer or reference to the base class, the actual function called depends on the dynamic type of the object:

```cpp
void print_area(const Shape& s) {
    std::cout << s.area() << "\n";  // virtual dispatch
}

Circle c(1.0);
Rectangle r(2.0, 3.0);
print_area(c);  // calls Circle::area
print_area(r);  // calls Rectangle::area
```

The compiler cannot know at compile time which `area` implementation to call; `s` might be any type derived from `Shape`. The decision is deferred to runtime.

**The vptr and vtable.** To implement this, the compiler inserts a hidden pointer (the *vptr*) into every object of a polymorphic class. This pointer refers to a static table of function pointers (the *vtable*) shared by all objects of the same dynamic type.

The Itanium C++ ABI (used by GCC, Clang, and most non-MSVC compilers) specifies the vtable layout precisely. The vtable contains components at negative offsets from the address the vptr points to:

```
Vtable for Circle (vptr points here → offset 0):
-16:  offset-to-top (ptrdiff_t, 0 for complete objects)
-8:   RTTI pointer (typeinfo for Circle)
 0:   &Circle::~Circle() [complete destructor]
+8:   &Circle::~Circle() [deleting destructor]
+16:  &Circle::area()
```

The *offset-to-top* field holds the displacement from the vptr location to the top of the complete object. For a complete object (not a base subobject), this is zero. For secondary vtables in multiple inheritance hierarchies, it can be negative. This field enables `dynamic_cast<void*>` to find the most-derived object.

The *RTTI pointer* points to the `typeinfo` structure used for `typeid` and `dynamic_cast`. It is always present, even when RTTI is disabled at compile time (in which case it is null).

Virtual function pointers follow in declaration order. The destructor occupies two slots: the *complete destructor* destroys the object but does not free memory, while the *deleting destructor* destroys and then calls `operator delete`. This distinction matters for correct cleanup of heap-allocated polymorphic objects.

A `Circle` object has the following memory layout:

```
Circle object:
+0:   vptr (points to Circle vtable, offset 0)
+8:   radius (double)
```

Every `Circle` instance carries the vptr as its first member, adding 8 bytes of overhead on 64-bit platforms.

**Virtual call in assembly.** When we call `s.area()` where `s` is a `Shape&`, the compiler generates code to load the vptr from the object, index into the vtable to find the function pointer for `area`, and call through that function pointer.

On x86-64:

```asm
; rdi = pointer to Shape object (this)
print_area:
    mov     rax, [rdi]          ; load vptr from object
    mov     rax, [rax + 16]     ; load area() pointer from vtable offset +16
    jmp     rax                 ; tail call to the virtual function
```

The two memory loads (vptr, then vtable entry) occur on every virtual call. More significantly, the indirect jump through `rax` is an *indirect branch*. The CPU's branch predictor must guess the target without knowing it until the register value is computed.

**Performance implications.** Virtual functions prevent the compiler from inlining. When we call `s.area()`, the compiler does not know which concrete function will be invoked, so it cannot substitute the function body at the call site. This blocks constant propagation, dead code elimination, and other optimizations that cross function boundaries.

The indirect branch also strains the CPU. Direct calls have predictable targets that the instruction prefetcher can anticipate. Indirect calls through registers require the *indirect branch predictor*, which maintains a table of recent targets for each call site. If a call site invokes many different implementations (a heterogeneous collection of shapes), the predictor may never stabilize, causing pipeline stalls on every misprediction.

The trade-off is compile time and binary size. There is only one copy of `print_area`, not one per shape type. The vtable adds per-class overhead, not per-use overhead like template instantiation. For large class hierarchies with many virtual methods, this can significantly reduce binary size compared to templated alternatives.

### Rust Generics: Monomorphization with Explicit Bounds

Rust generics follow the monomorphization strategy, like C++ templates, but with a crucial difference: constraints are explicit from the start.

```rust
fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

let x = max(3, 5);           // instantiates max::<i32>
let y = max(1.5, 2.5);       // instantiates max::<f64>
```

The bound `T: PartialOrd` states upfront that `T` must implement the `PartialOrd` trait. The compiler checks this bound at the call site: if we try to call `max` with a type that does not implement `PartialOrd`, the error message directly states that the trait bound is not satisfied.

Inside the function body, we can only use operations that `PartialOrd` guarantees. The compiler rejects code that would compile for some `T` but not others:

```rust
fn broken<T: PartialOrd>(a: T, b: T) -> T {
    println!("{}", a);  // error: T doesn't implement Display
    if a > b { a } else { b }
}
```

This differs from C++ templates, where the body is compiled *tentatively* and errors emerge during instantiation. Rust checks the generic function against its declared bounds before any instantiation occurs.

**Trait bounds vs Concepts.** Rust trait bounds and C++20 Concepts serve similar purposes: making generic constraints explicit. The key difference is integration with the type system.

A Rust trait is a first-class type system construct. Implementing a trait for a type adds that type to the trait's set of implementors. The relationship is explicit and verifiable by the compiler.

A C++ concept is a predicate that checks whether certain expressions are valid. The expressions in a `requires` clause are evaluated syntactically; the concept does not establish a formal relationship between the type and the operations. Two types might satisfy the same concept for entirely different reasons.

In practice, both approaches produce clear error messages and enable the compiler to reject invalid instantiations early.

**Monomorphization in the compiler.** When the Rust compiler encounters a generic function call, it records the concrete types used. During code generation, the *monomorphization collector* traverses the call graph to find all required instantiations:

```
main calls max::<i32>
main calls max::<f64>
max::<i32> needs PartialOrd::gt for i32
max::<f64> needs PartialOrd::gt for f64
```

The collector produces a list of *mono items*: concrete functions that need machine code generated. Each generic function paired with each set of concrete type arguments becomes a distinct mono item.

The compiler then partitions these items into *Codegen Units* (CGUs). For incremental compilation, the partitioner creates two CGUs per source module: one for stable, non-generic code, and one for monomorphized instances. This allows the compiler to reuse stable code when only the generic instantiations change.

**Reducing monomorphization bloat.** The same binary size concerns apply to Rust as to C++ templates. A generic function used with many types produces many copies.

The `cargo llvm-lines` tool shows which functions contribute most to generated LLVM IR:

```
$ cargo llvm-lines
Lines          Copies        Function name
-----          ------        -------------
30000          150           core::ptr::drop_in_place
12000          80            alloc::vec::Vec<T>::push
8000           40            core::result::Result<T,E>::map
```

Common utility functions like `Option::map` or `Result::map_err` get instantiated for every type they are used with. In large codebases, these can dominate binary size.

The standard mitigation is the *inner function pattern*: move the bulk of the logic into a non-generic inner function, leaving only a thin generic wrapper:

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

The outer generic function calls `as_ref()` to convert the generic `P` to a concrete `&Path`, then delegates to the non-generic `inner`. Now `inner` is compiled once, regardless of how many different path types are used.

### Rust Trait Objects: The Dynamic Alternative

When monomorphization costs are prohibitive, Rust offers trait objects as a dynamic dispatch alternative. Section 11 covered the representation; here we focus on the trade-off.

A trait object `&dyn Trait` is a wide pointer containing a data pointer and a vtable pointer. Calling a method involves loading the function pointer from the vtable and invoking it indirectly:

```rust
fn total_area(shapes: &[&dyn Shape]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}
```

Unlike C++, the vtable pointer is in the reference, not the object. A `Circle` has no embedded vptr; it gains virtual dispatch only when viewed through a `&dyn Shape`.

The trade-off mirrors C++: dynamic dispatch prevents inlining and introduces indirect branch overhead, but produces a single copy of `total_area` regardless of how many shape types exist.

### Comparing Implementations

Consider a function that computes the total area of shapes in a collection, implemented in all three languages.

**C (function pointers):**

```c
typedef struct Shape Shape;
typedef double (*area_fn)(const Shape*);

struct Shape {
    area_fn area;
    // shape-specific data follows
};

double total_area(Shape** shapes, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += shapes[i]->area(shapes[i]);
    }
    return sum;
}
```

We embed the vtable in each object as a function pointer. No type safety; we must ensure consistency manually.

**C++ (templates, static):**

```cpp
template<typename Container>
double total_area(const Container& shapes) {
    double sum = 0;
    for (const auto& s : shapes) {
        sum += s.area();
    }
    return sum;
}
```

Monomorphized for each container type. Requires homogeneous collections (all elements the same type within a container).

**C++ (virtual, dynamic):**

```cpp
double total_area(const std::vector<Shape*>& shapes) {
    double sum = 0;
    for (const auto* s : shapes) {
        sum += s->area();  // virtual call
    }
    return sum;
}
```

Single copy of the function. Supports heterogeneous collections (different derived types). Each call goes through the vtable.

**Rust (generics, static):**

```rust
fn total_area<T: Shape>(shapes: &[T]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}
```

Monomorphized for each `T`. Homogeneous collections only.

**Rust (trait objects, dynamic):**

```rust
fn total_area(shapes: &[&dyn Shape]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}
```

Single copy. Heterogeneous collections. Each `area()` call is indirect.

### The Assembly View

To make the trade-off concrete, consider a simple operation: incrementing a counter.

**Static dispatch:**

```rust
trait Counter {
    fn increment(&mut self);
}

struct SimpleCounter(u64);
impl Counter for SimpleCounter {
    fn increment(&mut self) { self.0 += 1; }
}

fn inc_static<T: Counter>(c: &mut T) {
    c.increment();
}
```

For `inc_static::<SimpleCounter>`, the compiler generates:

```asm
inc_static_SimpleCounter:
    add     qword ptr [rdi], 1
    ret
```

The generic function is monomorphized to a single `add` instruction. The method call is inlined entirely.

**Dynamic dispatch:**

```rust
fn inc_dynamic(c: &mut dyn Counter) {
    c.increment();
}
```

The compiler generates:

```asm
inc_dynamic:
    ; rdi = data pointer, rsi = vtable pointer
    mov     rax, [rsi + 24]    ; load increment from vtable
    jmp     rax                ; tail call
```

The function loads the method pointer from the vtable and jumps to it. The actual increment happens in the target function, which cannot be inlined here.

For a single increment, the difference is trivial. In a tight loop incrementing millions of times, the static version avoids the vtable load and indirect branch on every iteration. Whether this matters depends on the surrounding code and how well the branch predictor can anticipate the target.

### Choosing Between Dispatch Strategies

Static dispatch (monomorphization) fits when:

- The hot path requires maximum performance
- The set of types is small and known
- Inlining and optimization across the generic boundary matters
- Compile time and binary size are acceptable costs

Dynamic dispatch (vtables/trait objects) fits when:

- The type is not known until runtime (plugin systems, user-defined types)
- Binary size is a concern
- Compile time is a concern
- The performance difference is negligible for the use case

A common pattern combines both: we use generics in public APIs for flexibility, then internally convert to trait objects to reduce instantiation count:

```rust
pub fn process<W: Write>(writer: W) {
    process_dyn(&mut writer as &mut dyn Write)
}

fn process_dyn(writer: &mut dyn Write) {
    // large implementation, compiled once
}
```

The public API accepts any `Write` implementor. Internally, we immediately convert to a trait object, so `process_dyn` is compiled only once. The cost is one virtual dispatch per method call within `process_dyn`, but the binary contains only one copy of the implementation.

Both C++ and Rust provide both mechanisms because neither dominates the other in all scenarios. The choice depends on the specific requirements: static dispatch for hot paths, dynamic dispatch for flexibility and code size, and hybrid patterns when we need both.