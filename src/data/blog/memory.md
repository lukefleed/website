---
author: Luca Lombardo
pubDatetime: 2025-12-20T00:00:00.000Z
title: "Who Owns the Memory? Part 1: Objects"
slug: who-owns-the-memory-pt1
featured: false
draft: true
tags:
  - Rust
  - Programming
description: "From hardware to aliasing rules: understanding how C, C++, and Rust map types to memory."
---

## Table of contents

## Memory Is Just Bytes

A 64-bit processor sees memory as a flat array of $2^{64}$ addressable bytes. It does not know what a `struct` is. It does not know what an `int` is. When we execute `mov rax, [rbx]`, the CPU fetches 8 bytes starting at the address in `rbx`, shoves them into `rax`, and moves on. The semantic meaning of those bytes, whether they represent a pointer, a floating-point number, or part of a UTF-8 string—exists only in our source code and the instructions we generate.

This is the foundation everything else builds on. Before we discuss ownership models or type systems, we need to be precise about what actually happens when a program "allocates memory" or "accesses a variable."

### Virtual Address Space

Modern operating systems do not give processes direct access to physical RAM. Instead, each process operates within its own virtual address space, a fiction maintained by the MMU (Memory Management Unit) that maps virtual addresses to physical frames. The C standard captures this abstraction explicitly: pointers in C reference virtual memory, and the language makes no guarantees about physical layout.

This abstraction buys us two properties:

1. **Isolation**: A pointer in process A cannot reference memory in process B. Dereferencing an unmapped address triggers a page fault, typically terminating the process.
2. **Portability**: Code does not need to know the physical memory topology of the machine it runs on.

From our perspective as systems programmers, virtual memory means that the addresses we work with are translated by hardware before reaching DRAM. This translation has performance implications. TLB misses are expensive, but the abstraction holds: we operate on a contiguous address space that the OS manages for us.

### Alignment

Not all byte addresses are equal. On x86-64, loading a `uint64_t` from an address that is not divisible by 8 incurs a penalty. On stricter architectures like ARM (without unaligned access support) or older SPARC, it causes a hardware trap.

The reason is mechanical. DRAM is accessed in aligned chunks. When the CPU requests data at address `0x1003`, but the memory bus fetches 8-byte-aligned blocks, the memory controller must fetch two blocks (`0x1000-0x1007` and `0x1008-0x100F`), extract the relevant bytes, and reassemble them. This costs cycles.

The C standard formalizes this through the concept of alignment:
```c
#include <stdalign.h>
#include <stdio.h>

int main(void) {
    printf("alignof(int) = %zu\n", alignof(int));           // typically 4
    printf("alignof(double) = %zu\n", alignof(double));     // typically 8
    printf("alignof(max_align_t) = %zu\n", alignof(max_align_t)); // typically 16
}
```

The `alignof` operator (C11/C23) returns the required alignment for a type. Accessing an object at an address that violates its alignment is undefined behavior, not because the standard is being pedantic, but because the hardware cannot reliably execute it.

Consider this concrete failure case from [Modern C](https://gustedt.gitlabpages.inria.fr/modern-c/):
```c
union {
    unsigned char bytes[32];
    complex double val[2];
} overlay;

// complex double typically requires 8-byte alignment
complex double *p = (complex double *)&overlay.bytes[4];  // misaligned
*p = 1.0 + 2.0*I;  // undefined behavior
```

On x86-64 with alignment checking enabled, or on ARM, this crashes with a bus error. The pointer arithmetic is legal C, but the resulting address violates the alignment requirement of `complex double`. The hardware refuses.

### Cache Lines and Memory Bandwidth

Alignment interacts with another hardware reality: cache lines. On modern x86-64 processors, the L1 cache operates on 64-byte lines. When we read a single byte, the CPU actually fetches 64 bytes. If our data structures are laid out poorly, we waste bandwidth fetching bytes we never use.

Worse, if a single logical datum spans two cache lines, every access requires two cache fetches. For a `struct` that straddles a 64-byte boundary, this doubles memory traffic.

This is why compilers insert padding between struct fields. Consider:
```c
struct Bad {
    char a;     // 1 byte
    // 7 bytes padding
    double b;   // 8 bytes
    char c;     // 1 byte
    // 7 bytes padding (to align the whole struct)
};

struct Good {
    double b;   // 8 bytes
    char a;     // 1 byte
    char c;     // 1 byte
    // 6 bytes padding
};
```

`sizeof(struct Bad)` is 24 bytes. `sizeof(struct Good)` is 16 bytes. The compiler cannot reorder fields in C (the standard guarantees fields appear in declaration order for `repr(C)` compatibility), so the programmer must consider layout.

### The Allocator's View

When we call `malloc(n)`, we do not receive exactly `n` bytes of usable memory. The allocator maintains metadata like chunk headers, size fields, and free-list pointers that live adjacent to our allocation. In glibc's ptmalloc2, an allocated chunk looks roughly like this:
```
+------------------+
| prev_size        |  (8 bytes, used only if previous chunk is free)
+------------------+
| size       |N|M|P|  (8 bytes, includes flags in low 3 bits)
+------------------+
| user data...     |  <- pointer returned by malloc
|                  |
+------------------+
```

The `size` field stores the chunk size with three flag bits: `P` (previous chunk in use), `M` (chunk obtained via `mmap`), and `N` (non-main arena). The actual usable size is `size & ~0x7`.

This means:

1. Every allocation has overhead. Small allocations suffer proportionally more: a 16-byte allocation requires at least 32 bytes of actual memory (16 bytes data + 16 bytes metadata, depending on the allocator).
2. The allocator imposes its own alignment. `malloc` guarantees alignment suitable for any primitive type (`max_align_t`), which is 16 bytes on most 64-bit platforms.
3. Memory is not _free_. The allocator tracks allocated regions, and `free()` does not necessarily return memory to the OS, it typically returns it to a free list for reuse.

When we discuss ownership and resource management in later sections, keep this in mind: _deallocating memory_ at the language level means returning bytes to the allocator. The allocator decides when (if ever) to return pages to the operating system.

### Objects Are Bytes

The C standard makes this explicit: every object can be viewed as an array of `unsigned char`:
```c
int x = 42;
unsigned char *bytes = (unsigned char *)&x;
for (size_t i = 0; i < sizeof(int); i++) {
    printf("%02x ", bytes[i]);
}
// Output on little-endian x86-64: 2a 00 00 00
```

This is the object representation, the actual bytes in memory. The semantic interpretation (that these bytes represent the integer 42) is layered on top by the type system.

Rust and C++ inherit this model. When we say a Rust `i32` occupies 4 bytes with alignment 4, we mean exactly what C means: 4 contiguous bytes at an address divisible by 4. The type systems differ dramatically in what operations they permit on those bytes, but the physical representation is identical.

This is the starting point. We have bytes in a virtual address space, aligned for hardware access, managed by an allocator. Everything that follows, effective types, ownership, lifetimes—is a layer of abstraction over this physical reality.

## From Bytes to Objects

A region of memory becomes an *object* when we impose a type interpretation on it. The type dictates how many bytes participate, what their alignment must be, and what operations are valid. But the three languages differ fundamentally in when and how this imposition occurs, and what invariants the type carries.

### C: Effective Type Rules

In C, the relationship between memory and type is established through the concept of *effective type*. The effective type of an object determines how it may be accessed.

For declared variables, the effective type is simply the declared type:

```c
int x = 42;  // effective type of x is int
```

The variable `x` occupies `sizeof(int)` bytes at some address, and those bytes must be accessed as `int` or as `unsigned char`. Accessing them as `float*` is undefined behavior:

```c
int x = 42;
float *fp = (float *)&x;
float f = *fp;  // undefined behavior: access through incompatible type
```

This is not a runtime check. The compiler does not insert code to verify the access. Instead, the rule exists to enable optimization. When the compiler sees a write through `int*` and a read through `float*`, the strict aliasing rule permits it to assume these pointers reference different objects. The compiler can then reorder loads and stores, keep values in registers across the write, and eliminate redundant accesses.

The rule has a critical asymmetry. Any object can be viewed as an array of `unsigned char`:

```c
int x = 42;
unsigned char *bytes = (unsigned char *)&x;
for (size_t i = 0; i < sizeof(int); i++) {
    printf("%02x ", bytes[i]);  // valid: char access is always permitted
}
```

But the reverse is undefined:

```c
unsigned char buffer[sizeof(int)] = {0};
int *p = (int *)buffer;
int val = *p;  // undefined behavior
```

The buffer's effective type is `unsigned char[4]`. Accessing it through `int*` violates the effective type rule. The fact that the bytes happen to form a valid `int` representation is irrelevant. The compiler is entitled to assume this access cannot happen, and may generate code that produces garbage or crashes.

For dynamically allocated memory, the situation is different. Memory returned by `malloc` has no effective type until we write to it:

```c
void *p = malloc(sizeof(double));
double *dp = p;
*dp = 3.14;  // this write sets the effective type to double
```

After this write, the allocated region has effective type `double`. Subsequent reads through `double*` are valid. Reading through `int*` would again be undefined.

The effective type machinery exists purely for optimization. The compiler uses it to reason about aliasing. It provides no runtime safety.

### C++: Object Lifetime

C++ inherits C's effective type rules but adds a distinct concept: *object lifetime*. An object's lifetime is the interval during which accessing the object is well-defined.

The C++ standard (N4950, §6.7.3) specifies the boundaries precisely. For an object of type `T`:

> The lifetime of an object of type T begins when:
> (1.1) storage with the proper alignment and size for type T is obtained, and
> (1.2) its initialization (if any) is complete

> The lifetime of an object of type T ends when:
> (1.3) if T is a non-class type, the object is destroyed, or
> (1.4) if T is a class type, the destructor call starts, or
> (1.5) the storage which the object occupies is released, or is reused by an object that is not nested within o.

Consider placement new:

```cpp
struct Widget {
    int value;
    Widget(int v) : value(v) { }
    ~Widget() { std::cout << "destroyed\n"; }
};

alignas(Widget) unsigned char buffer[sizeof(Widget)];
Widget* w = new (buffer) Widget(42);  // lifetime begins here
w->~Widget();                          // lifetime ends here
// buffer still contains bytes, but no Widget object exists
```

Between placement new and the destructor call, a `Widget` object exists at that address. Before placement new and after the destructor, the bytes exist but no `Widget` does. Accessing `w->value` after `w->~Widget()` is undefined behavior, even though the bytes are still there and unchanged.

The destructor call does not free memory. It ends the object's lifetime while the storage remains intact. This is what placement new and explicit destructor calls rely on: the ability to construct an object in pre-existing storage, use it, destroy it, and potentially construct a different object in the same storage.

For trivial types (those without constructors, destructors, or virtual functions), C++ objects behave essentially like C objects. For class types with nontrivial special member functions, the lifetime boundaries become significant. A `std::string` accessed after destruction will likely read freed memory or corrupted pointers, because the destructor deallocated the internal buffer.

C++20 also introduced *implicit object creation*. Certain operations, such as `std::malloc`, implicitly create objects of *implicit-lifetime types* if doing so would give the program defined behavior:

```cpp
struct Point { int x, y; };  // implicit-lifetime type (trivial)

Point* p = (Point*)std::malloc(sizeof(Point));
p->x = 1;  // in C++20, this is well-defined
p->y = 2;  // malloc implicitly created the Point object
```

This was added to retroactively make well-defined the code patterns that had been common but technically undefined.

### Rust: Validity Invariants

Rust imposes a stronger requirement. Every type has *validity invariants*, and producing a value that violates its type's invariant is immediate undefined behavior. The compiler's optimizer assumes these invariants hold unconditionally.

The Rust Reference defines validity per type:

```rust
// A bool must be 0x00 (false) or 0x01 (true)
let b: bool = unsafe { std::mem::transmute(2u8) };  // UB: invalid bool

// A reference must be non-null, aligned, and point to a valid value
let r: &i32 = unsafe { std::mem::transmute(0usize) };  // UB: null reference

// A char must be a valid Unicode scalar value (not a surrogate)
let c: char = unsafe { std::mem::transmute(0xD800u32) };  // UB: surrogate

// An enum must have a valid discriminant
enum Status { Active = 0, Inactive = 1 }
let s: Status = unsafe { std::mem::transmute(2u8) };  // UB: invalid discriminant

// The never type must never exist
let n: ! = unsafe { std::mem::zeroed() };  // UB
```

The moment an invalid value is *produced*, undefined behavior has occurred. From the Rust Reference:

> The Rust compiler assumes that all values produced during program execution are valid, and producing an invalid value is hence immediate UB.

In C, you can have an `int` variable containing any 32-bit pattern, and as long as you do not read it in certain ways, no UB occurs. In Rust, if a `bool` contains the bit pattern `0x02`, UB has already happened at the point of creation, regardless of whether you subsequently read it.

Consider references. In C and C++, a pointer can be null, and dereferencing it is UB. But the pointer itself can exist and be passed around. In Rust:

```rust
let ptr: *const i32 = std::ptr::null();  // valid: raw pointer can be null
let r: &i32 = unsafe { &*ptr };          // UB occurs HERE, at reference creation
```

The UB does not occur when we read through the reference. It occurs when the reference is created. A `&T` carries an invariant: non-null, properly aligned, pointing to a valid `T`. Violating this invariant at any point is UB, regardless of what we do with the reference afterward.

This strictness enables more aggressive optimization. When the compiler sees a `&T`, it emits `dereferenceable` and `nonnull` annotations to LLVM. A match expression on a `bool` need not generate a default case for values 2-255. These optimizations would be unsound if invalid values could exist.

The cost is that more operations require `unsafe`. You cannot create a reference to potentially-invalid memory, even temporarily. You must use raw pointers and convert to references only when validity is guaranteed:

```rust
let ptr: *const i32 = some_ffi_function();

if !ptr.is_null() && ptr.is_aligned() {
    let r: &i32 = unsafe { &*ptr };  // sound: we verified validity
    println!("{}", *r);
}
```

### Object Representation vs. Value

All three languages distinguish between an object's *representation* (its bytes in memory) and its *value* (the semantic interpretation of those bytes). But they draw the line differently.

In C, you can freely inspect any object as `unsigned char[]`:

```c
double d = 3.14159;
unsigned char *bytes = (unsigned char *)&d;
// bytes[0..7] contain the IEEE 754 representation
```

The bytes are the object's representation. The value is what those bytes mean according to the IEEE 754 standard. C permits examining bytes without caring about their semantic meaning.

C++ inherits this but adds constraints around object lifetime. You can inspect the bytes of a live object. You cannot meaningfully inspect bytes of a destroyed object (the storage may be reused).

Rust permits byte-level inspection through raw pointers and transmutation, but adds validity constraints:

```rust
let x: i32 = 42;
let bytes: [u8; 4] = unsafe { std::mem::transmute(x) };
// bytes now contains the representation

// Going the other direction is more dangerous:
let bytes: [u8; 1] = [2];
let b: bool = unsafe { std::mem::transmute(bytes) };  // UB: 2 is not a valid bool
```

The asymmetry here mirrors C's effective type asymmetry. Converting a typed value to bytes is generally safe. Converting bytes to a typed value requires that the bytes actually constitute a valid value of that type.


## Storage Duration

Every object resides somewhere in memory. The *storage duration* of an object determines when that memory is allocated and when it becomes invalid. All three languages recognize the same fundamental categories, though they use different terminology and provide different guarantees about deallocation.

### The Four Categories

C defines four storage durations. C++ inherits the same four. Rust maps onto an equivalent model, though the language specification does not use identical terminology.

**Static storage duration**: The object exists for the entire execution of the program. In C and C++, this includes global variables, variables declared with `static`, and string literals. In Rust, this includes `static` items and string literals (which have type `&'static str`). The memory for these objects is typically placed in the `.data` or `.rodata` segment of the executable and requires no runtime allocation.

**Thread storage duration**: The object exists for the lifetime of a thread. C11 introduced `_Thread_local` (spelled `thread_local` since C23), C++11 introduced `thread_local`, and Rust provides `thread_local!` macro. Each thread gets its own instance of the variable, allocated when the thread starts and deallocated when it terminates.

**Automatic storage duration**: The object exists within a lexical scope, typically a function body or block. When execution enters the scope, space is reserved; when execution leaves, the space is released. In C and C++, local variables without `static` or `thread_local` have automatic storage. In Rust, all local bindings have automatic storage. This is typically implemented via the stack.

**Allocated (dynamic) storage duration**: The object's lifetime is controlled explicitly by the program. In C, this means `malloc`/`free`. In C++, this means `new`/`delete` or allocator-aware containers. In Rust, this means `Box`, `Vec`, `String`, and other heap-allocating types.

### Stack

Automatic storage is almost universally implemented using a call stack. When a function is called, the compiler reserves space for its local variables by adjusting the stack pointer. On x86-64 following the System V ABI, this looks like:

```asm
my_function:
    push rbp
    mov rbp, rsp
    sub rsp, 48          ; reserve 48 bytes for locals
    ; ... function body ...
    mov rsp, rbp
    pop rbp
    ret
```

The `sub rsp, 48` instruction allocates space for all local variables in a single operation. The compiler computes the required size at compile time by summing the sizes of all locals (accounting for alignment). Deallocation is equally cheap: `mov rsp, rbp` releases all that space instantly.

This has two consequences:

1. Allocation and deallocation of automatic storage is $O(1)$ regardless of how many objects are involved. A function with 100 local variables pays the same cost as one with 2.

2. The space is not initialized. After `sub rsp, 48`, those 48 bytes contain whatever was previously on the stack. In C, reading an uninitialized automatic variable is undefined behavior (the value is *indeterminate*). In C++, the same rule applies. In Rust, the compiler enforces definite initialization: you cannot read a variable before assigning to it.

```rust
fn example() {
    let x: i32;
    println!("{}", x);  // error: borrow of possibly-uninitialized variable
}
```

The Rust compiler tracks initialization state through control flow and rejects programs that might read uninitialized memory. This is a compile-time check with no runtime cost.

### Heap

Dynamic allocation is fundamentally different. When we call `malloc(n)`, the allocator must:

1. Find a contiguous region of at least `n` bytes that is not currently in use
2. Mark that region as allocated
3. Return a pointer to it

When we call `free(p)`, the allocator must:

1. Determine the size of the allocation (stored in metadata adjacent to the user data)
2. Mark that region as available for future allocations
3. Possibly coalesce adjacent free regions to reduce fragmentation

This involves data structure manipulation, potential system calls (if the allocator needs more memory from the OS), and can have variable latency depending on heap state. The cost is not $O(1)$.

In C, heap allocation is explicit:

```c
int* p = malloc(sizeof(int) * 100);
if (p == NULL) {
    // allocation failed
}
// ... use p ...
free(p);
```

The programmer is responsible for:
1. Checking for allocation failure
2. Calling `free` exactly once
3. Not using the pointer after `free`
4. Not freeing the same pointer twice

Violating any of these causes undefined behavior or memory leaks. The language provides no assistance.

In C++, dynamic allocation can be explicit (`new`/`delete`) or managed through RAII:

```cpp
// Explicit (dangerous)
int* p = new int[100];
delete[] p;

// RAII (safer)
auto v = std::make_unique<int[]>(100);
// v automatically deleted when it goes out of scope
```

`std::unique_ptr` wraps a raw pointer and calls `delete` in its destructor. When `v` goes out of scope, the destructor runs, the memory is freed. The programmer does not call `delete` manually.

This is opt-in. You can still use raw `new`/`delete`. You can still have dangling pointers. The compiler does not verify correctness.

In Rust, heap allocation is handled through owning types:

```rust
let v: Vec<i32> = Vec::with_capacity(100);
// v automatically deallocated when it goes out of scope
```

`Vec<T>` owns heap-allocated memory. When `v` goes out of scope, `Vec`'s `Drop` implementation runs, calling the allocator to free the buffer. There is no way to forget to free, no way to double-free, and no way to use after free (the compiler rejects such programs).

The difference from C++ is that Rust's ownership is not opt-in. Every heap allocation is owned by exactly one binding. Transferring ownership is a move. After a move, the original binding is unusable:

```rust
let v1 = vec![1, 2, 3];
let v2 = v1;           // v1 moved to v2
println!("{:?}", v1);  // error: borrow of moved value
```

### Who Calls Free?

The central difference between C, C++, and Rust in their treatment of dynamic storage is responsibility for deallocation.

In C, the programmer decides when to call `free`. The language does not track ownership. If you pass a pointer to a function, the function might free it, or it might not. The only way to know is documentation or convention.

```c
void process(int* data) {
    // Does this function free data? You have to read the docs.
}
```

In C++, RAII shifts responsibility to destructors. If you use `unique_ptr`, the destructor frees. If you use `shared_ptr`, the destructor decrements a reference count and frees when it reaches zero. But you can still use raw pointers, and the compiler cannot tell you which convention a given codebase follows.

```cpp
void process(int* data) {
    // Raw pointer: who owns this? Still ambiguous.
}

void process(std::unique_ptr<int[]> data) {
    // Ownership transferred: this function will free when done.
}
```

In Rust, the type system encodes ownership:

```rust
fn process(data: Vec<i32>) {
    // This function owns data. It will be freed when process returns.
}

fn process_ref(data: &Vec<i32>) {
    // This function borrows data. The caller retains ownership.
}

fn process_mut(data: &mut Vec<i32>) {
    // Mutable borrow. Caller retains ownership. No other access allowed during this call.
}
```

The signature tells you everything. `Vec<i32>` means ownership transfer. `&Vec<i32>` means immutable borrow. `&mut Vec<i32>` means mutable borrow. The compiler enforces these semantics. You cannot pass a `Vec` and then continue using it; the move would be rejected.

### Stack Allocation in Rust

Rust provides fine-grained control over whether data lives on the stack or heap. By default, local bindings are stack-allocated:

```rust
let x: [i32; 1000] = [0; 1000];  // 4000 bytes on the stack
```

This works until the array is too large for the stack (typically 1-8 MB depending on platform). For large allocations, use `Box`:

```rust
let x: Box<[i32; 1000000]> = Box::new([0; 1000000]);  // heap
```

`Box<T>` is a pointer to a heap allocation. It has the same size as a raw pointer (8 bytes on 64-bit), implements `Deref` so you can use it like a reference, and frees the allocation in its `Drop` implementation.

The memory layout of `Box<T>` is a single pointer:

```rust
use std::mem::size_of;
assert_eq!(size_of::<Box<[i32; 1000]>>(), 8);  // just a pointer
```

Unlike C++ `unique_ptr`, which may carry a deleter, `Box<T>` always uses the global allocator and has no space overhead. The deallocation function is known statically.

### How Does Vec Work?

`Vec<T>` is Rust's growable array. Its memory layout is three machine words:

```rust
// Conceptually:
struct Vec<T> {
    ptr: *mut T,      // pointer to heap allocation
    cap: usize,       // allocated capacity
    len: usize,       // number of initialized elements
}
```

On 64-bit, `size_of::<Vec<T>>()` is 24 bytes regardless of `T`.

When you create a `Vec`:

```rust
let mut v = Vec::with_capacity(10);
```

The allocator provides a buffer for 10 elements. `ptr` points to it, `cap` is 10, `len` is 0.

When you push elements:

```rust
v.push(1);  // len becomes 1
v.push(2);  // len becomes 2
```

Each `push` writes to `ptr.add(len)` and increments `len`. No reallocation occurs while `len < cap`.

When `len` reaches `cap` and you push again, `Vec` must reallocate:

1. Allocate a new buffer with larger capacity
2. Copy existing elements to the new buffer
3. Free the old buffer
4. Update `ptr` and `cap`

When `Vec` is dropped:

1. Call `drop` on each element (for types with destructors)
2. Free the buffer

All of this happens automatically. The programmer writes `v.push(x)` and the ownership system ensures the buffer is freed exactly once, when `v` goes out of scope.

This is the pattern throughout Rust's standard library. `String` owns a UTF-8 buffer. `HashMap` owns its backing storage. `File` owns a file descriptor. When the owner goes out of scope, the resource is released. The type system ensures there is always exactly one owner.