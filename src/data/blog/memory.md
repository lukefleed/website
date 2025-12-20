---
author: Luca Lombardo
pubDatetime: 2025-12-20T00:00:00Z
title: Who Owns the Memory?
slug: who-owns-the-memory
featured: false
draft: true
tags:
  - Rust
  - Programming
description: A comparative study of memory management in Rust, C and C++.
---

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

C++ inherits C's memory model but layers a more elaborate object model on top. Where C speaks of effective types, C++ speaks of *object lifetime*.

The C++ standard (N4950, §6.7.3) defines precisely when an object's lifetime begins and ends. For an object of type `T`:

> The lifetime of an object of type T begins when:
> (1.1) storage with the proper alignment and size for type T is obtained, and
> (1.2) its initialization (if any) is complete

> The lifetime of an object of type T ends when:
> (1.3) if T is a non-class type, the object is destroyed, or
> (1.4) if T is a class type, the destructor call starts, or
> (1.5) the storage which the object occupies is released, or is reused by an object that is not nested within o.

This is more nuanced than C's model. Consider:

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

Between placement new and the destructor call, a `Widget` object exists at that address. Before placement new and after the destructor, the bytes exist but no `Widget` does. Accessing `w->value` after `w->~Widget()` is undefined behavior, even though the bytes are still there.

This matters for type-based alias analysis. The C++ standard permits compilers to assume that objects of incompatible types do not alias, just as in C. But C++ adds the temporal dimension: accessing an object outside its lifetime is also undefined.

For trivial types (those without constructors, destructors, or virtual functions), C++ objects behave essentially like C objects. But for class types with nontrivial special member functions, the lifetime model is essential. The destructor call does not free memory. It ends the object's lifetime while leaving the storage intact. This separation between storage and lifetime is what enables placement new and explicit destructor calls.

C++11 also introduced *implicit object creation*. Certain operations (like `std::malloc`, or writing to an array of `unsigned char`) implicitly create objects of *implicit-lifetime types* if doing so would give the program defined behavior:

```cpp
struct Point { int x, y; };  // implicit-lifetime type

Point* p = (Point*)std::malloc(sizeof(Point));
p->x = 1;  // in C++20, this is well-defined
p->y = 2;  // malloc implicitly created the Point object
```

This was added to retroactively bless code patterns that had been common but technically undefined. The standard now says that `malloc` implicitly creates an object of the appropriate type if that would make the program have defined behavior.

### Rust: Validity Invariants

Rust takes a stricter position. Types carry *validity invariants*, and producing a value that violates its type's invariant is immediate undefined behavior. This is not merely an optimization hint. It is a semantic requirement that the compiler assumes always holds.

The Rust Reference enumerates the validity requirements:

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

The moment an invalid value is *produced*, undefined behavior has occurred. The Rust Reference states:

> The Rust compiler assumes that all values produced during program execution are valid, and producing an invalid value is hence immediate UB.

This is more aggressive than C or C++. In C, you can have an `int` variable containing any 32-bit pattern, including trap representations on exotic platforms, and as long as you do not read it, no UB occurs. In Rust, if a `bool` contains the bit pattern `0x02`, UB has already happened at the point of creation, regardless of whether you subsequently read it.

Consider references. In C and C++, a pointer can be null, and dereferencing it is UB. But the pointer itself can exist and be passed around. In Rust:

```rust
let ptr: *const i32 = std::ptr::null();  // valid: raw pointer can be null
let r: &i32 = unsafe { &*ptr };          // UB occurs HERE, at reference creation
```

The UB does not occur when we try to read through the reference. It occurs when the reference is created. A `&T` type has an invariant: it must be non-null, properly aligned, and point to a valid `T`. Violating this invariant at any point is UB, regardless of what we do with the reference afterward.

This strict approach enables more aggressive optimization. When the compiler sees a `&T`, it can assume the reference is valid for the duration of its use. It can emit `dereferenceable` and `nonnull` annotations to LLVM, enabling optimizations that would be unsound if null references could exist.

The trade-off is that more operations require `unsafe`. You cannot create a reference to potentially-invalid memory, even temporarily. You must use raw pointers for such cases, and only convert to references when you can guarantee validity:

```rust
// Working with potentially-invalid memory
let ptr: *const i32 = some_ffi_function();

// Check validity before creating reference
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

### Why This Matters

These rules determine what the compiler can assume, and therefore what optimizations are valid.

When C's strict aliasing allows the compiler to assume `int*` and `float*` do not alias, it can keep the int in a register across a write through the float pointer. This makes code faster but requires programmers to respect type boundaries.

When C++'s lifetime rules guarantee that an object cannot be accessed outside its lifetime, the compiler can elide redundant initializations and reuse storage more aggressively. But it requires programmers to manage lifetime correctly.

When Rust's validity invariants guarantee that every `bool` is 0 or 1, the compiler can use that bit pattern assumption in code generation. A match expression on a `bool` need not generate a default case for values other than `true` and `false`. But it requires that unsafe code never produce invalid values.

The progression is one of increasing commitment. C says: *these bytes have a type, access them correctly.* C++ adds: *this object exists during this window, do not touch it outside.* Rust adds: *this value is valid, it must satisfy its type's invariants at all times.*

Each layer shifts responsibility. C trusts the programmer entirely. C++ automates some lifetime management through RAII but trusts programmers with placement new and explicit destruction. Rust verifies aliasing and lifetime at compile time, leaving only validity invariants to unsafe code.

The next section examines where these objects live: storage duration and the distinction between stack, heap, and static allocation.
Pronto per la sezione 3?