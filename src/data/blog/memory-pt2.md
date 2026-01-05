---
author: Luca Lombardo
pubDatetime: 2025-12-24T00:00:00.000Z
title: "Who Owns the Memory? Part 2: Who Calls Free?"
slug: who-owns-the-memory-pt2
featured: false
draft: false
tags:
  - Rust
  - Programming
description: "Destructors, RAII, and the ownership question."
---

In the [first part](https://lukefleed.xyz/posts/who-owns-the-memory-pt1/) of this series, we saw that objects occupy storage, that storage has duration, and that the type system imposes structure on raw bytes. But we have sidestepped a question that dominates systems programming in practice: when heap-allocated memory must be released, who bears responsibility for releasing it?

You can discuss this article on [Lobsters](https://lobste.rs/s/n9tev4/who_owns_memory_part_2_who_calls_free) and [Hacker News](https://news.ycombinator.com/item?id=46487745)

## Table of Contents

## Ownership: Who Calls Free?

The stack is self-managing. When a function returns, the stack pointer moves and automatic storage vanishes. There is no decision to make, no function to call, no possibility of error. The heap is different. Memory obtained through `malloc` persists until someone calls `free`. The allocator cannot know when we are finished with an allocation; only our program logic knows that. And so the burden falls on us.

This burden carries severe consequences, often exploitable ones. If we free too early, subsequent accesses read garbage or, worse, data that a new allocation has placed there (a use-after-free, the vulnerability class behind a substantial fraction of remote code execution exploits). If we free twice, we corrupt the allocator's metadata; an attacker who controls the timing can often leverage this into arbitrary write primitives. If we never free, we leak, and the process grows until the operating system intervenes.

The question is how programming languages help us manage this responsibility, or whether they help at all.

### C: Manual Management

C offers two primitives: `malloc` to acquire and `free` to release. Everything else is convention.

Consider a function that opens a file and allocates a read buffer:

```c
typedef struct {
    int fd;
    char *buffer;
    size_t capacity;
} FileReader;

FileReader *filereader_open(const char *path, size_t buffer_size) {
    FileReader *reader = malloc(sizeof(FileReader));
    if (!reader) return NULL;

    reader->buffer = malloc(buffer_size);
    if (!reader->buffer) {
        free(reader);
        return NULL;
    }

    reader->fd = open(path, O_RDONLY);
    if (reader->fd < 0) {
        free(reader->buffer);
        free(reader);
        return NULL;
    }

    reader->capacity = buffer_size;
    return reader;
}
```

We acquire three resources: the struct itself, the buffer, and the file descriptor. Each acquisition can fail, and each failure path must release everything acquired before it. The code above handles this correctly. But observe the shape: the cleanup logic mirrors the acquisition logic in reverse, and we must write it by hand for every function that acquires resources.

The corresponding cleanup function:

```c
void filereader_close(FileReader *reader) {
    if (!reader) return;
    if (reader->fd >= 0) close(reader->fd);
    free(reader->buffer);
    free(reader);
}
```

What happens if we call `filereader_close` twice on the same pointer? The first call closes the descriptor and frees the memory. The second call passes the same address to `free`, corrupting the allocator's free list. The compiler does not warn us.

What happens if, between `filereader_open` and `filereader_close`, we reassign `reader->buffer` without freeing the old buffer? We leak. The original allocation becomes unreachable.

The fundamental problem is that C pointers carry no ownership semantics. When a function declares `void process(FileReader *reader)`, nothing in that signature tells us whether `process` will free the reader, expects us to free it afterward, or assumes it will remain valid for some longer duration. The type `FileReader *` means only "address of a FileReader"; it says nothing about responsibility.

Large C codebases develop conventions to manage this. The Linux kernel uses reference counting for shared structures, with `_get` and `_put` suffixes indicating acquisition and release. GLib uses `_new` for allocation, `_free` for deallocation, and `_ref`/`_unref` for reference-counted objects. These conventions work, but they are conventions, patterns enforced by code review rather than by the compiler. Every violation is a latent bug.

### C++: Binding Cleanup to Scope

C++ exploits a property that C has but does not leverage: local variables have a well-defined scope, and when that scope ends, the variable ceases to exist. The language allows us to attach custom cleanup logic to that moment through destructors.

A destructor is a special member function, denoted `~ClassName()`, that the compiler calls automatically when an object's lifetime ends. The call is not optional. It happens regardless of how control flow exits the scope: normal return, early return, exception propagation.

```cpp
class FileReader {
public:
    explicit FileReader(const char *path, size_t buffer_size)
        : buffer_(new char[buffer_size])
        , capacity_(buffer_size)
        , fd_(::open(path, O_RDONLY))
    {
        if (fd_ < 0) {
            delete[] buffer_;
            throw std::system_error(errno, std::generic_category());
        }
    }

    ~FileReader() {
        if (fd_ >= 0) ::close(fd_);
        delete[] buffer_;
    }

    FileReader(const FileReader &) = delete;
    FileReader &operator=(const FileReader &) = delete;

private:
    char *buffer_;
    size_t capacity_;
    int fd_;
};
```

When we write:

```cpp
void process_file(const char *path) {
    FileReader reader(path, 4096);
    do_something(reader);
}
```

the compiler generates an implicit call to `~FileReader()` at the closing brace, regardless of whether `do_something` returns normally or throws. We did not write this call; the language guarantees it.

The destruction sequence is precise. After executing the destructor body and destroying any automatic objects declared within it, the compiler destroys all non-static data members in reverse order of their declaration, then all direct base classes in reverse order of construction. This reverse ordering matters: later-declared members may depend on earlier ones, so we tear down in the opposite order we built up.

Consider what this means for exception safety. If any operation between resource acquisition and release throws, the destructor still runs during stack unwinding. We do not need explicit cleanup code on every exit path. The resource management logic is written once, in the destructor, and the compiler inserts the call at every point where it is needed.

The standard library provides RAII wrappers for common resources: `std::unique_ptr` for exclusive ownership of heap memory, `std::shared_ptr` for reference-counted shared ownership, `std::lock_guard` for mutexes, `std::fstream` for files. Using these types, we rarely write `new` or `delete` directly.

But RAII in C++ is opt-in. Nothing prevents us from writing:

```cpp
void leaky() {
    int *p = new int[1000];
    // forgot delete[]
}
```

Nothing prevents extracting a raw pointer and misusing it:

```cpp
void dangling() {
    int *raw;
    {
        auto owner = std::make_unique<int>(42);
        raw = owner.get();
    }  // owner destroyed here
    *raw = 10;  // use-after-free
}
```

The compiler does not track which pointers own and which merely observe. We can bypass RAII entirely with raw `new` and `delete`. We can hold raw pointers past their owners' lifetimes. We can `delete` through a base class pointer when the destructor is not virtual, which is undefined behavior even if no resources would leak, because the derived destructor never runs.

C++ gives us the machinery for safe resource management. Using that machinery is a choice the language cannot enforce. A codebase mixing raw pointers, `unique_ptr`, `shared_ptr`, and manual `new`/`delete` requires reasoning about ownership at every function boundary. The answer lives in programmers' heads, in comments, in coding standards. This is better than C, where even the machinery does not exist. But it remains insufficient to eliminate memory safety bugs from large codebases.


### Rust: Ownership in the Type System

Rust takes the RAII pattern and embeds it into the type system as a non-negotiable rule: every value has exactly one owner, and when that owner goes out of scope, the value is dropped. The compiler verifies this property statically, and no amount of programmer intent can bypass it.

Consider what happens when we allocate a vector:

```rust
fn example() {
    let v = vec![1, 2, 3];
}
```

The binding `v` owns the heap-allocated buffer. When `v` goes out of scope at the closing brace, Rust calls `drop` on the `Vec`, which deallocates the buffer. We cannot accidentally forget to free the memory.

The critical mechanism is the move. When we assign a value to another binding, ownership transfers:

```rust
let v1 = vec![1, 2, 3];
let v2 = v1;
```

After this assignment, `v1` is no longer valid. Any attempt to use it is a compile-time error. This is not a shallow copy followed by invalidation of the source, as in C++ move semantics where the moved-from object remains in a "valid but unspecified state" that we can still inspect and call methods on. In Rust, ownership transfer means the source binding ceases to exist as far as the type system is concerned. The compiler marks it as uninitialized. There is no moved-from state.

Why does this matter? Consider what would happen without this rule. If both `v1` and `v2` were valid after the assignment, both would attempt to free the same buffer when they went out of scope. The move rule makes double-free impossible: exactly one binding owns the allocation at any time, and exactly one drop occurs.

The same logic applies to function calls. When we pass a value to a function, ownership transfers to the function's parameter:

```rust
fn consume(v: Vec<i32>) {
    // v is owned here; dropped at end of function
}

fn main() {
    let data = vec![1, 2, 3];
    consume(data);
    // data is no longer valid here
}
```

The function signature `fn consume(v: Vec<i32>)` declares that `consume` takes ownership. The caller cannot use `data` after the call because ownership has moved. Compare this to `fn borrow(v: &Vec<i32>)`, which borrows without taking ownership, or `fn mutate(v: &mut Vec<i32>)`, which borrows mutably. The type encodes the ownership relationship.

#### The Drop Trait and Recursive Destruction

When a value goes out of scope, Rust runs its destructor. For types that implement the `Drop` trait, this means calling the `drop` method:

```rust
struct FileHandle {
    fd: i32,
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd); }
    }
}
```

The `drop` method receives `&mut self`, not `self`. We cannot move out of `self` during drop because drop takes a mutable reference. This prevents us from returning the inner file descriptor to avoid closing it. When `drop` returns, the value is gone.

After `drop` executes, Rust recursively drops all fields of the struct. This is automatic and unavoidable. If we have:

```rust
struct Connection {
    socket: TcpStream,
    buffer: Vec<u8>,
}
```

and `Connection` does not implement `Drop`, Rust still drops `socket` and `buffer` when a `Connection` goes out of scope. The compiler generates this "drop glue" for every type that needs it. We do not write boilerplate to drop children; the language handles it. If `Connection` does implement `Drop`, Rust first calls our `drop` method, then drops the fields. We cannot prevent the recursive field drops. After our `drop` returns, the fields will be dropped regardless of what we did.

The destruction order is deterministic and specified by the language. For local variables, drop order is the reverse of declaration order: later declarations are dropped first. The rationale is that later variables may hold references to earlier ones, so we must destroy the borrowers before the borrowed. For struct fields, drop order is declaration order (not reversed). For tuples, elements drop in order. For arrays, elements drop from index 0 to the end. For enums, only the active variant's fields are dropped. Closure captures by move are dropped in an unspecified order, so if destruction order among captured values matters, do not rely on closure drop order.

#### ManuallyDrop and Suppressing Automatic Destruction

The recursive drop behavior creates a problem when we need fine-grained control over destruction. Consider a type that wraps a `Box` and wants to deallocate the contents in a custom way:

```rust
struct SuperBox<T> {
    my_box: Box<T>,
}

impl<T> Drop for SuperBox<T> {
    fn drop(&mut self) {
        unsafe {
            // Deallocate the box's contents ourselves
            let ptr = Box::into_raw(self.my_box);  // ERROR: cannot move out of &mut self
            std::alloc::dealloc(ptr as *mut u8, Layout::new::<T>());
        }
    }
}
```

This does not compile. We cannot move `self.my_box` out of `&mut self`. And even if we could, after our `drop` returns, Rust would try to drop `my_box` again, causing a double-free.

One solution is `Option`:

```rust
struct SuperBox<T> {
    my_box: Option<Box<T>>,
}

impl<T> Drop for SuperBox<T> {
    fn drop(&mut self) {
        if let Some(b) = self.my_box.take() {
            // Handle b ourselves; self.my_box is now None
            // When Rust drops self.my_box, it drops None, which does nothing
        }
    }
}
```

This works, but it pollutes our type with `Option` semantics. A field that should always be `Some` must be declared as `Option` solely because of what happens in the destructor. Every access to `my_box` elsewhere in the code must handle the `None` case that should never occur outside of `drop`.

`ManuallyDrop<T>` provides a cleaner solution. It is a wrapper that suppresses automatic drop for its contents:

```rust
use std::mem::ManuallyDrop;

struct SuperBox<T> {
    my_box: ManuallyDrop<Box<T>>,
}

impl<T> Drop for SuperBox<T> {
    fn drop(&mut self) {
        unsafe {
            // Take ownership of the inner Box
            let b = ManuallyDrop::take(&mut self.my_box);
            // Now we own b and can do whatever we want
            // When our drop returns, Rust will "drop" self.my_box,
            // but ManuallyDrop's drop is a no-op
        }
    }
}
```

`ManuallyDrop<T>` has the same size and alignment as `T`. It implements `Deref` and `DerefMut`, so we can use the wrapped value normally. But when Rust drops a `ManuallyDrop<T>`, nothing happens. The inner `T` is not dropped. We are responsible for dropping it manually via `ManuallyDrop::drop(&mut x)` or taking ownership via `ManuallyDrop::take(&mut x)`.

This is useful beyond custom destructors. When we need to move a value out of a context where Rust would normally drop it, `ManuallyDrop` lets us suppress that drop and handle the value ourselves. The `unsafe` is required because we are taking responsibility for ensuring the value is eventually dropped (or intentionally leaked).

#### Drop Flags and Conditional Initialization

Rust tracks initialization state at compile time when possible. But consider:

```rust
let x;
if condition {
    x = Box::new(0);
}
```

At the end of scope, should Rust drop `x`? It depends on whether `condition` was true. When the compiler cannot determine initialization state statically, it inserts a runtime drop flag: a hidden boolean that tracks whether the value was initialized. At scope exit, Rust checks the flag before calling drop.

For straight-line code and branches that initialize consistently, the compiler can eliminate these flags through static analysis. The flags exist only when genuinely necessary, and the generated code checks them only at points where the initialization state is ambiguous.

#### You Cannot Call Drop Directly

Rust prevents explicit calls to the `Drop::drop` method:

```rust
let v = vec![1, 2, 3];
v.drop();  // error: explicit use of destructor method
```

If we could call `drop` directly, the value would still be in scope afterward. When the scope ends, Rust would call `drop` again. Instead, we use `std::mem::drop` for early cleanup:

```rust
let v = vec![1, 2, 3];
drop(v);  // v is moved into drop() and dropped there
```

The `drop` function takes ownership by value: `fn drop<T>(_: T) {}`. The value moves into the function and is dropped when the function returns. Since ownership moved, the original binding is invalid and no second drop occurs.

#### Drop Check: When Lifetimes and Destructors Collide

We now arrive at a subtle interaction between Rust's lifetime system and its destructor semantics. Consider this seemingly innocuous code:

```rust
struct Inspector<'a>(&'a u8);

struct World<'a> {
    inspector: Option<Inspector<'a>>,
    days: Box<u8>,
}

fn main() {
    let mut world = World {
        inspector: None,
        days: Box::new(1),
    };
    world.inspector = Some(Inspector(&world.days));
}
```

This compiles. The `Inspector` holds a reference to `days`, both are fields of `World`, and when `world` goes out of scope, both are dropped. The fact that `days` does not strictly outlive `inspector` does not matter here because neither has a destructor that could observe the other.

But watch what happens when we add a `Drop` impl to `Inspector`:

```rust
struct Inspector<'a>(&'a u8);

impl<'a> Drop for Inspector<'a> {
    fn drop(&mut self) {
        println!("I was only {} days from retirement!", self.0);
    }
}

struct World<'a> {
    inspector: Option<Inspector<'a>>,
    days: Box<u8>,
}

fn main() {
    let mut world = World {
        inspector: None,
        days: Box::new(1),
    };
    world.inspector = Some(Inspector(&world.days));
}
```

This does not compile:

```
error[E0597]: `world.days` does not live long enough
```

What changed? The `Drop` implementation. When `Inspector` has a destructor, that destructor might access the reference it holds. If `days` is dropped before `inspector`, the destructor would dereference freed memory. The borrow checker must now enforce that any data borrowed by `Inspector` outlives the `Inspector` itself, because the destructor could observe that data.

This is the drop checker (dropck). It enforces a stricter rule when types have destructors: for a generic type to soundly implement drop, its generic arguments must strictly outlive it. Not _live at least as long as_, but _strictly outlive_. The difference is subtle. Without a destructor, two values can go out of scope simultaneously because neither observes the other during destruction. With a destructor, the type with the destructor might observe its borrowed data, so that data must still be valid when the destructor runs.

Only generic types need to worry about this. If a type is not generic, the only lifetimes it can contain are `'static`, which truly lives forever. The problem arises when a type is generic over a lifetime or a type parameter, and it implements `Drop`, and that `Drop` could potentially access the borrowed data.

#### The `#[may_dangle]` Escape Hatch

The drop checker is conservative. It assumes that any `Drop` impl for a generic type might access data of the generic parameter. But this is often not true. Consider `Vec<T>`:

```rust
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        // Deallocate the buffer
        // We DO drop each T element, but we don't "use" T in the sense
        // of accessing references that T might contain
    }
}
```

When we drop a `Vec<&'a str>`, we deallocate the buffer. We do call drop on each `&'a str` element, but `&'a str` has no destructor; its drop is a no-op. The `Vec`'s drop does not actually dereference those `&'a str` values. It does not read the strings. It just deallocates the backing memory.

Yet the drop checker does not know this. It sees `impl<'a> Drop for Vec<&'a str>` and concludes that `'a` must strictly outlive the `Vec`. This prevents code like:

```rust
fn main() {
    let mut v: Vec<&str> = Vec::new();
    let s: String = "Short-lived".into();
    v.push(&s);
    drop(s);  // s dropped while v still holds a reference
}
```

This is correctly rejected. But it also rejects:

```rust
fn main() {
    let mut v: Vec<&str> = Vec::new();
    let s: String = "Short-lived".into();
    v.push(&s);
}  // s and v dropped here, but in what order?
```

The second example should be fine. Both `v` and `s` are dropped at the end of `main`. Variables are dropped in reverse declaration order, so `v` drops first, then `s`. By the time `s` drops, `v` is already gone. The references in `v` are never dereferenced during `v`'s destruction.

To allow this pattern, Rust provides an unstable, unsafe attribute: `#[may_dangle]`. It tells the drop checker "I promise not to access this generic parameter in my destructor":

```rust
unsafe impl<#[may_dangle] T> Drop for Vec<T> {
    fn drop(&mut self) {
        // ...
    }
}
```

The `#[may_dangle]` on `T` is a promise that the `Drop` impl does not access `T` values in a way that requires them to be valid. The drop checker relaxes its requirements: `T` may now dangle (be a reference to freed memory) when the `Vec` is dropped.

But this is a lie, or at least an incomplete truth. `Vec<T>`'s destructor does drop each `T` element. If `T` has a destructor that accesses borrowed data, that data must still be valid. The promise is more precisely: "I do not access `T` myself, but I may trigger `T`'s destructor." This distinction matters for the soundness of the overall system.

#### PhantomData and Drop Check Interaction

When we use `#[may_dangle]`, we are opting out of the drop checker's conservative assumptions. But we must opt back in for cases where we do transitively drop `T`. This is where `PhantomData` enters the picture.

Consider the actual implementation of `Vec`:

```rust
struct Vec<T> {
    ptr: *const T,  // raw pointer, no ownership semantics
    len: usize,
    cap: usize,
}
```

A raw pointer `*const T` does not imply ownership. The drop checker does not assume that `Vec` owns `T` values just because it contains a `*const T`. If we write:

```rust
unsafe impl<#[may_dangle] T> Drop for Vec<T> {
    fn drop(&mut self) {
        // drop elements, deallocate buffer
    }
}
```

we have told the drop checker that `T` may dangle. But `Vec` does own `T` values and does drop them. If `T` is `PrintOnDrop<'a>` (a type with a destructor that dereferences `'a`), then `'a` must be valid when `Vec` drops, because `Vec` will invoke `T`'s destructor.

We need to tell the drop checker: "`T` may dangle, except if `T` itself has drop glue that would observe borrowed data." The mechanism for this is `PhantomData<T>`:

```rust
use std::marker::PhantomData;

struct Vec<T> {
    ptr: *const T,
    len: usize,
    cap: usize,
    _marker: PhantomData<T>,
}
```

`PhantomData<T>` is a zero-sized type that tells the compiler "act as if this struct owns a `T`." It affects variance, auto-trait inference, and critically, drop check. When the drop checker sees `PhantomData<T>` in a struct, it knows that dropping the struct may involve dropping `T` values.

The interaction is:

1. `#[may_dangle] T` on the `Drop` impl says "I promise not to access `T` directly in my destructor."
2. `PhantomData<T>` in the struct says "but I do own `T` values and will drop them."
3. The drop checker combines these: `T` itself may dangle (its references may be invalid), but if `T` has drop glue, that glue will run, and whatever `T`'s glue accesses must still be valid.

This is subtle. Suppose `T` is `&'a str`. The type `&'a str` has no destructor (dropping a reference is a no-op). So `PhantomData<&'a str>` does not introduce additional constraints. The `#[may_dangle]` applies fully, and `'a` may dangle.

Now suppose `T` is `PrintOnDrop<'a>`, a type with a destructor that dereferences `'a`. The `PhantomData<PrintOnDrop<'a>>` tells the drop checker that `Vec` will drop `PrintOnDrop<'a>` values. The `#[may_dangle]` says `Vec` itself won't access `'a`. But dropping `PrintOnDrop<'a>` will access `'a`. So `'a` must still be valid when `Vec` is dropped, despite the `#[may_dangle]`.

The rules compose correctly. The `#[may_dangle]` is not a blanket permission to let everything dangle. It is permission for the specific `Drop` impl to not access the parameter, combined with the `PhantomData` indicating that transitive drops may still occur.

Without `#[may_dangle]`, implementing a collection like `Vec` would be unnecessarily restrictive. Every `Vec<&'a T>` would require `'a` to strictly outlive the `Vec`, even when the `Vec` is dropped before the borrowed data goes out of scope. The combination of `#[may_dangle]` and `PhantomData` allows the standard library to express the precise ownership semantics: "we will drop `T` values, but we do not otherwise observe them in our destructor."

#### Leaks Are Safe

Here we encounter a design decision that surprises many: leaking memory is safe in Rust. The function `std::mem::forget` takes ownership of a value and does not run its destructor:

```rust
let v = vec![1, 2, 3];
std::mem::forget(v);
// v's destructor never runs; the heap allocation is leaked
```

This is a safe function. It does not require `unsafe`. The reasoning is that _safe_ in Rust means _cannot cause undefined behavior_. Leaking memory is wasteful, but it does not corrupt memory, create dangling pointers, or cause data races. A program that leaks is buggy but not undefined.

Moreover, leaks can occur in safe code without calling `forget`. The simplest example is a reference cycle with `Rc`:

```rust
use std::cell::RefCell;
use std::rc::Rc;

struct Node {
    next: Option<Rc<RefCell<Node>>>,
}

fn create_cycle() {
    let a = Rc::new(RefCell::new(Node { next: None }));
    let b = Rc::new(RefCell::new(Node { next: Some(a.clone()) }));
    a.borrow_mut().next = Some(b.clone());
}
```

When `create_cycle` returns, the `Rc`s go out of scope, but each has a reference count of 2 due to the cycle. The count decrements to 1, not 0. The nodes are never freed. This is safe code with no `unsafe` blocks, and it leaks.

Since leaks are possible in safe code, the language cannot assume destructors always run. This has profound implications for unsafe code. Any type whose safety invariants depend on the destructor running is unsound in the presence of `mem::forget` or cycles.

The standard library learned this lesson with `thread::scoped`, an API that allowed spawning threads referencing stack data. It relied on a guard whose destructor joined the thread:

```rust
pub fn scoped<'a, F>(f: F) -> JoinGuard<'a>
where F: FnOnce() + Send + 'a
```

The guard's lifetime tied it to the borrowed data. When the guard dropped, it joined the thread, ensuring the thread finished before the data went out of scope. But `mem::forget(guard)` prevented the destructor from running. The thread would continue executing while the stack data it referenced was freed. Use-after-free from safe code. The API was unsound and had to be removed.

The correct design principle is that unsafe code cannot rely on destructors running to maintain safety invariants. Safe abstractions must account for the possibility that destructors are skipped. The standard library's `Vec::drain` is a good example. Draining a vector moves elements out one at a time. If the drain iterator is forgotten mid-iteration, some elements have been moved out and the vector's length is wrong. Rather than leaving the vector in an inconsistent state, `Drain` sets the vector's length to zero at the start of iteration. If `Drain` is forgotten, the remaining elements leak (their destructors do not run, their memory is not reused), but the vector is in a valid state (empty). Leaks amplify leaks, but undefined behavior does not occur.


## Beyond RAII

We have built a comprehensive picture of ownership. Rust makes ownership tracking mandatory and statically verified. Yet even Rust's model has limits.

### The Single-Destructor Problem

The RAII model, whether in C++ or Rust, shares a structural limitation: the destructor is a single, parameterless function that returns nothing. When an object goes out of scope, exactly one action occurs. We cannot choose between alternatives. We cannot pass runtime information to the cleanup logic. We cannot receive a result from it.

For many resources this constraint is invisible. A file handle has one sensible cleanup action: close the descriptor. A heap allocation has one cleanup action: free the memory. A mutex guard has one cleanup action: unlock the mutex. The destructor does the obvious thing, and the single-destructor model works well.

But consider a database transaction. A transaction must eventually either commit (make changes permanent) or rollback (discard changes). These are fundamentally different operations with different semantics, different failure modes, and often different parameters. A commit might require a priority level. A rollback might need to log the reason for abandonment. The destructor cannot accommodate this. It must pick one.

The standard workaround in C++ and Rust is to default to rollback and provide an explicit `commit` method:

```cpp
class Transaction {
public:
    explicit Transaction(Database& db) : db_(db), committed_(false) {
        db_.begin();
    }

    void commit() {
        db_.commit();
        committed_ = true;
    }

    ~Transaction() {
        if (!committed_) {
            db_.rollback();
        }
    }

private:
    Database& db_;
    bool committed_;
};
```

We call `commit()` when the transaction succeeds and let the destructor rollback on error paths or early returns. Exception safety falls out naturally; if an exception propagates, the destructor runs, and uncommitted transactions rollback.

The problem is that forgetting to call `commit()` is not a compile-time error. If we write a function that successfully completes its work but neglects to call `commit()`, the destructor happily rolls back. The program is wrong, but the compiler cannot tell us. We have traded one category of bug (forgetting to cleanup) for another (forgetting to finalize). The second category is arguably more insidious because the cleanup happens, silently doing the wrong thing.

Rust's ownership system does not help here:

```rust
struct Transaction<'a> {
    db: &'a mut Database,
    committed: bool,
}

impl<'a> Transaction<'a> {
    fn commit(mut self) {
        self.db.commit();
        self.committed = true;
    }
}

impl<'a> Drop for Transaction<'a> {
    fn drop(&mut self) {
        if !self.committed {
            self.db.rollback();
        }
    }
}
```

The `commit` method takes `self` by value, consuming the transaction. After calling `commit`, the binding is gone. But if we never call `commit`, the transaction goes out of scope, `drop` runs, and we rollback. No compiler error. The type system tracked ownership but not obligation.

### Defer: Explicit Scope-Bound Cleanup

Some languages take a different approach. Rather than binding cleanup to object destruction, they provide explicit defer statements that execute at scope exit.

Zig's `defer` runs an expression unconditionally when control leaves the enclosing block:

```zig
fn processFile(path: []const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const buffer = try allocator.alloc(u8, 4096);
    defer allocator.free(buffer);

    // work with file and buffer
    // both are cleaned up when we exit, regardless of how
}
```

The cleanup code sits immediately after the acquisition code. We see allocation and deallocation together, which aids comprehension. Zig extends this with `errdefer`, which executes only if the function returns an error, separating error-path cleanup from success-path transfer.

The defer model has a structural limitation that RAII does not: the cleanup is scope-bound. We cannot return a resource to our caller the way we return an RAII object. The defer runs when the current scope exits, period. RAII is more flexible; we can return an RAII object, store it in a data structure, transfer ownership to another thread. The cleanup travels with the object. Defer is local; RAII is transferable.

But defer has an advantage in simplicity. We do not need to define a type, implement a trait, worry about drop order among struct fields. We write the cleanup code inline, at the point of acquisition. For resources that do not leave the current function, defer is often cleaner.

### Linear Types: Enforcing Explicit Consumption

The transaction example shows a gap in RAII's guarantees. RAII ensures that cleanup happens, but not that we made an explicit decision about *which* cleanup to perform. The destructor chooses for us, silently.

Linear types close this gap. A linear type must be explicitly consumed; it cannot simply go out of scope. If we try to let a linear value fall out of scope without passing it to a consuming function, the compiler rejects the program.

Consider a hypothetical extension to Rust:

```rust
// Hypothetical syntax
#[linear]
struct Transaction { db: Database }

fn commit(txn: Transaction) -> Result<(), Error> {
    txn.db.commit()?;
    destruct txn;  // explicitly consume
    Ok(())
}

fn rollback(txn: Transaction, reason: &str) {
    txn.db.rollback();
    destruct txn;  // explicitly consume
}

fn do_work(db: Database) {
    let txn = Transaction { db };
    // ERROR: `txn` goes out of scope without being consumed
    // must call either commit(txn) or rollback(txn, ...)
}
```

The transaction cannot be ignored. Forgetting to decide is a compile-time error, not a runtime silent rollback. Languages with linear types (Vale, Austral, and to some extent Haskell with LinearTypes) can express patterns that RAII cannot.

### Why Rust Does Not Have Linear Types

Rust's types are affine, not linear. An affine type can be used *at most* once; a linear type must be used *exactly* once. The difference matters when panics unwind the stack.

Rust permits silent dropping because `Drop::drop` must always be callable. When a scope exits, when a panic unwinds, when we reassign a variable, Rust calls `drop`. The entire language assumes that any value can be dropped at any time.

Linear types would break this assumption. If a value must be explicitly consumed, what happens when a panic unwinds through a function holding that value? The unwinding code cannot know which consuming function to call, what parameters to pass, what to do with the return value. Every generic container, every iterator adapter, every function that might discard a value would need to be reconsidered.

Rust chose affine types, accepting that the compiler cannot enforce explicit consumption but gaining a simpler model where any value can be dropped. The `#[must_use]` attribute provides a weaker guarantee: a warning (not an error) if a value is unused. It catches some mistakes but does not provide the hard guarantee that linear types would.

## When Things Go Wrong

RAII ensures cleanup happens when scope ends. But it says nothing about how we *signal* failures. When a function cannot complete its task, it must communicate this to the caller, and the caller must have a chance to respond. The three languages take fundamentally different approaches to this problem, and the differences reveal deep assumptions about what error handling should look like.


### C: Return Codes and errno

C library functions indicate failure through their return values, but the convention varies by function and must be looked up individually. The C standard defines several patterns. Functions like `fopen` return a null pointer on failure; the null serves as an out-of-band sentinel that cannot be confused with a valid result. Functions like `puts` and `fclose` return `EOF`, a special error code, on failure. Functions like `fgetpos` and `fsetpos` return a nonzero value on failure, where zero indicates success and the actual return value is otherwise unneeded. Functions like `thrd_create` return a special success code, and any other value indicates a specific failure condition. Functions like `printf` return a negative value on failure, where success produces a positive count.

```c
if (puts("hello world") == EOF) {
    perror("can't output to terminal:");
    exit(EXIT_FAILURE);
}
```

The `perror` function prints a diagnostic message based on the current value of `errno`, a thread-local variable that library functions set when they fail. The combination of checking the return value and consulting `errno` provides both detection and diagnosis. But `errno` is fragile. It must be checked immediately after the failing call. Any intervening function call, even a successful one, may modify `errno`. If we need to preserve error information across other operations, we must copy it:

```c
FILE *f = fopen(path, "r");
if (!f) {
    int saved = errno;
    log_message("fopen failed");   // might modify errno
    errno = saved;
    return -1;
}
```

The more serious problem is propagation. Consider a function that opens a file, allocates a buffer, reads data, and processes it. Each step can fail. Each failure requires releasing resources acquired in previous steps:

```c
int process_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char *buf = malloc(4096);
    if (!buf) {
        fclose(f);
        return -1;
    }

    if (fread(buf, 1, 4096, f) == 0 && ferror(f)) {
        free(buf);
        fclose(f);
        return -1;
    }

    // process buf...

    free(buf);
    fclose(f);
    return 0;
}
```

The cleanup code is duplicated at each error site. If we add a fourth resource, we must update three error paths. The duplication invites mistakes. Miss one cleanup and we leak.

The `goto` statement consolidates cleanup into a single location:

```c
int process_file(const char *path) {
    int result = -1;
    FILE *f = NULL;
    char *buf = NULL;

    f = fopen(path, "r");
    if (!f) goto cleanup;

    buf = malloc(4096);
    if (!buf) goto cleanup;

    if (fread(buf, 1, 4096, f) == 0 && ferror(f)) goto cleanup;

    // process buf...
    result = 0;

cleanup:
    free(buf);
    if (f) fclose(f);
    return result;
}
```

All resources must be initialized to null at the top. The cleanup block must handle partially-initialized state; `free(NULL)` is defined to do nothing, but `fclose(NULL)` is undefined behavior, hence the explicit check. The pattern requires discipline. Reviewers must verify that every resource acquired before a `goto` is released in the cleanup block, and that the cleanup block handles every possible partial state.

For errors that cannot be handled locally, C provides `setjmp` and `longjmp`. The `setjmp` macro marks a location in the code; `longjmp` transfers control directly to that location, unwinding the call stack without executing intervening code. This is C's mechanism for non-local jumps, used when an error deep in a call chain must abort the entire operation:

```c
#include <setjmp.h>

jmp_buf error_handler;

void deep_function(void) {
    if (catastrophic_failure()) {
        longjmp(error_handler, 1);  // never returns
    }
}

int main(void) {
    if (setjmp(error_handler) != 0) {
        // longjmp landed here
        fprintf(stderr, "operation aborted\n");
        return EXIT_FAILURE;
    }

    deep_function();
    return EXIT_SUCCESS;
}
```

The `longjmp` function never returns to its caller. Control transfers directly to the `setjmp` site as if `setjmp` had just returned, but with a nonzero value indicating which `longjmp` triggered. This bypasses all intermediate stack frames. Local variables in those frames are not cleaned up; their destructors (in the C sense of cleanup code we might have written) do not run. Resources acquired between `setjmp` and `longjmp` leak unless we manually track and release them. The `jmp_buf` must remain valid when `longjmp` is called; if the function containing `setjmp` has returned, the behavior is undefined.

An `int` return code provides limited information. We can distinguish success from failure, and `errno` codes like `ENOENT` or `EACCES` give some context, but rich error information requires either out-parameters, custom error structs, or global state.


### C++: Exceptions and Hidden Control Flow

C++ exceptions address the propagation problem directly. When a function cannot fulfill its contract, it executes `throw` with an exception object. Control transfers immediately to the nearest enclosing `catch` block whose type matches the thrown object, skipping all intermediate code but calling destructors along the way.

```cpp
std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open: " + path);
    }
    std::stringstream buf;
    buf << f.rdbuf();
    return buf.str();
}

void process() {
    try {
        auto content = read_file("data.txt");
        // use content...
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << '\n';
    }
}
```

The `try` block delimits code where exceptions may be caught. The `catch` clause specifies a type; if the thrown object's type matches (including derived classes), that handler executes. Multiple `catch` clauses can handle different exception types; the runtime tries them in order and executes the first match. Catching by const reference avoids copying and preserves polymorphism for exception hierarchies.

When an exception propagates through a scope, the compiler ensures that destructors run for all local objects in reverse order of construction. This is *stack unwinding*. RAII objects release their resources automatically on error paths:

```cpp
void safe_operation() {
    auto resource = std::make_unique<Widget>();
    might_throw();
}   // resource destroyed whether or not might_throw() throws
```

The destructor runs regardless of how we exit the scope. We wrote no cleanup code; the compiler inserted it.

Here lies the problem with exceptions: *every function call is a potential exit point*. In the code above, if `might_throw()` throws, control leaves `safe_operation` immediately. This is convenient when we want automatic cleanup. It is treacherous when we are maintaining invariants across multiple operations.

```cpp
void transfer(Account& from, Account& to, int amount) {
    from.withdraw(amount);  // might throw
    to.deposit(amount);     // might throw
}
```

If `deposit` throws after `withdraw` succeeds, the money has left `from` but never arrived at `to`. The program is in an inconsistent state. The *strong exception guarantee* (if an operation fails, state is unchanged) requires explicit effort:

```cpp
void transfer(Account& from, Account& to, int amount) {
    int withdrawn = from.withdraw(amount);  // might throw
    try {
        to.deposit(amount);
    } catch (...) {
        from.deposit(withdrawn);  // rollback
        throw;
    }
}
```

Now we are back to writing manual error handling, but with a twist: the error paths are invisible. Looking at a function that calls other functions, we cannot tell which calls might throw without examining their declarations (or their transitive callees). The C code with `goto cleanup` was verbose, but every exit point was visible in the source.

Destructors must not throw. Since C++11, destructors are implicitly `noexcept`. If a destructor throws during stack unwinding from another exception, the program has two active exceptions with no way to handle both; `std::terminate` is called. This constraint shapes class design: destructors perform only operations that cannot fail, or they catch and suppress exceptions internally.

The `noexcept` specifier declares that a function does not throw. If it does throw, `std::terminate` is called rather than unwinding. Move constructors and move assignment operators should be `noexcept` when possible; this enables `std::vector` to move elements during reallocation rather than copying them. If a move constructor might throw, the vector must copy to preserve the strong exception guarantee. Copying is dramatically slower for types with expensive copy operations. The performance of your vector depends on whether your move constructor is marked `noexcept`.

The *exception safety guarantees* classify behavior when exceptions occur. The *nothrow* guarantee means the function never throws; destructors and swap functions provide this. The *strong* guarantee means if an exception is thrown, the program state is unchanged; `std::vector::push_back` provides this. The *basic* guarantee means if an exception is thrown, the program is in a valid state with no leaks, but the state may have changed. Every function in a C++ codebase implicitly has one of these guarantees, whether the author thought about it or not.


### C++23: std::expected and the Missing Operator

C++23 introduced `std::expected<T, E>`, a vocabulary type that holds either a value of type `T` or an error of type `E`. The design is explicitly modeled on Rust's `Result` type and Haskell's `Either`.

```cpp
#include <expected>

std::expected<int, std::errc> parse_int(std::string_view s) {
    int value;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (ec != std::errc{}) {
        return std::unexpected(ec);
    }
    return value;
}
```

The `std::unexpected` wrapper distinguishes error values from success values. Callers inspect the result:

```cpp
auto result = parse_int("42");
if (result) {
    use(*result);
} else {
    handle(result.error());
}
```

C++23 added monadic operations for chaining. The `and_then` method applies a function if a value is present, where the function returns another `std::expected`. The `transform` method applies a function that returns a plain value, wrapping the result. The `or_else` method handles the error case:

```cpp
auto result = parse(input)
    .and_then(validate)
    .transform([](int n) { return n * 2; });
```

The type exists. The monadic operations exist. What C++ lacks is the operator that makes the pattern ergonomic.

Consider a function that calls two fallible operations and combines their results. With exceptions, C++ is concise:

```cpp
auto strcat(int i) -> std::string {
    return std::format("{}{}", foo(i), bar(i));
}
```

We do not even need to know that `foo` and `bar` might throw. Errors propagate invisibly.

With `std::expected`, the same logic becomes:

```cpp
auto strcat(int i) -> std::expected<std::string, E> {
    auto f = foo(i);
    if (!f) {
        return std::unexpected(f.error());
    }

    auto b = bar(i);
    if (!b) {
        return std::unexpected(b.error());
    }

    return std::format("{}{}", *f, *b);
}
```

The manual error propagation dominates the function. We give names to the `expected` objects (`f`, `b`) rather than to the values we care about. We access values through `*f` and `*b`. The ceremony obscures the logic.

Rust solves this with the `?` operator:

```rust
fn strcat(i: i32) -> Result<String, E> {
    Ok(format!("{}{}", foo(i)?, bar(i)?))
}
```

One character per fallible operation. The happy path reads as naturally as the exception version. The error path is explicit but minimal.

C++ cannot adopt Rust's `?` syntax. The conditional operator `?:` creates ambiguity: in `a ? *b ? *c : d`, the parser cannot determine whether `?` begins a conditional or is a postfix propagation operator. [P2561R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2561r0.html) proposes `??` as an alternative, but as of C++26, no propagation operator has been adopted.

The monadic operations provide a workaround:

```cpp
auto strcat(int i) -> std::expected<std::string, E> {
    return foo(i).and_then([&](int f) {
        return bar(i).transform([&](int b) {
            return std::format("{}{}", f, b);
        });
    });
}
```

This works. It is also nested, lambda-heavy, and harder to read than either the exception version or the Rust version. The ergonomics gap matters. If the safe pattern is more verbose than the dangerous pattern, programmers will choose the dangerous pattern. Codebases that adopt `std::expected` must fight against the syntax to use it consistently.

A function using `std::expected` that calls code using exceptions must catch and convert at the boundary:

```cpp
std::expected<Data, Error> safe_wrapper() {
    try {
        return legacy_function_that_throws();
    } catch (const std::exception& e) {
        return std::unexpected(Error{e.what()});
    }
}
```

A function using exceptions that calls code returning `std::expected` must check and throw:

```cpp
Data wrapper_for_exception_code() {
    auto result = modern_function();
    if (!result) {
        throw std::runtime_error(result.error().message());
    }
    return *result;
}
```

Mixing styles is awkward but inevitable. Real codebases will contain both patterns indefinitely. The boundary code is boilerplate that adds nothing but conversion overhead.


### Rust: Result and the ? Operator

Rust represents recoverable errors as values. Operations that can fail return `Result<T, E>`, and the type system requires handling both possibilities.

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_file(path: &str) -> Result<String, io::Error> {
    let mut f = File::open(path)?;
    let mut contents = String::new();
    f.read_to_string(&mut contents)?;
    Ok(contents)
}
```

The `?` operator is syntactic sugar for early return on error. Applied to a `Result`, it extracts the `Ok` value or returns the `Err` from the enclosing function. The desugaring is approximately:

```rust
match expr {
    Ok(val) => val,
    Err(e) => return Err(From::from(e)),
}
```

The `From::from(e)` call enables automatic error type conversion. A function returning `Result<T, MyError>` can use `?` on any `Result` whose error type implements `From<OriginalError> for MyError`.

On x86-64, the `?` operator compiles to a test of the discriminant and a conditional jump:

```asm
    test    eax, eax
    jne     .Lerror
```

The cost is just one comparison and one conditional branch per `?`. This is the same cost as the manual `if (!result)` check in C++, but without the syntactic overhead.

Because `Result` is marked `#[must_use]`, ignoring a return value triggers a compiler warning. Errors cannot be silently dropped. The type system forces acknowledgment of every failure mode.

The `?` operator also works on `Option<T>`:

```rust
fn get_username(id: UserId) -> Option<String> {
    let user = users.get(&id)?;
    let profile = user.profile()?;
    Some(profile.name.clone())
}
```

The same syntax handles both "might fail with error" (`Result`) and "might be absent" (`Option`). Conversion between them is explicit: `result.ok()` discards the error to produce an `Option`, while `option.ok_or(err)` attaches an error to produce a `Result`.


### Panics and Unwinding

Panics handle unrecoverable errors: violated assertions, out-of-bounds access, explicit `panic!()` calls. By default, panics unwind the stack, calling `Drop` implementations for local values. The mechanism uses platform unwinding libraries, the same infrastructure as C++ exceptions.

```rust
fn get(v: &[i32], i: usize) -> i32 {
    v[i]  // panics if i >= v.len()
}
```

Unlike exceptions, panics are not meant to be caught routinely. The `std::panic::catch_unwind` function can intercept a panic, but its purpose is FFI boundaries and thread isolation, not error handling:

```rust
use std::panic;

let result = panic::catch_unwind(|| {
    panic!("oops");
});
assert!(result.is_err());
```

A panic reaching an `extern "C"` boundary is undefined behavior. Rust code that might panic must catch the panic before calling into C. Compiling with `panic=abort` eliminates unwinding entirely; panics terminate the process immediately.

The distinction between `Result` and panic corresponds to recoverable versus unrecoverable errors. A missing file is recoverable: try a different path, prompt the user, report and continue. An index out of bounds is a bug: assumptions are violated, continuing risks corruption. Recoverable errors return `Result`. Bugs panic. The type system encodes this distinction.


### Exception Safety Through Control Flow

The `?` operator achieves exception safety through normal control flow. When `?` encounters an error, it returns from the function. RAII objects drop as part of that return. The compiler generates the same cleanup code for early returns via `?` as for normal returns.

```rust
fn safe_operation() -> Result<(), Error> {
    let resource = acquire()?;
    might_fail()?;
    Ok(())
}
```

If `might_fail()` returns `Err`, the function returns early. Before returning, `resource` drops. This happens through the normal return path, not unwinding. The generated code is a conditional branch to the function epilogue.

Every potential exit point is visible in the source. The `?` marks where control might leave. There are no hidden exits, no invisible propagation. We can read the function and see exactly where early returns occur. The C++ exception model hides exit points; the Rust model makes them explicit while keeping them syntactically light.

Rust code achieves the basic exception safety guarantee automatically: early returns via `?` release all local resources. The strong guarantee requires the same care as in C++; modifying state before a fallible operation requires rollback logic on error. But the explicit control flow makes reasoning about state easier.


### Panic Safety in Unsafe Code

Safe Rust cannot violate memory safety through panics. Unsafe code can create intermediate states that violate invariants; if a panic occurs in that window, destructors may observe invalid state.

Consider extending a vector by cloning elements from a slice:

```rust
impl<T: Clone> Vec<T> {
    unsafe fn extend_unchecked(&mut self, src: &[T]) {
        self.reserve(src.len());
        let old_len = self.len();
        self.set_len(old_len + src.len());  // length updated first

        for (i, x) in src.iter().enumerate() {
            // clone() might panic!
            self.as_mut_ptr().add(old_len + i).write(x.clone());
        }
    }
}
```

If `clone()` panics after `set_len`, the vector claims more initialized elements than exist. When it drops, it calls `drop` on uninitialized memory. The fix is to update length *after* initialization, or use a guard:

```rust
struct SetLenOnDrop<'a, T> {
    vec: &'a mut Vec<T>,
    len: usize,
}

impl<T> Drop for SetLenOnDrop<'_, T> {
    fn drop(&mut self) {
        unsafe { self.vec.set_len(self.len); }
    }
}

impl<T: Clone> Vec<T> {
    fn extend_from_slice(&mut self, src: &[T]) {
        self.reserve(src.len());
        let mut guard = SetLenOnDrop {
            vec: self,
            len: self.len()
        };

        for x in src {
            unsafe {
                self.as_mut_ptr().add(guard.len).write(x.clone());
            }
            guard.len += 1;  // only increment after successful write
        }

        std::mem::forget(guard);  // don't run Drop, length is correct
    }
}
```

The guard tracks successfully initialized elements. If `clone()` panics, the guard drops, setting the vector's length to the count of *actually* initialized elements. The invariant holds across the panic. Safe Rust cannot violate memory safety through panics. Unsafe code must maintain invariants across every potential panic point.




## Move Semantics

All this discussion about ownership assumes that ownership can be transferred efficiently. When we say a value "moves" from one binding to another, what actually happens to the bytes? And why is this operation preferable to copying?

The answer varies dramatically across C, C++, and Rust, and the differences expose fundamental assumptions each language makes about values, identity, and resources.

### The Cost of Copies

Consider what happens when we pass a value to a function. In the simplest model, the caller's value is copied into the callee's parameter. For a 32-bit integer, this means copying 4 bytes, negligible. But what about a dynamically-sized container?

A `std::vector<int>` in C++ or a `Vec<i32>` in Rust has a particular structure: a pointer to heap-allocated storage, a length, and a capacity. The struct itself is small (typically 24 bytes on 64-bit systems), but it can own a large heap allocation.

```cpp
struct Vec {
    int* data;     // 8 bytes
    size_t len;    // 8 bytes
    size_t cap;    // 8 bytes
};
// sizeof(Vec) == 24, but data might point to way more memory
```

If we copy this struct byte-for-byte, we produce two `Vec` instances pointing to the same heap allocation. This is a shallow copy. It creates a problem: both instances believe they own the memory. When one destructor frees it, the other's pointer dangles. When the second destructor runs, it double-frees.

The alternative is a deep copy: allocate new heap storage, copy all data, and update the new struct's pointer. This is correct but expensive. Copying takes time. More importantly, it allocates memory, which involves a system call (or at minimum, allocator bookkeeping) and pollutes the cache with data we may never touch again.

The overhead becomes prohibitive when values pass through function boundaries repeatedly. A function that takes a `vector` by value, processes it, and returns a new `vector` might copy millions of bytes twice: once on entry, once on return. In performance-sensitive code, we want to avoid passing large objects by value entirely, preferring pointers or references. But this complicates APIs and obscures ownership.

### The Shallow Copy Escape

We can observe however that when we pass a vector to a function that consumes it, the caller no longer needs its copy. The bytes in the caller's stack frame become dead immediately after the call. If we could somehow *transfer* ownership without physically duplicating the heap data, we would get the clarity of value semantics with the efficiency of pointer passing.

This is exactly what move semantics provide. Instead of copying the entire data structure, we copy only the struct (the pointer, length, and capacity), then invalidate the source somehow so it does not attempt cleanup.

The _somehow_ is where languages diverge.

In C, there is no language-level support. We must implement this manually:

```c
typedef struct {
    int* data;
    size_t len;
    size_t cap;
} Vec;

void transfer(Vec* dest, Vec* src) {
    *dest = *src;           // shallow copy the struct
    src->data = NULL;       // invalidate source
    src->len = 0;
    src->cap = 0;
}
```

After `transfer`, the destination owns the heap memory, and the source is in a _moved-from_ state (nulled pointers, zero sizes). This works, but nothing enforces it. We can still access `src->data` after the transfer. However, we would read NULL, or worse, the invalidation was forgotten and we would read a pointer that someone else will free.

### C++11: Rvalue References and Move Semantics

Before C++11, the language had one kind of reference: the lvalue reference, written `T&`. An lvalue is roughly "something with a name and an address", a variable, a dereferenced pointer, an array element. Lvalue references bind to lvalues and provide aliased access to existing objects.

But consider a temporary:

```cpp
std::vector<int> make_vector() {
    return std::vector<int>{1, 2, 3, 4, 5};
}

void consume(std::vector<int> v);

consume(make_vector());  // the argument is a temporary
```

The return value of `make_vector()` is not an lvalue, it has no name, no persistent storage, no address we can take. It is an rvalue, a temporary that will be destroyed at the end of the full expression. Before C++11, passing this temporary to `consume` meant copying it, even though the original was about to disappear. We duplicated data that would be destroyed moments later.

C++11 introduced rvalue references, written `T&&`. An rvalue reference binds to rvalues; temporaries, expressions, values returned by functions. The type system now distinguishes "I want to observe this object" (`const T&`) from "I want to steal from this object" (`T&&`).

This distinction enables overloading. A class can define two versions of a constructor:

```cpp
class vector {
public:
    // Copy constructor: source is const lvalue reference
    vector(const vector& other) {
        data_ = new int[other.cap_];
        std::copy(other.data_, other.data_ + other.len_, data_);
        len_ = other.len_;
        cap_ = other.cap_;
    }

    // Move constructor: source is rvalue reference
    vector(vector&& other) noexcept {
        data_ = other.data_;
        len_ = other.len_;
        cap_ = other.cap_;
        // Invalidate source
        other.data_ = nullptr;
        other.len_ = 0;
        other.cap_ = 0;
    }
};
```

When the argument is an rvalue (a temporary, or the result of `std::move`), overload resolution selects the move constructor. We perform the shallow copy and invalidate the source, exactly as in the C version, but now the selection happens automatically based on value category.

The key insight is that `std::move` does not move anything. It is a cast:

```cpp
template<typename T>
constexpr std::remove_reference_t<T>&& move(T&& t) noexcept {
    return static_cast<std::remove_reference_t<T>&&>(t);
}
```

It takes a reference (of any kind, due to reference collapsing) and returns an rvalue reference to the same object. The object does not move. We simply change how the type system categorizes it.

This cast enables us to move from named variables:

```cpp
std::vector<int> v1{1, 2, 3};
std::vector<int> v2 = std::move(v1);  // v1 cast to rvalue, move constructor called
```

After this line, `v2` owns the heap allocation that `v1` previously owned. `v1` is in a moved-from state: its data pointer is null, its length and capacity are zero. The destructor will run when `v1` goes out of scope, but it will do nothing because there is nothing to free.

#### The Implicitly-Declared Move Constructor

The compiler can generate move constructors automatically. If a class has no user-declared copy constructor, copy assignment operator, move assignment operator, or destructor, the compiler declares an implicit move constructor. This implicit version performs member-wise move: for each non-static data member, it moves that member using the member's own move constructor (or copies it, for types without move constructors).

```cpp
struct Wrapper {
    std::vector<int> data;
    std::string name;
    int id;
    // implicit move constructor:
    // Wrapper(Wrapper&& other) noexcept
    //     : data(std::move(other.data))
    //     , name(std::move(other.name))
    //     , id(other.id) {}  // int is trivially movable (just copied)
};
```

For trivially movable types (essentially, types compatible with C), the move constructor copies the object representation as if by `std::memmove`. No per-member move occurs at runtime; the operation reduces to a memory copy of the struct's bytes.

The rule about user-declared special member functions is important. If we declare a destructor (even a defaulted one), the implicit move constructor is not generated:

```cpp
struct C {
    std::vector<int> data;
    ~C() {}  // destructor declared, even though empty
};

C c1;
C c2 = std::move(c1);  // calls copy constructor, not move!
```

This behavior exists because a user-declared destructor suggests the class manages resources in ways the compiler cannot infer. The conservative choice is to fall back to copying. We can force generation of the move constructor with `= default`:

```cpp
struct D {
    std::vector<int> data;
    ~D() {}
    D(D&&) = default;  // explicitly request move constructor
};
```

#### Move Assignment

The same pattern applies to assignment. The move assignment operator takes an rvalue reference and transfers ownership:

```cpp
vector& operator=(vector&& other) noexcept {
    if (this != &other) {
        delete[] data_;        // free our current storage
        data_ = other.data_;
        len_ = other.len_;
        cap_ = other.cap_;
        other.data_ = nullptr;
        other.len_ = 0;
        other.cap_ = 0;
    }
    return *this;
}
```

The self-assignment check is necessary because `std::move` can be applied to any lvalue, including the left-hand side of the assignment itself (though this would be perverse).

The destructor of the moved-from object still runs. Move semantics transfer ownership of *resources*, but the source object continues to exist until its scope ends. The moved-from state must be valid enough for the destructor to execute safely. For `vector`, that means null pointer and zero lengths,the destructor checks for null before freeing, or simply has no work to do.

The `noexcept` specification on move constructors matters for optimization. `std::vector` needs to relocate elements when it grows. If the element type's move constructor is `noexcept`, the vector can move elements to the new buffer. If it might throw, the vector must copy instead to preserve the strong exception guarantee. If an exception occurs during relocation, the original vector must remain intact. The difference can be dramatic for vectors of vectors.

#### Value Categories in Depth

C++11 refined the notion of value categories beyond the simple lvalue/rvalue split. We have three categories:

* An **lvalue** designates an object with identity that persists beyond a single expression. Variables, function returns by reference, dereferenced pointers. The address can be taken.

* A **prvalue** (pure rvalue) is a temporary with no identity. Literals, function returns by value (before binding), results of arithmetic expressions. These initialize objects or compute values, but have no persistent address.

* An **xvalue** (expiring value) has identity but can be moved from. The result of `std::move()`, the result of a cast to rvalue reference, the return value of a function returning `T&&`. The object exists, has an address, but we have permission to transfer its resources.

Overload resolution uses these categories:

```cpp
void f(Widget& w);        // lvalue reference overload
void f(const Widget& w);  // const lvalue reference overload
void f(Widget&& w);       // rvalue reference overload

Widget w;
const Widget cw;
f(w);                // calls f(Widget&)
f(cw);               // calls f(const Widget&)
f(Widget{});         // calls f(Widget&&) - prvalue
f(std::move(w));     // calls f(Widget&&) - xvalue
```

When both `const T&` and `T&&` overloads exist, rvalues (prvalues and xvalues) prefer the `T&&` overload. Lvalues can only bind to the lvalue reference overloads. If only `const T&` is provided, it accepts everything; rvalues bind to const lvalue references, which is why copying was the fallback before C++11.

This machinery operates entirely at compile time. By the time we reach machine code, there are no value categories, no rvalue references, just addresses and data. The type system's job was to select the right constructor or operator; having done so, the generated code performs the memory operations we specified.

### The Moved-From Problem

We have seen that in C++ rvalue references enable overloading, move constructors transfer resources, `std::move` casts to rvalue. But we glossed over a critical detail. After a move, what happens to the source object?

The source object still exists. It has a name, an address, a type. The destructor will run when it goes out of scope. We can call methods on it, read its fields, pass it to functions. The move constructor transferred its *resources*, but the object itself remains.

The C++ standard describes moved-from objects as being in a _valid but unspecified state_. Here _Valid_ means the object satisfies the invariants of its type sufficiently to be destroyed and to have certain operations performed on it (typically assignment and, for some types, queries like `empty()`). _Unspecified_ means we cannot know what values its members hold without inspecting them.

For `std::unique_ptr`, the moved-from state is fully specified: the pointer becomes null. We can observe this:

```cpp
std::unique_ptr<int> p = std::make_unique<int>(42);
std::unique_ptr<int> q = std::move(p);

// p still exists, and we can use it
if (p) {
    std::cout << *p;  // does not execute
} else {
    std::cout << "p is null";  // this executes
}

int* raw = p.get();  // returns nullptr
p.reset(new int(7)); // we can even reuse p
```

The moved-from `unique_ptr` is a perfectly functional object. It holds a null pointer, knows it holds a null pointer, and behaves consistently. The `get()` method returns null. The `bool` conversion returns false. We can `reset()` it with a new pointer and continue using it. This is well-defined behavior.

For `std::vector`, the situation is murkier. The standard guarantees only that the moved-from vector is in a _valid but unspecified state_. In practice, most implementations leave it empty:

```cpp
std::vector<int> v1{1, 2, 3, 4, 5};
std::vector<int> v2 = std::move(v1);

std::cout << v1.size();  // likely prints 0, but not guaranteed
```

The output is _likely_ zero because implementations typically null out the source's data pointer and set its size to zero. But the standard does not require this. An implementation could leave `v1` with garbage values, a dangling pointer, or some other state that satisfies _valid_ (meaning the destructor and assignment still work) without being predictable.

Here is where the design becomes problematic. The compiler does not prevent us from using moved-from objects. There is no warning, no error, nothing. If we forget that we moved from a variable and try to use it, the code compiles and runs:

```cpp
void process(std::vector<int> data);

void example() {
    std::vector<int> v{1, 2, 3, 4, 5};
    process(std::move(v));

    // Bug: v has been moved from
    for (int x : v) {         // compiles fine
        std::cout << x << " "; // prints nothing, or garbage, or crashes
    }
}
```

This is not undefined behavior in the strict sense. Iterating over an empty vector is well-defined, but it is almost certainly a bug. The programmer intended to iterate over the original data and forgot that `process` consumed it. The program silently does the wrong thing.

The cppreference example demonstrates this explicitly:

```cpp
A a1 = f(A());
std::cout << "Before move, a1.s = " << std::quoted(a1.s)
          << " a1.k = " << a1.k << '\n';
A a2 = std::move(a1);
std::cout << "After move, a1.s = " << std::quoted(a1.s)
          << " a1.k = " << a1.k << '\n';
```

Output:
```
Before move, a1.s = "test" a1.k = -1
After move, a1.s = "" a1.k = 0
```

The moved-from object is accessible, observable, and the program continues without complaint. The `std::string` member is empty; the `int` member is zero.

The fundamental issue here is that C++ chose to preserve the moved-from object's existence for backward compatibility and flexibility. Some use cases genuinely benefit from reusing moved-from objects, reassigning to them, or swapping with another object. The cost of this flexibility is that the type system cannot enforce the discipline of "don't use it after moving."

Static analyzers and compilers can sometimes detect use-after-move, but they cannot do so reliably in all cases. The analysis is flow-sensitive and context-dependent, and function boundaries obscure the dataflow. A function that takes `T&&` might move from its parameter, or it might not, the caller cannot tell from the signature alone.


### Rust: Moves Without Ghosts

Rust takes a different approach entirely. When a value moves, the source binding becomes invalid. Not valid but unspecified, *invalid*. The compiler rejects any subsequent use:

```rust
fn process(data: Vec<i32>);

fn example() {
    let v = vec![1, 2, 3, 4, 5];
    process(v);

    for x in v {         // error: use of moved value: `v`
        println!("{}", x);
    }
}
```

The error message is unambiguous:

```
error[E0382]: use of moved value: `v`
 --> src/main.rs:7:14
  |
4 |     let v = vec![1, 2, 3, 4, 5];
  |         - move occurs because `v` has type `Vec<i32>`, which does not implement the `Copy` trait
5 |     process(v);
  |             - value moved here
6 |
7 |     for x in v {
  |              ^ value used here after move
```

There is no moved-from state to observe because there is no way to observe it. The binding `v` is not null, not empty, not unspecified. It simply does not exist from the compiler's perspective after the move. The name remains in scope (you can shadow it with a new binding), but the compiler's initialization tracking marks it as uninitialized.

At the assembly level, the actual data movement is nearly identical to C++. The `Vec`'s three words (pointer, length, capacity) are copied from one stack location to another, or into registers for a function call. There is no heap allocation, no deep copy, just 24 bytes shuffled around. The difference is purely a compile-time concept: Rust tracks that the source is no longer valid.

```rust
fn example() {
    let v1 = vec![1, 2, 3];
    let v2 = v1;           // v1 moved to v2
    println!("{:?}", v1);  // error: borrow of moved value
}
```

The generated assembly for `let v2 = v1` is a `memcpy` of the struct, essentially identical to what C++ would generate. But where C++ would let us access `v1` afterward (finding it in some "valid but unspecified" state), Rust stops compilation.

This tracking happens through dataflow analysis in the compiler. Each variable has an initialization state that the compiler updates as it processes statements. When `v1` is assigned to `v2`, the compiler marks `v1` as uninitialized. Any subsequent use of `v1` is an error, as if we had declared `let v1: Vec<i32>;` without initializing it.

What about reinitialization? A moved-from variable can be assigned a new value:

```rust
fn example() {
    let mut v = vec![1, 2, 3];
    let v2 = v;            // v is now uninitialized
    v = vec![4, 5, 6];     // v is reinitialized
    println!("{:?}", v);   // ok: prints [4, 5, 6]
}
```

The compiler's tracking is flow-sensitive. After the move, `v` is uninitialized. After the reassignment, `v` is initialized with a new value. The `mut` is required because reinitialization is a form of mutation in Rust's model.

Control flow complicates the analysis. If a move occurs in one branch but not another, the variable's initialization state depends on which path was taken:

```rust
fn example(condition: bool) {
    let v = vec![1, 2, 3];
    if condition {
        drop(v);           // v moved into drop()
    }
    println!("{:?}", v);   // error: v might have been moved
}
```

The compiler cannot statically determine which branch executes, so it conservatively assumes `v` might be uninitialized. This occasionally forces us to restructure code or use `Option<T>` to represent _maybe moved_ states explicitly.

For cases where the compiler cannot determine initialization statically, Rust uses *drop flags*. These are runtime boolean values, typically stored on the stack, that track whether a value has been moved. When the variable goes out of scope, the generated code checks the flag before calling the destructor:

```rust
fn example(condition: bool) {
    let x;
    if condition {
        x = Box::new(0);
        println!("{}", x);
    }
    // x goes out of scope: compiler generates code to check if x was initialized
}
```

The drop flag mechanism tells us something about the design trade-off Rust accepted. In straight-line code, the compiler knows exactly which bindings are initialized at every point, and generates direct drops with no runtime overhead. But conditional moves force a choice: either reject some valid programs (overly conservative static analysis), or emit a runtime check. Rust chose the latter for flexibility, keeping the flag on the stack where it costs a byte and a conditional branch at scope exit. For hot loops, we can restructure code to ensure static initialization tracking; for cold paths, the flag is negligible.

What we cannot do in safe Rust is observe a moved-from binding. The asymmetry with C++ lies entirely in what the compiler permits us to write, since both languages copy the same bytes at runtime and both leave the source's memory untouched until the stack frame is reclaimed. C++ allows the moved-from object to participate in subsequent expressions; Rust does not.

### Copy, Move, Clone

We have been speaking of "move" as if it were a single concept, but Rust distinguishes three related operations: implicit copy, move, and explicit clone. Understanding when each applies requires understanding what the type system knows about the data.

An `i32` is 4 bytes. When we write `let y = x` where `x: i32`, the compiler generates a `mov` instruction that copies those 4 bytes. After the assignment, both `x` and `y` hold independent copies of the same value. We can use both. This is a *copy*.

A `Vec<i32>` is 24 bytes on the stack (pointer, length, capacity), but those 24 bytes control an arbitrarily large heap allocation. When we write `let y = x` where `x: Vec<i32>`, the compiler generates the same kind of `mov` instructions to copy those 24 bytes. But now both `x` and `y` would point to the same heap allocation. If we allowed both to be used, we would have aliasing, and when both go out of scope, we would have double-free. So after the assignment, `x` is invalidated. This is a *move*.

At the machine level, copy and move are identical. Both copy the bytes that constitute the value. The difference is in what the compiler permits afterward. For `Copy` types, the source remains valid. For non-`Copy` types, the source is invalidated.

Rust uses the `Copy` trait to mark types where this byte-for-byte duplication is semantically complete. If copying the bytes gives us two independent, fully functional values, the type can be `Copy`. Integers, floats, `bool`, `char`, raw pointers, and tuples or arrays of `Copy` types are all `Copy`. The defining characteristic is that there is no additional resource management beyond the bytes themselves.

The `Copy` trait has a constraint: a type cannot implement both `Copy` and `Drop`. If a type has a destructor, duplicating its bytes creates two values that will both try to run cleanup. For `Vec`, this means double-free. For `File`, this means closing the same file descriptor twice. The mutual exclusion between `Copy` and `Drop` is enforced by the compiler:

```rust
#[derive(Copy, Clone)]
struct Point { x: i32, y: i32 }  // ok: no Drop, all fields Copy

#[derive(Copy, Clone)]
struct Wrapper(Vec<i32>);        // error: Vec is not Copy
```

The error message is direct:

```
error[E0204]: the trait `Copy` cannot be implemented for this type
 --> src/main.rs:4:10
  |
4 | #[derive(Copy, Clone)]
  |          ^^^^
5 | struct Wrapper(Vec<i32>);
  |                -------- this field does not implement `Copy`
```

C++ has a parallel concept in *trivially copyable* types. The C++ standard defines a trivially copyable class as one where each eligible copy constructor, move constructor, copy assignment operator, and move assignment operator is trivial, and the destructor is trivial and non-deleted. "Trivial" here means the compiler-generated default does the right thing, which for these operations means bitwise copy. A `struct` containing only integers and other trivially copyable types is trivially copyable.

The difference is enforcement. In C++, `std::is_trivially_copyable_v<T>` is a compile-time query we can use in `static_assert` or SFINAE, but the language does not prevent us from memcpy-ing a non-trivially-copyable object. We might get away with it if the object has no internal pointers or virtual functions. We might corrupt memory if it does. In Rust, attempting to derive `Copy` on a non-qualifying type is a hard error.

`Clone` is the explicit deep copy operation. Where `Copy` happens implicitly on assignment, `Clone::clone()` must be called explicitly. The implementation can do anything: allocate new memory, copy all elements, increment reference counts, whatever is appropriate for the type. For `Vec<T>`, `clone()` allocates a new buffer and clones each element.

The relationship between `Copy` and `Clone` is that `Copy` is a supertrait of `Clone`. Every `Copy` type must also implement `Clone`, and for `Copy` types, `clone()` is equivalent to a byte copy. This might seem redundant, but it allows generic code to work uniformly:

```rust
fn duplicate<T: Clone>(x: &T) -> T {
    x.clone()
}
```

This function works for both `i32` (where `clone` compiles to a simple load) and `String` (where `clone` allocates and copies). The call site is uniform; the generated code is not.

When we see `.clone()` in Rust code, we know that something potentially expensive is happening. The Rust philosophy is that expensive operations should be visible. Making us write `.clone()` forces acknowledgment of this cost.

C++ takes the opposite approach for copy constructors. Given `std::vector<int> v2 = v1;`, this invokes the copy constructor, which allocates and copies all elements. The syntax is identical to copying an `int`. We must know that `vector` has an expensive copy constructor; the code does not tell us. Move semantics (`v2 = std::move(v1)`) were added in C++11 partly to make expensive operations more visible, but copy remains implicit.

One subtlety: Rust's `clone()` is not always a deep copy in the intuitive sense. For `Rc<T>`, calling `clone()` increments the reference count and returns a new `Rc` pointing to the same allocation. The data is shared, not duplicated. This is the correct semantics for `Rc`, since the entire point of reference counting is to share data. But it means we cannot assume that `clone()` produces an independent copy. The trait's contract is weaker: `clone()` produces a value that is semantically equivalent to the original for the purposes of the type's interface.

### Elision: When Neither Happens

We have seen that moving a value copies its bytes from source to destination. For a 24-byte `Vec` struct, this means three 8-byte writes. But what about returning a `Vec` from a function? Naively, we might expect: the function constructs a `Vec` in its stack frame, then on return, the `Vec` is moved to the caller's stack frame, then the caller receives the returned value. This would mean writing those 24 bytes twice.

The answer to whether we can avoid this lies in how function returns work at the ABI level. The Itanium C++ ABI, which governs calling conventions on most Unix-like systems, distinguishes between *trivial* and *non-trivial* return types. A type is non-trivial for purposes of calls if it has a non-trivial destructor or a non-trivial copy or move constructor. For non-trivial return types, the ABI specifies that the caller passes an address as an implicit parameter, and the callee constructs the return value directly into this address.

The Itanium ABI goes further. The address passed need not point to temporary memory on the caller's stack. Copy elision may cause it to point anywhere: to a local variable's storage, to global memory, to heap-allocated memory. The pointer is passed as if it were the first parameter in the function prototype, preceding all other parameters including `this`. If the return type has a non-trivial destructor, the caller is responsible for destroying the object after, and only after, the callee returns normally. If an exception is thrown after construction but before the return completes, the callee must destroy the return value before propagating the exception.

This machinery enables what C++ calls *copy elision*. The returned object is constructed directly in its final location. Two forms are commonly discussed:

**RVO (Return Value Optimization)** applies when we return a prvalue, a temporary with no name:

```cpp
std::vector<int> make_vector() {
    return std::vector<int>{1, 2, 3};
}
```

The `vector` is constructed directly into the caller-provided address. No temporary exists in `make_vector`'s stack frame.

**NRVO (Named Return Value Optimization)** applies when we return a named local variable:

```cpp
std::vector<int> make_vector() {
    std::vector<int> v{1, 2, 3};
    v.push_back(4);
    return v;
}
```

Here the compiler can allocate `v` directly in the caller-provided space from the start. If it does, there is no copy or move on return. If it does not (perhaps because control flow makes this impossible), the return invokes the move constructor.

Before C++17, both forms of elision were permitted but not guaranteed. A conforming compiler could choose not to elide, and the program would fall back to copy or move constructors. Code relying on elision for correctness, such as returning a non-copyable non-movable type, was non-portable.

C++17 changed this for prvalues through a reformulation of value categories. The standard now specifies that prvalues are not *materialized* until needed. A prvalue does not create a temporary object; it initializes a target object directly. The C++ standard states that the materialization of a temporary object is generally delayed as long as possible to avoid creating unnecessary temporary objects. The result is *guaranteed copy elision* for prvalues. The statement `std::vector<int> v = make_vector();` constructs the vector directly into `v`, mandated by the standard rather than left as an optional optimization.

NRVO remains optional. The standard permits but does not require it. In practice, every major compiler performs NRVO when control flow permits. Multiple return statements returning different named variables typically defeat NRVO because the compiler cannot know at the function's entry which variable will be returned.

How does this relate to Rust? Rust does not have *copy elision* as a named language concept because the problem is different. Rust moves are defined as bitwise copies that invalidate the source. There is no move constructor, no user code that might run, no observable side effects to elide. Moving a `Vec` copies 24 bytes regardless of context.

Rust does perform the same underlying ABI-level optimization. When a function returns a value that cannot fit in registers, the caller passes a hidden pointer in `rdi` (System V) or `rcx` (Microsoft x64), and the callee writes directly to that location. But Rust does not need language-level elision rules because there is no observable difference. Bitwise copy is bitwise copy; constructing directly into the destination versus constructing locally and then copying produces identical bit patterns.

```rust
fn make_vec() -> Vec<i32> {
    vec![1, 2, 3, 4, 5]
}

fn caller() {
    let v = make_vec();
}
```

With optimizations, `make_vec` receives the hidden pointer in `rdi` and writes the `Vec`'s three fields (pointer, capacity, length) directly to that address. There is no intermediate `Vec` on `make_vec`'s stack frame that gets copied out. The heap allocation happens once, the `Vec` header is written once, directly to where `caller` wants it.

For types that fit in registers, both languages return values in RAX and RDX. A function returning `(i32, i32)` involves no memory operations for the return itself.

The consequence is that returning large values by value in Rust is not expensive because of the return mechanism. The cost is in constructing the data structure: the allocations, the element initialization, the potential reallocations. The mechanics of getting the result back to the caller add nothing beyond what the ABI already requires for any struct return.

## Smart Pointers and Reference Counting

Move semantics solve the problem of transferring exclusive ownership efficiently. But some data structures cannot express their sharing patterns through exclusive ownership alone. A graph where multiple edges reference the same node, a cache that outlives any single user, a callback that must remain valid for multiple callers: these require *shared* ownership, where the resource is released only when the last owner disappears.

We have seen how C++ and Rust handle exclusive heap ownership through `unique_ptr` and `Box`. The move semantics we examined apply directly: `unique_ptr` leaves a nullptr after move, `Box` leaves no valid binding at all. Now we turn to the harder problem of shared ownership, where multiple owners must coordinate to determine when deallocation occurs.

### Shared Ownership: `shared_ptr`

When multiple parts of a program need to share ownership of heap-allocated data, we need reference counting. C++'s `shared_ptr` implements this with a *control block*: a separate heap allocation that stores metadata about the shared object.

A typical `shared_ptr<T>` contains two pointers: one to the managed object (`T*`), and one to the control block. The control block contains:

- A strong reference count (the number of `shared_ptr`s pointing to the object)
- A weak reference count (the number of `weak_ptr`s, plus one if the strong count is nonzero)
- A pointer to the managed object (or the object itself, if allocated together)
- The deleter (type-erased, since different `shared_ptr`s with different deleters can share ownership)
- The allocator (also type-erased)

When we copy a `shared_ptr`, the strong count is incremented atomically. When a `shared_ptr` is destroyed, the strong count is decremented. If the strong count reaches zero, the managed object is destroyed (the deleter is invoked). The control block itself is not destroyed until the weak count also reaches zero, because `weak_ptr`s need to query the control block to determine whether the object still exists.

```cpp
auto p = std::make_shared<Widget>();  // one allocation: control block + Widget
auto q = p;                           // strong count: 2
q.reset();                            // strong count: 1
// p destroyed, strong count: 0, Widget destroyed
```

The `make_shared` function is preferred over `shared_ptr<T>(new T)` because it performs a single allocation for both the control block and the object, improving cache locality and reducing allocation overhead. The downside is that the memory for the object cannot be released until all weak references are also gone, since the control block and object share a single allocation.

### Shared Ownership: `Rc<T>` and `Arc<T>`

Rust separates the single-threaded and multi-threaded cases into distinct types. `Rc<T>` (reference counted) uses non-atomic operations and is not thread-safe. `Arc<T>` (atomically reference counted) uses atomic operations and can be shared across threads.

The layout of `Rc<T>` is similar in spirit to `shared_ptr`, but simpler. An `Rc<T>` is a single pointer to a heap allocation containing:

- A strong count (`Cell<usize>`, non-atomic)
- A weak count (`Cell<usize>`, non-atomic)
- The data (`T`)

There is no separate deleter or allocator. `Rc<T>` always uses the global allocator and always drops `T` in place. This means `Rc<T>` is a single pointer, 8 bytes, not two pointers like `shared_ptr`.

```rust
use std::rc::Rc;

let a = Rc::new(Widget::new());  // allocates RcBox containing counts + Widget
let b = Rc::clone(&a);           // strong count: 2
drop(b);                         // strong count: 1
// a dropped, strong count: 0, Widget dropped, RcBox deallocated
```

The convention is to write `Rc::clone(&a)` rather than `a.clone()`. Both work identically, but the explicit form makes clear that we are incrementing a reference count, not performing a deep copy.

`Arc<T>` has the same layout, with the counts replaced by `AtomicUsize`:

```rust
use std::sync::atomic::AtomicUsize;

pub struct ArcInner<T> {
    strong: AtomicUsize,
    weak: AtomicUsize,
    data: T,
}
```

The atomic operations add overhead. Incrementing an atomic counter requires synchronization at the hardware level: a `lock` prefix on x86, load-linked/store-conditional sequences on ARM. In high-contention scenarios, cache lines bounce between cores. If shared ownership is needed but thread safety is not, `Rc` avoids this cost entirely.

Rust enforces this separation at compile time. `Rc<T>` does not implement `Send` or `Sync`. Attempting to move an `Rc` across thread boundaries is a type error:

```rust
use std::rc::Rc;
use std::thread;

let rc = Rc::new(42);
thread::spawn(move || {
    println!("{}", rc);  // error: `Rc<i32>` cannot be sent between threads safely
});
```

The error is not a runtime panic. The code does not compile. We cannot accidentally introduce data races by using the wrong smart pointer type.

### Reference Count Synchronization

The increment operation in `Arc::clone` uses `Ordering::Relaxed`. This might seem dangerous for a multi-threaded primitive, but the reasoning is precise that incrementing the count establishes no ordering relationship with any other memory access. We already have access to the `Arc`, which means we already have a valid reference to the data. We are simply recording that one more owner exists. The only requirement is atomicity itself, ensuring that two concurrent increments do not lose a count.

```rust
impl<T> Clone for Arc<T> {
    fn clone(&self) -> Arc<T> {
        let inner = unsafe { self.ptr.as_ref() };
        inner.rc.fetch_add(1, Ordering::Relaxed);
        Arc { ptr: self.ptr, phantom: PhantomData }
    }
}
```

The decrement is where the complexity lies. Consider what happens when the last owner drops its `Arc`. At that moment, no other thread holds a reference, and we must deallocate. But we must first ensure that all writes performed by any previous owner are visible to us. If thread A writes to `arc.data` and then drops its `Arc`, and thread B subsequently drops the last `Arc`, thread B must see thread A's writes before running the destructor.

The standard solution uses release-acquire synchronization:

```rust
impl<T> Drop for Arc<T> {
    fn drop(&mut self) {
        let inner = unsafe { self.ptr.as_ref() };
        if inner.rc.fetch_sub(1, Ordering::Release) != 1 {
            return;
        }
        std::sync::atomic::fence(Ordering::Acquire);
        unsafe { drop(Box::from_raw(self.ptr.as_ptr())); }
    }
}
```

Every `fetch_sub` with `Release` ordering guarantees that all prior writes in that thread are visible to any thread that subsequently performs an `Acquire` operation on the same atomic variable and observes the stored value. When thread B's `fetch_sub` returns 1, indicating it was the last owner, the subsequent `Acquire` fence synchronizes with all the `Release` decrements that preceded it. Thread B now sees every write that any previous owner performed before dropping.

On x86, the `Release` ordering on `fetch_sub` compiles to a simple `lock xadd` instruction, which already provides the necessary ordering guarantees due to x86's strong memory model. The `Acquire` fence compiles to nothing, as x86 loads already have acquire semantics. On ARM, the situation is different: `Release` requires a `dmb ish` barrier before the store, and `Acquire` requires a barrier after the load.

Using `Relaxed` for the decrement would allow the deallocating thread to see stale data, potentially reading uninitialized memory or seeing partially-written values. Using `SeqCst` everywhere would be correct but unnecessarily expensive, adding full memory barriers where weaker ordering suffices. The release-acquire pattern is the minimum synchronization required for correctness.

### Raw Pointers, Variance, and the Drop Checker

An `Arc` internally holds a raw pointer to the heap allocation. A naive definition might use `*mut ArcInner<T>`, but this creates two problems that `NonNull<T>` and `PhantomData<T>` solve.

Raw pointers are *invariant* in their type parameter. If we have an `Arc<&'static str>`, we should be able to use it where an `Arc<&'a str>` is expected (assuming `'static: 'a`), because a longer-lived reference can substitute for a shorter-lived one. But `*mut T` does not allow this substitution. `NonNull<T>` is a pointer wrapper that is *covariant* in `T`, restoring the expected subtyping relationship.

The second problem concerns the drop checker. When the compiler analyzes whether a type can be safely dropped, it needs to know what the type logically owns. A raw pointer carries no ownership information; the compiler assumes `*mut T` does not own a `T`. But `Arc<T>` *does* own a `T`, at least when it is the last owner. If we do not communicate this to the compiler, code that should be rejected might compile:

```rust
fn bad<'a>(arc: Arc<&'a str>) {
    let local = String::from("local");
    // Without PhantomData, the compiler might not realize
    // that dropping arc could access the &str
}
```

Adding `PhantomData<ArcInner<T>>` tells the drop checker that `Arc<T>` behaves as if it contains an `ArcInner<T>`, which contains a `T`. The drop checker then ensures that `T` outlives the `Arc`, preventing dangling references in the destructor.

The final structure:

```rust
pub struct Arc<T> {
    ptr: NonNull<ArcInner<T>>,
    phantom: PhantomData<ArcInner<T>>,
}
```

This is still a single pointer in size, 8 bytes. `PhantomData` is a zero-sized type; it affects the type system without occupying memory. `NonNull` is a `#[repr(transparent)]` wrapper around `*const T`, also pointer-sized. The layout is identical to a raw pointer, but the semantics are correct for a reference-counted smart pointer.

### Weak References and the Control Block Lifetime

A `weak_ptr` or `Weak` does not keep the managed object alive. It holds a pointer to the control block (or inner allocation), not to the object itself. When we attempt to *upgrade* a weak reference to a strong one, the operation must atomically check whether the object still exists and, if so, increment the strong count before another thread can decrement it to zero.

Consider the race: thread A holds the last `shared_ptr` and is about to drop it. Thread B holds a `weak_ptr` and calls `lock()`. If B reads the strong count as 1, then A decrements it to 0 and deallocates, then B increments it to 1, we have a use-after-free. The upgrade must be atomic with respect to the decrement.

In C++, `weak_ptr::lock()` performs a compare-and-swap loop. It loads the strong count, checks if it is zero (returning an empty `shared_ptr` if so), then attempts to atomically increment it from the observed value. If another thread modified the count in the meantime, the CAS fails and the loop retries. The implementation looks roughly like:

```cpp
shared_ptr<T> weak_ptr<T>::lock() const noexcept {
    long count = control_block->strong_count.load(std::memory_order_relaxed);
    while (count != 0) {
        if (control_block->strong_count.compare_exchange_weak(
                count, count + 1, std::memory_order_acq_rel)) {
            return shared_ptr<T>(/* from control block */);
        }
        // count was updated by compare_exchange_weak on failure
    }
    return shared_ptr<T>();  // empty
}
```

Rust's `Weak::upgrade()` follows the same pattern. The `acq_rel` ordering on the successful CAS ensures that if we observe a nonzero count, our subsequent access to the data sees all writes that happened before any previous owner released their reference.

The control block has a two-phase destruction. When the strong count reaches zero, the managed object is destroyed: its destructor runs, and if allocated separately from the control block, its memory is freed. The control block itself survives. Only when the weak count also reaches zero is the control block deallocated. This separation allows weak references to safely query whether the object exists.

With `make_shared`, the object and control block occupy a single allocation. The object's destructor runs when the strong count hits zero, but the memory cannot be freed until the weak count also hits zero, because the control block metadata lives in the same allocation. Long-lived weak references to `make_shared` objects keep the entire allocation alive, even though the object has been destroyed. Separate allocation via `shared_ptr<T>(new T)` avoids this: the object's memory is freed immediately when the strong count hits zero, and only the smaller control block remains for weak reference bookkeeping.

### The Cost of Atomics in Reference Counting

`Arc::clone` compiles to a `lock xadd` instruction on x86. The `lock` prefix asserts exclusive ownership of the cache line containing the reference count, forcing all other cores to invalidate their copies. On a system with 8 cores all cloning the same `Arc`, each clone causes 7 cache invalidations. The reference count ping-pongs between cores, and each increment waits for the cache line to arrive in the exclusive state.

```asm
; Arc::clone on x86-64
mov     rax, qword ptr [rdi]        ; load pointer to ArcInner
lock    xadd qword ptr [rax], 1     ; atomic increment of refcount
```

The `lock xadd` instruction alone takes roughly 15-30 cycles on modern x86 when uncontended, rising to 100+ cycles under contention as cores compete for the cache line. On ARM, the equivalent uses load-exclusive/store-exclusive pairs:

```asm
; Arc::clone on ARM64
.retry:
    ldxr    x1, [x0]          ; load-exclusive refcount
    add     x1, x1, #1
    stxr    w2, x1, [x0]      ; store-exclusive
    cbnz    w2, .retry        ; retry if store failed
```

The store-exclusive fails if another core modified the cache line between the load and store, requiring a retry. Under high contention, threads spin in this loop, wasting cycles.

The decrement in `Arc::drop` has the same atomic cost, plus the `Acquire` fence when we observe the count reaching zero. On x86, the fence is free because `lock xadd` already has acquire-release semantics. On ARM, it expands to a `dmb ish` (data memory barrier, inner shareable domain), which stalls the pipeline until all prior memory operations complete.

Beyond instruction costs, reference counting interacts poorly with modern cache hierarchies. The reference count and the data occupy the same allocation, often the same cache line. If threads read the data while another thread clones or drops the `Arc`, false sharing occurs: the reading threads' cache lines are invalidated by the write to the reference count, even though the data itself is unchanged. This is unavoidable without splitting the count into a separate allocation, which would add indirection and defeat the single-pointer layout.

`Rc<T>` avoids all of this. With no threading, the increment is a simple `add` instruction, the decrement a simple `sub`. No cache coherence traffic, no memory barriers, no retry loops. The type system's guarantee that `Rc` is not `Send` lets us omit the synchronization entirely.

```asm
; Rc::clone on x86-64
mov     rax, qword ptr [rdi]        ; load pointer to RcBox
inc     qword ptr [rax]             ; non-atomic increment
```

### C's Approach: Manual Reference Counting

C has no built-in reference-counted smart pointers, but the pattern is pervasive. The Linux kernel's `kref`, GLib's `GObject`, and COM's `IUnknown` all implement reference counting manually, each with its own conventions and failure modes.

A single-threaded implementation is pretty straightforward:

```c
struct Widget {
    int refcount;
    // ... data ...
};

struct Widget *widget_ref(struct Widget *w) {
    w->refcount++;
    return w;
}

void widget_unref(struct Widget *w) {
    if (--w->refcount == 0) {
        widget_destroy(w);
        free(w);
    }
}
```

Threading requires atomic operations. Before C11, these were compiler or platform-specific: GCC's `__sync_fetch_and_add`, MSVC's `InterlockedIncrement`, or inline assembly. C11 standardized atomics, but the memory ordering must still be chosen correctly:

```c
#include <stdatomic.h>

struct Widget {
    atomic_int refcount;
    // ... data ...
};

struct Widget *widget_ref(struct Widget *w) {
    atomic_fetch_add_explicit(&w->refcount, 1, memory_order_relaxed);
    return w;
}

void widget_unref(struct Widget *w) {
    if (atomic_fetch_sub_explicit(&w->refcount, 1, memory_order_release) == 1) {
        atomic_thread_fence(memory_order_acquire);
        widget_destroy(w);
        free(w);
    }
}
```

The ordering arguments mirror what we saw in `Arc`: `relaxed` for increment (we already have a valid reference), `release` for decrement (publish our writes), `acquire` fence before destruction (synchronize with all releasers). Getting any of these wrong introduces data races that manifest as sporadic corruption, use-after-free in specific timing windows, or crashes that appear once per million operations.

The Linux kernel's `kref` wraps this pattern into a small struct with `kref_get` and `kref_put` operations. This discipline is not enforced, nothing prevents calling `kref_get` on an object whose count has already reached zero, or forgetting to call `kref_put` on an error path. Static analysis tools like Sparse and Coverity catch some bugs; the rest are found through testing, fuzzing, or production crashes.

Every usage site must remember to call `widget_ref` when acquiring a reference and `widget_unref` when releasing one. There is no automatic cleanup on scope exit, no destructor to invoke at the end of a block. If a function returns early due to an error, every acquired reference must be explicitly released. Missing a single `unref` leaks memory; an extra `unref` corrupts the count or triggers a double-free.

---

Part III will examine how types are represented in memory and how polymorphism is implemented. Rust reorders struct fields by default to minimize padding; `repr(C)` restores predictable layout for FFI. Fat pointers carry metadata alongside the data address: slices carry length, trait objects carry vtable pointers. We will compare monomorphization (C++ templates, Rust generics) against dynamic dispatch (C++ virtual functions, Rust trait objects), examining the generated code and the trade-offs in binary size, call overhead, and cache behavior. Closures will complete the picture: anonymous structs that capture their environment, with the `Fn`/`FnMut`/`FnOnce` traits encoding how they access captured variables.
