---
author: Luca Lombardo
pubDatetime: 2025-12-23T00:00:00.000Z
title: "Who Owns the Memory? Part 2: Ownership and Move Semantics"
slug: who-owns-the-memory-pt2
featured: false
draft: false
tags:
  - Rust
  - Programming
description: "Exploring ownership in Rust, C, C++ and its impact on memory management."
---

## Table of Contents


## Ownership: Who Calls Free?

We have established that objects occupy storage, that storage has duration, and that the type system imposes structure on raw bytes. But we have sidestepped a question that dominates systems programming in practice: when heap-allocated memory must be released, who bears responsibility for releasing it?

The stack is self-managing. When a function returns, the stack pointer moves and automatic storage vanishes. There is no decision to make, no function to call, no possibility of error. The heap is different. Memory obtained through `malloc` persists until someone calls `free`. The allocator cannot know when we are finished with an allocation; only our program logic knows that. And so the burden falls on us.

This burden is not merely administrative. The consequences of mismanagement are severe and often exploitable. If we free too early, subsequent accesses read garbage or, worse, data that a new allocation has placed there (a use-after-free, the vulnerability class behind a substantial fraction of remote code execution exploits). If we free twice, we corrupt the allocator's metadata; an attacker who controls the timing can often leverage this into arbitrary write primitives. If we never free, we leak, and the process grows until the operating system intervenes.

The question is how programming languages help us manage this responsibility, or whether they help at all.

### C: Discipline Without Enforcement

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

C++ gives us the machinery for safe resource management. Using that machinery is a choice the language cannot enforce. A codebase mixing raw pointers, `unique_ptr`, `shared_ptr`, and manual `new`/`delete` requires reasoning about ownership at every function boundary. The answer is not in the types; it is in the programmers' heads, in comments, in coding standards. This is better than C, where even the machinery does not exist. But it is not sufficient to eliminate memory safety bugs from large codebases.


### Rust: Ownership in the Type System

Rust takes the RAII pattern and embeds it into the type system as a non-negotiable rule: every value has exactly one owner, and when that owner goes out of scope, the value is dropped. This is not a convention that programmers may follow or ignore. It is a property that the compiler verifies statically.

Consider what happens when we allocate a vector:

```rust
fn example() {
    let v = vec![1, 2, 3];
}
```

The binding `v` owns the heap-allocated buffer. When `v` goes out of scope at the closing brace, Rust calls `drop` on the `Vec`, which deallocates the buffer. This means that we can't never accidentally forget to free the memory.

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

### The Single-Destructor Problem


The RAII model we have described, whether in C++ or Rust, shares the same structural limitation: the destructor is a single, parameterless function that returns nothing. When an object goes out of scope, exactly one action occurs. We cannot choose between alternatives. We cannot pass runtime information to the cleanup logic. We cannot receive a result from it.

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

The idea that we call `commit()` when the transaction succeeds, let the destructor rollback on error paths or early returns. Exception safety falls out naturally; if an exception propagates, the destructor runs, and uncommitted transactions rollback.

The problem is that forgetting to call `commit()` is not a compile-time error. If we write a function that successfully completes its work but neglects to call `commit()`, the destructor happily rolls back. The program is wrong, but the compiler cannot tell us. We have traded one category of bug (forgetting to cleanup) for another (forgetting to finalize). The second category is arguably more insidious because the cleanup happens, silently doing the wrong thing.

Rust's ownership system does not help here. We can implement the same pattern:

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

The `commit` method takes `self` by value, consuming the transaction. After calling `commit`, the binding is gone, and no second commit or rollback can occur. But if we never call `commit`, the transaction goes out of scope, `drop` runs, and we rollback. No compiler error. The type system tracked ownership but not obligation.

#### Defer: Explicit Scope-Bound Cleanup

Some languages take a different approach entirely. Rather than binding cleanup to object destruction, they provide explicit defer statements that execute at scope exit.

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

The cleanup code sits immediately after the acquisition code. We see allocation and deallocation together, which aids comprehension. The `defer` executes in reverse order of declaration, matching the natural dependency order (later allocations may depend on earlier ones). If we return early, throw an error, or fall through normally, the deferred expressions run.

Zig extends this with `errdefer`, which executes only if the function returns an error:

```zig
fn createResource() !*Resource {
    const memory = try allocator.alloc(u8, size);
    errdefer allocator.free(memory);  // only on error path
    
    const handle = try system.open(memory);
    errdefer system.close(handle);  // only on error path
    
    return Resource{ .memory = memory, .handle = handle };
    // success: errdefers do not run, caller now owns resources
}
```

This separates error-path cleanup from success-path transfer. On success, we return the resources to the caller; the `errdefer` statements do not execute. On failure, we clean up everything we allocated before returning the error. The distinction between "cleanup on all paths" and "cleanup on error paths" is explicit in the code.

The defer model has a structural limitation that RAII does not: the cleanup is scope-bound. We cannot return a "thing that will be cleaned up later" to our caller the way we return an RAII object. The defer runs when the current scope exits, period. If we want the cleanup to happen in a different scope, we must structure our code so that scope is the right one, or we must not use defer at all.

RAII is more flexible in this regard. We can return an RAII object, store it in a data structure, transfer ownership to another thread. The cleanup travels with the object. Defer is local; RAII is transferable.

But defer has an advantage in simplicity. We do not need to define a type, implement a trait, worry about drop order among struct fields. We write the cleanup code inline, at the point of acquisition. For resources that do not leave the current function, defer is often cleaner.

#### Linear Types

This transaction shows us a gap in the type system's guarantees. RAII ensures that cleanup happens, but not that we made an explicit decision about which cleanup to perform. The destructor chooses for us, silently.

Linear types close this gap. A linear type must be explicitly consumed; it cannot simply go out of scope. If we try to let a linear value fall out of scope without passing it to a consuming function, the compiler rejects the program.

Consider a hypothetical extension to Rust with linear types:

```rust
// Hypothetical syntax
#[linear]
struct Transaction {
    db: Database,
}

fn commit(txn: Transaction) -> Result<(), Error> {
    txn.db.commit()?;
    destruct txn;  // explicitly consume
    Ok(())
}

fn rollback(txn: Transaction, reason: &str) {
    txn.db.rollback();
    log::info!("Transaction rolled back: {}", reason);
    destruct txn;  // explicitly consume
}

fn do_work(db: Database) {
    let txn = Transaction { db };
    // ... work ...
    
    // ERROR: `txn` goes out of scope without being consumed
    // must call either commit(txn) or rollback(txn, ...)
}
```

The transaction cannot be ignored. We must explicitly choose to commit or rollback. The compiler enforces this. Forgetting to decide is a compile-time error, not a runtime silent rollback.

Rather than one implicit destructor, we have multiple explicit consuming functions. These functions can take parameters (the rollback reason), return values (the commit result), and perform arbitrarily different logic. The compiler guarantees we call exactly one of them.

Languages with linear types (Vale, Austral, and to some extent Haskell with LinearTypes) can express patterns that RAII cannot. A cache entry handle that must be explicitly removed or retained. A promise that must be fulfilled exactly once. A thread handle that must be joined or detached, never silently dropped. Each of these represents a future obligation that the type system tracks.

#### Why Rust Does Not Have Linear Types

Rust's types are affine, not linear. An affine type can be used at most once: we can consume it, or we can let it drop, but we cannot use it twice. A linear type must be used exactly once: we must consume it, we cannot let it silently drop.

The difference is the _at most_ versus _exactly_. Rust permits silent dropping because `Drop::drop` must always be callable. When a scope exits, when a panic unwinds the stack, when we reassign a variable, Rust calls `drop`. The entire language assumes that any value can be dropped at any time.

Linear types would break this assumption. If a value must be explicitly consumed, what happens when a panic unwinds through a function holding that value? The unwinding code cannot know which consuming function to call, what parameters to pass, what to do with the return value. The type's constraint is violated by the mechanics of stack unwinding.

One could imagine solutions: a separate `on_panic` handler for linear types, or restricting linear types to no_unwind contexts, or transforming panics into error returns that the programmer must handle. These are all possible, but they represent significant complexity and would require changes throughout the ecosystem. The standard library's `Vec::pop` returns an `Option<T>`, assuming it can drop the popped element if the caller ignores the return value. With linear types in `T`, this interface would be unsound. Every generic container, every iterator adapter, every function that might discard a value would need to be reconsidered.

Rust chose affine types, accepting that the compiler cannot enforce explicit consumption but gaining a simpler model where any value can be dropped. The `#[must_use]` attribute provides a weaker form of the linear guarantee: a warning (not an error) if a value is unused. It catches some mistakes but does not provide the hard guarantee that linear types would.

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

The overhead becomes prohibitive when values pass through function boundaries repeatedly. A function that takes a `vector` by value, processes it, and returns a new `vector` might copy millions of bytes twice—once on entry, once on return. In performance-sensitive code, we wanto to avoid passing large objects by value entirely, preferring pointers or references. But this complicates APIs and obscures ownership.
.

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

The `noexcept` specification on move constructors matters for optimization. `std::vector` needs to relocate elements when it grows. If the element type's move constructor is `noexcept`, the vector can move elements to the new buffer. If it might throw, the vector must copy instead to preserve the strong exception guarantee—if an exception occurs during relocation, the original vector must remain intact. The difference can be dramatic for vectors of vectors.

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

When both `const T&` and `T&&` overloads exist, rvalues (prvalues and xvalues) prefer the `T&&` overload. Lvalues can only bind to the lvalue reference overloads. If only `const T&` is provided, it accepts everything—rvalues bind to const lvalue references, which is why copying was the fallback before C++11.

This machinery operates entirely at compile time. By the time we reach machine code, there are no value categories, no rvalue references, just addresses and data. The type system's job was to select the right constructor or operator; having done so, the generated code performs the memory operations we specified.

<!-- ### The Moved-From Problem

The move constructor transfers resources from the source object to the destination. But what happens to the source? In C++, the source object continues to exist. Its lifetime does not end at the move. It remains a valid object, occupying the same storage, and its destructor will be called when it goes out of scope.

The C++ standard describes this state as "valid but unspecified." The object is in some state that satisfies its class invariants (so the destructor can run safely), but the exact state is not defined. For `std::string`, the moved-from string might be empty, or it might contain some unspecified characters. For `std::vector`, the moved-from vector might be empty, or it might have some capacity. The standard does not say.

Some types have a fully specified moved-from state. `std::unique_ptr` is defined to be null after a move. We can rely on this:

```cpp
std::unique_ptr<int> p = std::make_unique<int>(42);
std::unique_ptr<int> q = std::move(p);
assert(p == nullptr);  // guaranteed by the standard
```

But this guarantee is type-specific. For user-defined types, the moved-from state depends on how the move constructor was written. And critically, the compiler does not prevent us from using a moved-from object.

```cpp
std::vector<int> v = {1, 2, 3};
std::vector<int> w = std::move(v);
v.push_back(4);  // compiles and runs
std::cout << v.size() << "\n";  // prints... something
```

This code compiles without warning. It runs without crashing. The `push_back` operates on whatever state `v` happens to be in after the move. If the move left `v` empty, we get a vector containing just `4`. If the move left `v` with some residual capacity, we get the same result but perhaps with less reallocation. The behavior is implementation-defined, which is a polite way of saying unpredictable.

The situation is worse for types where the moved-from state is less benign:

```cpp
class Connection {
public:
    Connection(Connection&& other) noexcept
        : socket_(other.socket_)
    {
        other.socket_ = -1;  // sentinel value
    }
    
    void send(const char* data) {
        if (socket_ == -1) {
            // what do we do here?
        }
        ::send(socket_, data, strlen(data), 0);
    }
    
    ~Connection() {
        if (socket_ != -1) {
            ::close(socket_);
        }
    }

private:
    int socket_;
};
```

The move constructor sets the source's socket to -1 to prevent double-close. But `send` must now check for this sentinel value. What should it do if called on a moved-from connection? Throw an exception? Return silently? Assert and crash? The type designer must make this decision, document it, and hope that callers read the documentation.

The fundamental problem is that move in C++ is a value transformation, not a lifetime termination. The source object persists. It has a valid address. Its members are accessible. The type system does not distinguish between a moved-from object and a normal object. As far as the compiler is concerned, `v` after `std::move(v)` is still a `std::vector<int>`, fully usable.

This design was intentional. The C++ committee wanted moves to be non-destructive because destructors must run, exception handling requires valid objects during stack unwinding, and backward compatibility demanded that existing code patterns remain valid. The cost is that use-after-move bugs are possible, common, and invisible to the compiler.

Static analyzers can catch some of these bugs. Clang's `-Wuse-after-move` warning detects simple cases:

```cpp
std::vector<int> v = {1, 2, 3};
std::vector<int> w = std::move(v);
v.push_back(4);  // warning: use after move
```

But the analysis is flow-sensitive and conservative. It cannot track moves through function calls, conditionals, or loops. It misses many real bugs and occasionally produces false positives. The warning is useful but not a substitute for language-level guarantees.


### Rust: Moves Without Ghosts

Rust takes a different approach. When a value is moved, the source binding becomes invalid. Not invalid in the sense of holding a sentinel value, but invalid in the sense of not existing. The compiler refuses to generate code that accesses it.

```rust
let v = vec![1, 2, 3];
let w = v;
v.push(4);  // error: borrow of moved value: `v`
```

This is a compile-time error. The program is rejected before it runs. There is no moved-from state because there is no moved-from object. The binding `v` is simply gone.

How does this work at the machine level? After all, `v` occupied stack space. That space still exists. The bytes that were `v`'s pointer, length, and capacity are still there. What prevents us from reading them?

The answer is that the Rust compiler tracks initialization state as part of its semantic analysis. Every binding has a status: initialized or uninitialized. When we write `let w = v`, the compiler performs a bitwise copy of `v`'s bytes into `w`'s storage, then marks `v` as uninitialized. Any subsequent attempt to use `v` is rejected during compilation.

Consider the generated code for a simple move:

```rust
fn example() {
    let v: Vec<i32> = vec![1, 2, 3];
    let w = v;
    // v is now uninitialized
    drop(w);
}
```

In the compiled output, the move from `v` to `w` is just a `memcpy` of 24 bytes (the size of `Vec` on 64-bit: pointer, length, capacity). There is no destructor call for `v`. There is no nullification of `v`'s pointer. The bytes that were `v` are simply abandoned. When `w` goes out of scope, `drop` is called on `w`, freeing the heap buffer. The bytes that were `v` are eventually overwritten by subsequent stack usage.

This is cheaper than C++ move semantics. A C++ move constructor typically copies the fields and then nullifies the source's pointer to prevent double-free. Rust's move copies the fields and does nothing else. The "nullification" is handled by the compiler's knowledge that `v` is no longer initialized; no runtime operation is needed.

What about conditional initialization? Rust handles this with drop flags:

```rust
fn example(condition: bool) {
    let x;
    if condition {
        x = vec![1, 2, 3];
    }
    // Is x initialized here?
}
```

At the closing brace, should the compiler call `drop` on `x`? It depends on whether the `if` branch executed. When the compiler cannot determine initialization state statically, it inserts a hidden boolean—a drop flag—that tracks whether `x` was initialized. At scope exit, the generated code checks the flag before calling `drop`.

For straight-line code and simple branches, the compiler can often eliminate drop flags through static analysis:

```rust
let mut x = Box::new(0);    // x is initialized
let y = x;                  // y is initialized, x is uninitialized
x = Box::new(1);            // x is reinitialized
                            // at scope exit: drop y, drop x
```

The compiler knows the state of every binding at every point in this code. No runtime flags are needed. The optimization is called "static drop semantics," and it applies to the vast majority of real code.

The contrast with C++ is stark. In C++, after `std::move(v)`, we have a zombie object: it exists, it has a type, we can call methods on it, but it is semantically dead. The programmer must remember not to use it. The compiler offers no help (aside from optional warnings that catch only simple cases).

In Rust, after `let w = v`, the binding `v` does not exist. There is no zombie. There is nothing to accidentally use. The compiler has removed `v` from the set of valid names in scope. Attempting to use it is not a warning; it is an error that prevents compilation.

This extends to function calls. When we pass a value to a function that takes ownership, the binding is moved:

```rust
fn consume(v: Vec<i32>) {
    // v is owned here
}

fn main() {
    let data = vec![1, 2, 3];
    consume(data);
    println!("{:?}", data);  // error: borrow of moved value
}
```

The signature `fn consume(v: Vec<i32>)` declares that `consume` takes ownership. After the call, `data` is invalid. The compiler enforces this. There is no way to accidentally use `data` after passing it to `consume`.

Compare to C++:

```cpp
void consume(std::vector<int> v) {
    // v is a copy or a move, depending on how we called
}

int main() {
    std::vector<int> data = {1, 2, 3};
    consume(std::move(data));
    std::cout << data.size() << "\n";  // compiles, runs, prints something
}
```

The C++ code compiles and runs. The `std::move` casts `data` to an rvalue, allowing the move constructor to be selected. But `data` still exists after the call. It is in its "valid but unspecified" state. The programmer can still access it, and the compiler will not object.

Rust's move semantics eliminate use-after-move bugs at the language level. The bugs cannot exist because the language does not permit the code patterns that would cause them. This is not a lint or a warning or a best practice. It is a hard constraint enforced by the type system.

The cost is that Rust programmers must be explicit about ownership. When a function needs temporary access to a value without taking ownership, it must use a reference: `fn borrow(v: &Vec<i32>)` or `fn mutate(v: &mut Vec<i32>)`. The type signature communicates ownership intent, and the compiler verifies that the caller respects it. This is more verbose than C++'s implicit copies and moves, but it eliminates an entire category of bugs while making ownership relationships explicit in the code. -->