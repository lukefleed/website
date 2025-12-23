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

The RAII model we have described, whether in C++ or Rust, shares a fundamental structural limitation: the destructor is a single, parameterless function that returns nothing. When an object goes out of scope, exactly one action occurs. We cannot choose between alternatives. We cannot pass runtime information to the cleanup logic. We cannot receive a result from it.

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

The intent is clear: call `commit()` when the transaction succeeds, let the destructor rollback on error paths or early returns. Exception safety falls out naturally; if an exception propagates, the destructor runs, and uncommitted transactions rollback.

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

The transaction problem reveals a gap in the type system's guarantees. RAII ensures that cleanup happens, but not that we made an explicit decision about which cleanup to perform. The destructor chooses for us, silently.

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