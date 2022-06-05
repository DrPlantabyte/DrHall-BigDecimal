# Dr. Hall's BigDecimal Library
A simple BigDecimal implementation for scientific computing.

## Arbitrary Precision Integer and Decimal Numbers
The DrHall-BigDecimal provides an implementation of two common numerical data 
types, generally known as big-int and big-decimal. The former represents integer numbers with no upper limit on the number of digits, while the latter represents decimal numbers (aka floating-point numbers) with an arbitrary number of decimal places (vs `f32` which can support 5 decimal places and `f64` which supports 15 decimal places).

## Examples
TODO: Coming soon...

## About DrHall-BigDecimal
DrHall-BigDecimal works by doing arithmatic the same way a human would: using a base-10 number system and operating on the digits one-at-a-time. This isn't particularly efficient on computers (which use base-2 number systems), but performance is a non-goal (see below) for this library. If you need speed, use the built-in `f64` type instead.

### Goals
* **Ergonomic** - easy to use
* **Simple** - easy to understand
* **Accurate** - if you didn't need accuracy, you'd use `f64` instead
* **Basic Arithmatic** - add, subtract, multiply, divide, power, and logarithm
* **Stable** - not afraid to lock-in the API
* **Tested** - test-driven development

### Non-Goals
* **Performance** - DrHall-BigDecimal is not trying to be particularly fast nor memory efficient
* **Trigonometry** - no sine functions or other complicated math functions
* **Complex Numbers** - imaginary numbers not supported
* **Cool** - templates, macros, and other fancy features are generally avoided
* **BigDecimal.java** - this library is inspired by Java's BigDecimal class, but isn't trying to reach feature-parity with it 
* **Perfect** - prioritizing release of a decent stable API over development 
  of a perfect API

