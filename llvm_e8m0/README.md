# probing llvm's APFloat8 e8m0 data type's rounding behavior

A script to verify that LLVM's float32 -> e8m0 casts uses round to nearest, ties to even rounding - it does!

To use E8M0 in LLVM you need LLVM v20.0+, I had to build LLVM from source since latest released version was 19.x.

To run:

```
./run.sh
```
