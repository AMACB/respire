# Respire

This repository contains the implementation of the Respire private information retrieval protocol,
accompanying the paper *Respire: High-Rate PIR for Databases with Small Records*.

## Quickstart
Install a recent version of [Rust](https://www.rust-lang.org/tools/install) (`>= 1.76.0` is known to work).
Then, you can directly build and run the code with `cargo`:

```
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin <name> <number of trials>
```

The various parameter pre-defined configurations are under different binary names.
The list of these names can be obtained by running `cargo run --release --bin`.
The format of these names follows the format `<DB size with suffix><record size>_<batch size>`.
For example, `256m256_4` refers to a configuration with a 256 MB database (the `m` suffix indicating MB), 256 byte records, and a batch size of 4.

Upon running, the following information will be printed:

* Whether or not AVX2 is enabled (should be automatically enabled if your machine supports it)
* The database configuration
* The full parameter set
* The various communication sizes and the rate
* The estimated error rate (as computed by the error analysis)

Then, one-time setup is performed, and the specified number of trials of PIR queries are run.
The timings for trial will be printed as they are run, and finally a summary is printed at the end. 

Set the environment variable `RUST_LOG=info` to get more detailed information during execution.
Among other things, this will enable printing out intermediate values in the error analysis, as well as the measured error in each trial.

A simple one-liner to run 5 trials for each configuration is the following (run inside `src/bin`):
```shell
for name in `ls -v *.rs`; do
  RUSTFLAGS="-C target-cpu=native" cargo run --release --bin "${name%.*}" 5 2>&1 | tee "${name%.*}.out";
done
```
This will save the outputs into correponding `.out` files
