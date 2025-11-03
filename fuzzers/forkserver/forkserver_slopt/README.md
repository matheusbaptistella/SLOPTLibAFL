# SLOPT Forkserver Fuzzer

## Usage
You can build this example by running the command:
```
cargo build --release --bins
```

Then you can fuzz an intrumented target with the command:
```
target/release/afl-fuzz -t 2000 -i /path/to/seeds -o /path/to/output -- /path/to/target @@
```

## Additional Information
The file `results.zip` contains the results collected from the the experiments of this implementation. Also, under the `seeds` folder you can find the seeds used for the fuzzing campaign of the PUTs mentioned in the publication.

The folder `pbs-scripts/` contains the PBS script utilized to start the fuzzing camapigns on the cluster.