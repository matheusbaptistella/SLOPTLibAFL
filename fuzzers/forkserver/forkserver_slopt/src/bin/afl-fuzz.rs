use core::time::Duration;
use std::path::PathBuf;

use clap::Parser;
use libafl::{
    Error, corpus::{CachedOnDiskCorpus, Corpus, InMemoryCorpus, OnDiskCorpus}, events::SimpleEventManager, executors::{HasObservers, StdChildArgs, forkserver::ForkserverExecutor}, feedback_and_fast, feedback_or, feedbacks::{CrashFeedback, MaxMapFeedback, TimeFeedback}, fuzzer::{Fuzzer, StdFuzzer}, inputs::BytesInput, monitors::SimpleMonitor, mutators::{HavocScheduledMutator, havoc_mutations}, observers::{CanTrack, HitcountsMapObserver, StdMapObserver, TimeObserver}, schedulers::{
        IndexesLenTimeMinimizerScheduler, StdWeightedScheduler, powersched::PowerSchedule
    }, stages::{AflStatsStage, CalibrationStage, StdMutationalStage}, state::{HasCorpus, StdState}
};
use libafl_bolts::{
    AsSliceMut, StdTargetArgs, Truncate, current_nanos, fs::get_unique_std_input_file, rands::StdRand, shmem::{ShMem, ShMemProvider, UnixShMemProvider}, tuples::{Handled, tuple_list}
};

#[derive(Debug, Parser)]
#[command(
    name = "afl-fuzz",
    about = "This is an implementation of SLOPT using LibAFL components",
    author = "matheusbaptistella <mbapts@gmail.com>"
)]
struct Opt {
    #[arg(
        help = "The directory to read initial inputs from ('seeds')",
        short = 'i',
        required = true
    )]
    in_dir: PathBuf,

    #[arg(help = "The directory to write output", short = 'o', required = true)]
    out_dir: PathBuf,

    #[arg(
        help = "Timeout for each individual execution, in milliseconds",
        short = 't',
        long = "timeout",
        default_value = "1000"
    )]
    timeout: u64,

    #[arg(
        help = "If not set, the child's stdout and stderror will be redirected to /dev/null",
        short = 'd',
        long = "debug-child",
        default_value = "false"
    )]
    debug_child: bool,

    #[arg(
        help = "Target and arguments",
        name = "arguments",
        num_args(1..),
        allow_hyphen_values = true,
    )]
    arguments: Vec<String>,
}

pub fn main() {
    env_logger::init();
    const MAP_SIZE: usize = 1_048_576;
    const AFL_HARNESS_FILE_INPUT: &str = "@@";
    const AFL_DEFAULT_INPUT_LEN_MAX: usize = 1_048_576;
    const AFL_DEFAULT_INPUT_LEN_MIN: usize = 1;

    let opt = Opt::parse();
    let default_out_dir = opt.out_dir.join("default");

    let cur_input_dir =
        PathBuf::from(std::env::var("AFL_TMPDIR").expect("Provide a value in AFL_TMPDIR"));

    let (program, args) = opt
        .arguments
        .split_first()
        .expect("Missing target: pass `-- /path/to/target @@`");

    // The unix shmem provider supported by AFL++ for shared memory
    let mut shmem_provider = UnixShMemProvider::new().unwrap();

    // The coverage map shared between observer and executor
    let mut shmem = shmem_provider.new_shmem(MAP_SIZE).unwrap();
    // let the forkserver know the shmid
    unsafe {
        shmem.write_to_env("__AFL_SHM_ID").unwrap();
    }
    let shmem_buf = shmem.as_slice_mut();

    // Create an observation channel using the signals map
    let edges_observer = unsafe {
        HitcountsMapObserver::new(StdMapObserver::new("shared_mem", shmem_buf)).track_indices()
    };

    // Create an observation channel to keep track of the execution time
    let time_observer = TimeObserver::new("time");

    let map_feedback = MaxMapFeedback::new(&edges_observer);

    let calibration = CalibrationStage::new(&map_feedback);

    let mut feedback = feedback_or!(map_feedback, TimeFeedback::new(&time_observer));

    // A feedback to choose if an input is a solution or not
    // We want to do the same crash deduplication that AFL does
    let mut objective = feedback_and_fast!(
        // Must be a crash
        CrashFeedback::new(),
        // Take it only if trigger new coverage over crashes
        // Uses `with_name` to create a different history from the `MaxMapFeedback` in `feedback` above
        MaxMapFeedback::with_name("mapfeedback_metadata_objective", &edges_observer)
    );

    // create a State from scratch
    let mut state = StdState::new(
        // RNG
        StdRand::with_seed(current_nanos()),
        // Corpus that will be evolved, we keep it in memory for performance
        InMemoryCorpus::<BytesInput>::new(),
        // Corpus in which we store solutions (crashes in this example),
        // on disk so the user can get them after stopping the fuzzer
        OnDiskCorpus::new(default_out_dir.join("crashes")).unwrap(),
        // States of the feedbacks.
        // The feedbacks can report the data that should persist in the State.
        &mut feedback,
        // Same for objective feedbacks
        &mut objective,
    )
    .unwrap();

    // The Monitor trait define how the fuzzer stats are reported to the user
    let monitor = SimpleMonitor::new(|s| println!("{s}"));

    // The event manager handle the various events generated during the fuzzing loop
    // such as the notification of the addition of a new item to the corpus
    let mut mgr = SimpleEventManager::new(monitor);

    let scheduler = IndexesLenTimeMinimizerScheduler::new(
        &edges_observer,
        StdWeightedScheduler::with_schedule(
            &mut state,
            &edges_observer,
            Some(PowerSchedule::fast()),
        ),
    );

    // A fuzzer with feedbacks and a corpus scheduler
    let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

    let afl_stats_stage = AflStatsStage::builder()
        .map_observer(&edges_observer)
        .stats_file(default_out_dir.join("fuzzer_stats"))
        .plot_file(default_out_dir.join("plot_data"))
        .uses_autotokens(false)
        .exec_timeout(opt.timeout)
        .banner(program.clone())
        .version("afl-fuzz 0.0.1".to_string())
        .build()
        .expect("Failed to build afl stats");

    // let file = get_unique_std_input_file();
    let testcase_name = get_unique_std_input_file();
    let testcase_path =  cur_input_dir.join(&testcase_name);

    // let observer_ref = edges_observer.handle();

    let mut builder = ForkserverExecutor::builder()
        .program(program)
        .debug_child(opt.debug_child)
        .shmem_provider(&mut shmem_provider)
        // .parse_afl_cmdline(opt.arguments)
        .coverage_map_size(MAP_SIZE)
        .timeout(Duration::from_millis(opt.timeout))
        .kill_signal(nix::sys::signal::SIGKILL)
        .min_input_size(AFL_DEFAULT_INPUT_LEN_MIN)
        .max_input_size(AFL_DEFAULT_INPUT_LEN_MAX);
        // .arg_input_file(cur_input_dir.join(file))
        // .build_dynamic_map(edges_observer, tuple_list!(time_observer))
        // .unwrap();

    for arg in args {
        if arg == AFL_HARNESS_FILE_INPUT {
            builder = builder.arg_input_file(testcase_path.clone());
        }
        else {
            builder = builder.arg(arg);
        }
    }

    let mut executor = builder.build(tuple_list!(time_observer, edges_observer)).unwrap();
    // let mut executor = builder.build_dynamic_map(edges_observer, tuple_list!(time_observer))
    //     .unwrap();

    // if let Some(dynamic_map_size) = executor.coverage_map_size() {
    //     executor.observers_mut()[&observer_ref]
    //         .as_mut()
    //         .truncate(dynamic_map_size);
    // }
    

    // In case the corpus is empty (on first run), reset
    if state.must_load_initial_inputs() {
        state
            .load_initial_inputs(&mut fuzzer, &mut executor, &mut mgr, &[opt.in_dir.clone()])
            .unwrap_or_else(|err| panic!("Failed to load initial corpus: {err:?}"));
        println!("We imported {} inputs from disk.", state.corpus().count());
    }

    let mutator = HavocScheduledMutator::with_max_stack_pow(havoc_mutations(), 6);

    let mutational_stage = StdMutationalStage::new(mutator);

    let mut stages = tuple_list!(calibration, mutational_stage, afl_stats_stage);

    fuzzer
        .fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr)
        .expect("Error in the fuzzing loop");
}