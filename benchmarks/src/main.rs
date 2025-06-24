use std::collections::HashMap;
use std::fmt::Write as _;

use arroy::distances::Cosine;
use benchmarks::scenarios::ScenarioSearch;
use benchmarks::{arroy_bench, scenarios, MatLEView, RNG_SEED};
use byte_unit::Byte;
use clap::Parser;
use enum_iterator::Sequence;
use itertools::{iproduct, Itertools};
use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::seq::SliceRandom as _;
use rand::SeedableRng;
use rayon::slice::ParallelSliceMut;
use roaring::RoaringBitmap;
use slice_group_by::GroupBy;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

fn parse_number_with_underscores(s: &str) -> Result<usize, std::num::ParseIntError> {
    s.replace('_', "").parse()
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The datasets to run and all of them are ran if empty.
    #[arg(long, value_enum)]
    datasets: Vec<scenarios::Dataset>,

    /// Ignored
    #[arg(long, value_enum)]
    contenders: Vec<scenarios::ScenarioContender>,

    /// Ignored
    #[arg(long, value_enum)]
    distances: Vec<scenarios::ScenarioDistance>,

    /// Ignored
    #[arg(long, value_enum)]
    over_samplings: Vec<scenarios::ScenarioOversampling>,

    /// Ignored
    #[arg(long, value_enum)]
    filterings: Vec<scenarios::ScenarioFiltering>,

    /// The list of recall to be tested.
    #[arg(long, default_value_t = String::from("1,10,20,50,100,500"))]
    recall_tested: String,

    /// Set the different number of documents to evaluate from the dataset.
    #[arg(long, value_delimiter = ',', value_parser = parse_number_with_underscores)]
    count: Vec<usize>,

    /// Set different number of trees to generate to try for each number of documents
    #[arg(long, value_delimiter = ',')]
    nb_trees: Vec<usize>,

    /// These numbers correspond to the numbers of chunks that the dataset will be split into for indexing.
    ///
    /// Each number corresponds to a new indexation in x chunks. Use a comma to separate multiple features.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    number_of_chunks: Vec<usize>,

    /// The time to sleep between each chunk indexing specified in seconds.
    ///
    /// This is useful when profiling, it helps quickly identifying when each steps took place.
    /// Also, it's not counted in any of the individual reported indexing time metrics but it is counted in the total indexing time.
    #[arg(long, default_value_t = 0)]
    sleep_between_chunks: usize,

    /// Memory available for indexing.
    #[arg(long, default_value_t = Byte::MAX)]
    memory: Byte,

    /// The number of threads to use for indexing. If not specified the maximum number of threads will be used.
    #[arg(long)]
    threads: Option<usize>,

    /// When set to true, will print all the steps it goes through.
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() {
    let Args {
        datasets,
        count,
        nb_trees,
        number_of_chunks,
        contenders,
        distances,
        over_samplings,
        filterings,
        sleep_between_chunks,
        memory,
        recall_tested,
        threads,
        verbose,
    } = Args::parse();

    if verbose {
        // Initialize tracing with the specified level
        let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            let filter = format!("arroy=debug,benchmarks=debug");
            EnvFilter::new(filter)
        });

        FmtSubscriber::builder()
            .with_env_filter(env_filter)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .init();
    }

    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    }

    let datasets = set_or_all::<_, MatLEView<f32>>(datasets);
    let contenders = vec![scenarios::ScenarioContender::Arroy];
    let distances = vec![scenarios::ScenarioDistance::Cosine];
    let over_samplings = vec![scenarios::ScenarioOversampling::X1];
    let filterings = vec![scenarios::ScenarioFiltering::NoFilter];
    let recall_tested: Vec<usize> = recall_tested
        .split(',')
        .enumerate()
        .filter(|(_, n)| !n.trim().is_empty())
        .map(|(i, n)| {
            n.trim()
                .parse()
                .unwrap_or_else(|_| panic!("Could not parse recall value `{n}` at index `{i}`."))
        })
        .collect();

    assert!(datasets.len() == 1, "Cannot use more than one dataset");
    assert!(number_of_chunks.len() == 1, "Cannot use more than one chunk");
    assert!(!nb_trees.is_empty(), "Must specify at least one number of trees with --nb-trees 1,2,3");
    assert!(!count.is_empty(), "Must specify at least one number of vectors with --count 1000,2000,3000");

    let scenaris: Vec<_> = iproduct!(datasets, distances, contenders, over_samplings, filterings)
        .map(|(dataset, distance, contender, oversampling, filtering)| {
            (dataset, distance, contender, ScenarioSearch { oversampling, filtering })
        })
        .sorted()
        .collect();

    let mut header = String::new();
    header.push_str(&format!("nb vectors,nb trees,db size in bytes,recall score,"));
    recall_tested.iter().for_each(|recall| write!(&mut header, "recall@{recall},").unwrap());
    let header = header.trim_end_matches(",");
    println!("{header}");

    for grp in scenaris
        .linear_group_by(|(da, dia, ca, _), (db, dib, cb, _)| da == db && dia == dib && ca == cb)
    {
        for count in &count {
            for nb_trees in &nb_trees {
                // need to be filled up for the end log
                let mut line = String::new();
                line.push_str(&format!("{count},{nb_trees},"));

                let (dataset, distance, contender, _) = &grp[0];
                let search: Vec<&ScenarioSearch> = grp.iter().map(|(_, _, _, s)| s).collect();

                let points: Vec<_> =
                    dataset.iter().take(*count).enumerate().map(|(i, v)| (i as u32, v)).collect();
                let memory = memory.as_u64() as usize;

                let max = recall_tested.iter().max().copied().unwrap_or_default();
                // If we have no recall we can skip entirely the generation of the queries
                let queries = if max == 0 {
                    Vec::new()
                } else {
                    let mut rng = StdRng::seed_from_u64(RNG_SEED);
                    (0..100)
                        .map(|_| points.choose(&mut rng).unwrap())
                        .map(|(id, target)| {
                            let mut points = points.clone();
                            points.par_sort_unstable_by_key(|(_, v)| {
                                OrderedFloat(benchmarks::distance::<Cosine>(target, v))
                            });

                            // We collect the different filtered versions here.
                            let filtered: HashMap<_, _> = search
                                .iter()
                                .map(|ScenarioSearch { filtering, .. }| {
                                    let candidates = match filtering {
                                        scenarios::ScenarioFiltering::NoFilter => None,
                                        filtering => {
                                            let total = points.len() as f32;
                                            let filtering = filtering.to_ratio_f32();
                                            Some(
                                                points
                                                    .iter()
                                                    .map(|(id, _)| id)
                                                    .take((total * filtering) as usize)
                                                    .collect::<RoaringBitmap>(),
                                            )
                                        }
                                    };

                                    // This is the real expected answer without the filtered out candidates.
                                    let answer = points
                                        .iter()
                                        .map(|(id, _)| *id)
                                        .filter(|&id| {
                                            candidates.as_ref().map_or(true, |c| c.contains(id))
                                        })
                                        .take(max)
                                        .collect::<Vec<_>>();

                                    (*filtering, (candidates, answer))
                                })
                                .collect();

                            (id, target, filtered)
                        })
                        .collect()
                };

                for number_of_chunks in &number_of_chunks {
                    match contender {
                        scenarios::ScenarioContender::Qdrant => {
                            println!("Qdrant is not supported yet")
                        }
                        scenarios::ScenarioContender::Arroy => match distance {
                            scenarios::ScenarioDistance::Cosine => {
                                arroy_bench::prepare_and_run::<Cosine, _>(
                                    &mut line,
                                    &points,
                                    Some(*nb_trees),
                                    *number_of_chunks,
                                    sleep_between_chunks,
                                    memory,
                                    verbose,
                                    |line,time_to_index, env, database| {
                                        arroy_bench::run_scenarios(
                                            line,
                                            env,
                                            time_to_index,
                                            distance,
                                            *number_of_chunks,
                                            &search,
                                            &queries,
                                            &recall_tested,
                                            database,
                                        );
                                    },
                                )
                            }
                        },
                    }
                }
                let line =line.trim_end_matches(",");
                println!("{line}");
            }
        }

        println!();
    }
}

fn set_or_all<S, T>(datasets: Vec<S>) -> Vec<T>
where
    S: Sequence,
    S: Into<T>,
{
    if datasets.is_empty() {
        enum_iterator::all::<S>().map(Into::into).collect()
    } else {
        datasets.into_iter().map(Into::into).collect()
    }
}
