use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use arroy::internals::{self, NodeCodec};
use arroy::{Database, Distance, ItemId, Writer, WriterProgress};
use byte_unit::{Byte, UnitType};
use heed::EnvOpenOptions;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use roaring::RoaringBitmap;

use crate::Recall;
use crate::{scenarios::*, IndexingMetrics};
const TWENTY_HUNDRED_MIB: usize = 2000 * 1024 * 1024 * 1024;

pub fn prepare_and_run<D, F>(
    line: &mut String,
    points: &[(u32, &[f32])],
    nb_trees: Option<usize>,
    number_of_chunks: usize,
    sleep_between_chunks: usize,
    memory: usize,
    verbose: bool,
    execute: F,
) where
    D: Distance,
    F: FnOnce(&mut String, &IndexingMetrics, &heed::Env, Database<D>),
{
    let dimensions = points[0].1.len();

    let dir = tempfile::tempdir().unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path()) }.unwrap();

    let mut arroy_seed = StdRng::seed_from_u64(13);
    let mut wtxn = env.write_txn().unwrap();
    let database =
        env.create_database::<internals::KeyCodec, NodeCodec<D>>(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();

    let duration = load_into_arroy(
        line,
        &mut arroy_seed,
        &env,
        database,
        dimensions,
        memory,
        points,
        number_of_chunks,
        nb_trees,
        sleep_between_chunks,
        verbose,
    );

    (execute)(line, &duration, &env, database);
}

#[allow(clippy::too_many_arguments)]
pub fn run_scenarios<D: Distance>(
    line: &mut String,
    env: &heed::Env,
    time_to_index: &IndexingMetrics,
    distance: &ScenarioDistance,
    number_of_chunks: usize,
    search: &[&ScenarioSearch],
    queries: &[(&u32, &&[f32], HashMap<ScenarioFiltering, (Option<RoaringBitmap>, Vec<u32>)>)],
    recall_tested: &[usize],
    database: arroy::Database<D>,
) {
    let mut recalls = Vec::new();

    for ScenarioSearch { oversampling, filtering } in search {
        let mut time_to_search = Duration::default();
        for &number_fetched in recall_tested {
            let (correctly_retrieved, duration) = queries
                .par_iter()
                .map(|(&id, _target, relevants)| {
                    let rtxn = env.read_txn().unwrap();
                    let reader = arroy::Reader::open(&rtxn, 0, database).unwrap();

                    let (candidates, relevants) = &relevants[filtering];
                    // Only keep the top number fetched documents.
                    let relevants = relevants.get(..number_fetched).unwrap_or(relevants);

                    let now = std::time::Instant::now();
                    let mut nns = reader.nns(number_fetched);
                    if let Some(oversampling) = oversampling.to_non_zero_usize() {
                        nns.oversampling(oversampling);
                    }
                    if let Some(candidates) = candidates.as_ref() {
                        nns.candidates(candidates);
                    }
                    let arroy_answer = nns.by_item(&rtxn, id).unwrap().unwrap();
                    let elapsed = now.elapsed();

                    let mut correctly_retrieved = Some(0);
                    for (id, _dist) in arroy_answer {
                        if relevants.contains(&id) {
                            if let Some(cr) = &mut correctly_retrieved {
                                *cr += 1;
                            }
                        } else if let Some(cand) = candidates.as_ref() {
                            // We set the counter to -1 if we return a filtered out candidated
                            if !cand.contains(id) {
                                correctly_retrieved = None;
                            }
                        }
                    }

                    (correctly_retrieved, elapsed)
                })
                .reduce(
                    || (Some(0), Duration::default()),
                    |(aanswer, aduration), (banswer, bduration)| {
                        (aanswer.zip(banswer).map(|(a, b)| a + b), aduration + bduration)
                    },
                );

            time_to_search += duration;
            // If non-candidate documents are returned we show a recall of -1
            let recall =
                correctly_retrieved.map_or(-1.0, |cr| cr as f32 / (number_fetched as f32 * 100.0));
            recalls.push(Recall(recall));
        }
    }

    let recall_score = recalls.iter().map(|r| r.0).sum::<f32>() / recalls.len() as f32;
    line.push_str(&format!("{recall_score:#.2},"));
    for recall in &recalls {
        line.push_str(&format!("{:#.2},", recall.0));
    }
}

#[allow(clippy::too_many_arguments)]
fn load_into_arroy<D: arroy::Distance>(
    line: &mut String,
    rng: &mut StdRng,
    env: &heed::Env,
    database: Database<D>,
    dimensions: usize,
    memory: usize,
    points: &[(ItemId, &[f32])],
    number_of_chunks: usize,
    nb_trees: Option<usize>,
    sleep_between_chunks: usize,
    verbose: bool,
) -> IndexingMetrics {
    let mut metrics = IndexingMetrics::new();
    let avg_chunk_size = points.len() / number_of_chunks;
    let mut nb_vectors = 0;
    let (progress_sender, progress_receiver) = std::sync::mpsc::channel();

    if verbose {
        std::thread::spawn(move || log_progress(progress_receiver));
    }

    for points in points.chunks(avg_chunk_size) {
        if sleep_between_chunks != 0 {
            std::thread::sleep(Duration::from_secs(sleep_between_chunks as u64));
        }
        tracing::info!("Inserting chunk of size {} in arroy", points.len());
        let mut wtxn = env.write_txn().unwrap();
        metrics.start_insertion();
        let writer = Writer::<D>::new(database, 0, dimensions);
        for (i, vector) in points.iter() {
            assert_eq!(vector.len(), dimensions);
            writer.add_item(&mut wtxn, *i, vector).unwrap();
        }
        metrics.end_insertion();

        tracing::info!("Starts building the trees");

        let mut builder = writer.builder(rng);
        if let Some(nb_trees) = nb_trees {
            builder.n_trees(nb_trees);
        }
        if verbose {
            builder.progress(|progress| progress_sender.send(progress).unwrap());
        }
        metrics.start_building();
        builder.available_memory(memory).build(&mut wtxn).unwrap();
        metrics.end_building();
        wtxn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();
        let reader = arroy::Reader::open(&rtxn, 0, database).unwrap();
        metrics.new_nb_trees(reader.n_trees());
        drop(rtxn);

        nb_vectors += points.len();
        metrics.new_nb_vectors(nb_vectors);
        metrics.new_database_size(env.non_free_pages_size().unwrap() as usize);
        line.push_str(&format!("{},", env.non_free_pages_size().unwrap()));
    }

    metrics.end();
    metrics
}

fn log_progress(recv: Receiver<WriterProgress>) {
    let mut time = std::time::Instant::now();
    let mut last_progress = None;

    loop {
        let progress = recv.recv_timeout(Duration::from_secs(10));
        match progress {
            Ok(progress) => {
                tracing::info!("    {progress:?}, last step took: {:.2?}", time.elapsed());
                last_progress = Some(progress);
                time = std::time::Instant::now();
            }
            Err(RecvTimeoutError::Disconnected) => {
                break;
            }
            Err(RecvTimeoutError::Timeout) => {
                if let Some(ref last_progress) = last_progress {
                    if let Some(ref sub) = last_progress.sub {
                        let current = sub.current.load(Ordering::Relaxed);
                        let percentage = current as f32 / sub.max as f32 * 100.0;
                        tracing::info!("    {last_progress:?}, has been running for: {:.2?}, {percentage:#.2}%", time.elapsed());
                    }
                }
            }
        }
    }
}
