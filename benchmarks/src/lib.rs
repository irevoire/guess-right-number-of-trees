#![allow(clippy::type_complexity)]

mod arroy_bench;
mod dataset;
mod qdrant;

use std::fmt;
use std::fmt::Write;

use arroy::distances::*;
use arroy::ItemId;
use arroy_bench::measure_arroy_distance;
pub use dataset::*;
use qdrant_client::qdrant::quantization_config;

use crate::qdrant::measure_qdrant_distance;

pub const RECALL_TESTED: [usize; 6] = [1, 10, 20, 50, 100, 500];
pub const RNG_SEED: u64 = 38;

pub fn bench_over_all_distances(dimensions: usize, vectors: &[(u32, &[f32])]) {
    println!("\x1b[1m{}\x1b[0m vectors are used for this measure", vectors.len());
    let mut recall_tested = String::new();
    RECALL_TESTED.iter().for_each(|recall| write!(&mut recall_tested, "{recall:4}, ").unwrap());
    let recall_tested = recall_tested.trim_end_matches(", ");
    println!("Recall tested is:             [{recall_tested}]");

    for func in &[
        // arroy
        bench_arroy_distance::<Angular, 1, 100>(),
        bench_arroy_distance::<BinaryQuantizedAngular, 1, 100>(),
        bench_arroy_distance::<BinaryQuantizedAngular, 3, 100>(),
        bench_arroy_distance::<Angular, 1, 50>(),
        bench_arroy_distance::<BinaryQuantizedAngular, 1, 50>(),
        bench_arroy_distance::<BinaryQuantizedAngular, 3, 50>(),
        bench_arroy_distance::<Angular, 1, 2>(),
        bench_arroy_distance::<BinaryQuantizedAngular, 1, 2>(),
        bench_arroy_distance::<BinaryQuantizedAngular, 3, 2>(),
        // qdrant
        bench_qdrant_distance::<Angular, false, 100>(),
        bench_qdrant_distance::<Angular, true, 100>(),
        bench_qdrant_distance::<BinaryQuantizedAngular, false, 100>(),
        bench_qdrant_distance::<BinaryQuantizedAngular, true, 100>(),
        bench_qdrant_distance::<Angular, false, 50>(),
        bench_qdrant_distance::<Angular, true, 50>(),
        bench_qdrant_distance::<BinaryQuantizedAngular, false, 50>(),
        bench_qdrant_distance::<BinaryQuantizedAngular, true, 50>(),
        bench_qdrant_distance::<Angular, false, 2>(),
        bench_qdrant_distance::<Angular, true, 2>(),
        bench_qdrant_distance::<BinaryQuantizedAngular, false, 2>(),
        bench_qdrant_distance::<BinaryQuantizedAngular, true, 2>(),
        // bench_arroy_distance::<Angular, 1>(),
        // bench_qdrant_distance::<BinaryQuantizedAngular, false>(),
        // bench_qdrant_distance::<BinaryQuantizedAngular, true>(),
        // bench_arroy_distance::<BinaryQuantizedAngular, 1>(),
        // bench_arroy_distance::<BinaryQuantizedAngular, 3>(),
        // manhattan
        // bench_qdrant_distance::<Manhattan, false>(),
        // bench_arroy_distance::<Manhattan, 1>(),
        // bench_qdrant_distance::<BinaryQuantizedManhattan, false>(),
        // bench_arroy_distance::<BinaryQuantizedManhattan, 1>(),
        // bench_arroy_distance::<BinaryQuantizedManhattan, 3>(),
        // euclidean
        // bench_qdrant_distance::<Euclidean, false>(),
        // bench_arroy_distance::<Euclidean, 1>(),
        // bench_qdrant_distance::<BinaryQuantizedEuclidean, false>(),
        // bench_arroy_distance::<BinaryQuantizedEuclidean, 1>(),
        // bench_arroy_distance::<BinaryQuantizedEuclidean, 3>(),
        // dot-product
        // bench_qdrant_distance::<DotProduct, false>(),
        // bench_arroy_distance::<DotProduct, 1>(),
    ] {
        (func)(dimensions, vectors);
    }
}

/// A generalist distance trait that contains the informations required to configure every engine
trait Distance {
    const BINARY_QUANTIZED: bool;
    const QDRANT_DISTANCE: qdrant_client::qdrant::Distance;
    type ArroyDistance: arroy::Distance;

    fn name() -> &'static str;
    fn qdrant_quantization_config() -> quantization_config::Quantization;
    fn real_distance(a: &[f32], b: &[f32]) -> f32;
}

macro_rules! arroy_distance {
    ($distance:ty => real: $real:ident, qdrant: $qdrant:ident, bq: $bq:expr) => {
        impl Distance for $distance {
            const BINARY_QUANTIZED: bool = $bq;
            const QDRANT_DISTANCE: qdrant_client::qdrant::Distance =
                qdrant_client::qdrant::Distance::$qdrant;
            type ArroyDistance = $distance;

            fn name() -> &'static str {
                stringify!($distance)
            }
            fn qdrant_quantization_config() -> quantization_config::Quantization {
                qdrant_client::qdrant::BinaryQuantization::default().into()
            }
            fn real_distance(a: &[f32], b: &[f32]) -> f32 {
                let a = ndarray::aview1(a);
                let b = ndarray::aview1(b);
                fast_distances::$real(&a, &b)
            }
        }
    };
}

arroy_distance!(BinaryQuantizedAngular => real: cosine, qdrant: Cosine, bq: true);
arroy_distance!(Angular =>  real: cosine, qdrant: Cosine, bq: false);
arroy_distance!(BinaryQuantizedEuclidean => real: euclidean, qdrant: Euclid, bq: true);
arroy_distance!(Euclidean => real: euclidean, qdrant: Euclid, bq: false);
arroy_distance!(BinaryQuantizedManhattan => real: manhattan, qdrant: Manhattan, bq: true);
arroy_distance!(Manhattan => real: manhattan, qdrant: Manhattan, bq: false);
// arroy_distance!(DotProduct => real: dot, qdrant: Dot);

fn bench_arroy_distance<
    D: Distance,
    const OVERSAMPLING: usize,
    const FILTER_SUBSET_PERCENT: usize,
>() -> fn(usize, &[(u32, &[f32])]) {
    measure_arroy_distance::<D, OVERSAMPLING, FILTER_SUBSET_PERCENT>
}

fn bench_qdrant_distance<D: Distance, const EXACT: bool, const FILTER_SUBSET_PERCENT: usize>(
) -> fn(usize, &[(u32, &[f32])]) {
    measure_qdrant_distance::<D, EXACT, FILTER_SUBSET_PERCENT>
}

fn partial_sort_by<'a, D: crate::Distance>(
    mut vectors: impl Iterator<Item = (ItemId, &'a [f32])>,
    sort_by: &[f32],
    elements: usize,
) -> Vec<(ItemId, &'a [f32], f32)> {
    let mut ret = Vec::with_capacity(elements);
    ret.extend(vectors.by_ref().take(elements).map(|(i, v)| (i, v, distance::<D>(sort_by, v))));
    ret.sort_by(|(_, _, left), (_, _, right)| left.total_cmp(right));

    if ret.is_empty() {
        return ret;
    }

    for (item_id, vector) in vectors {
        let distance = distance::<D>(sort_by, vector);
        if distance < ret.last().unwrap().2 {
            match ret.binary_search_by(|(_, _, d)| d.total_cmp(&distance)) {
                Ok(i) | Err(i) => {
                    ret.pop();
                    ret.insert(i, (item_id, vector, distance))
                }
            }
        }
    }

    ret
}

fn distance<D: crate::Distance>(left: &[f32], right: &[f32]) -> f32 {
    D::real_distance(left, right)
}

pub struct Recall(pub f32);

impl fmt::Debug for Recall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            // red
            f32::NEG_INFINITY..=0.25 => write!(f, "\x1b[1;31m")?,
            // yellow
            0.25..=0.5 => write!(f, "\x1b[1;33m")?,
            // green
            0.5..=0.75 => write!(f, "\x1b[1;32m")?,
            // blue
            0.75..=0.90 => write!(f, "\x1b[1;34m")?,
            // cyan
            0.90..=0.999 => write!(f, "\x1b[1;36m")?,
            // underlined cyan
            0.999..=f32::INFINITY => write!(f, "\x1b[1;4;36m")?,
            _ => (),
        }
        write!(f, "{:.2}\x1b[0m", self.0)
    }
}
