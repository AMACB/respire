use crate::pir::pir::{Respire, PIR};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

pub trait BatchRespire: PIR {
    type BaseRespire: PIR + Respire;
    const NUM_BUCKET: usize;
}

pub struct BatchRespireImpl<
    const BATCH_SIZE: usize,
    const NUM_BUCKET: usize,
    const NUM_RECORDS: usize,
    BaseRespire: PIR + Respire,
> {
    phantom: PhantomData<BaseRespire>,
}

impl<
        const BATCH_SIZE: usize,
        const NUM_BUCKET: usize,
        const NUM_RECORDS: usize,
        BaseRespire: PIR + Respire,
    > BatchRespire for BatchRespireImpl<BATCH_SIZE, NUM_BUCKET, NUM_RECORDS, BaseRespire>
{
    type BaseRespire = BaseRespire;
    const NUM_BUCKET: usize = NUM_BUCKET;
}

impl<
        const BATCH_SIZE: usize,
        const NUM_BUCKET: usize,
        const NUM_RECORDS: usize,
        BaseRespire: PIR + Respire,
    > PIR for BatchRespireImpl<BATCH_SIZE, NUM_BUCKET, NUM_RECORDS, BaseRespire>
{
    type QueryKey = BaseRespire::QueryKey;
    type PublicParams = BaseRespire::PublicParams;
    type Query = ();
    type Response = BaseRespire::Response;
    type Database = Vec<<BaseRespire as PIR>::Database>;
    type RecordBytes = BaseRespire::RecordBytes;
    const NUM_RECORDS: usize = NUM_RECORDS;
    const BATCH_SIZE: usize = BATCH_SIZE;

    fn encode_db<I: ExactSizeIterator<Item = Self::RecordBytes>>(
        records_iter: I,
    ) -> Self::Database {
        let record_count = records_iter.len();
        let mut bucket_layouts = vec![Vec::with_capacity(BaseRespire::DB_SIZE); Self::NUM_BUCKET];
        let records = records_iter.collect_vec();
        for (i, r) in records.iter().enumerate() {
            let (b1, b2, b3) = Self::idx_to_buckets(i);
            bucket_layouts[b1].push(Some(i));
            bucket_layouts[b2].push(Some(i));
            bucket_layouts[b3].push(Some(i));
        }
        let max_count = bucket_layouts.iter().map(|b| b.len()).max().unwrap();
        eprintln!(
            "Encoding batch DB with {} records ({} buckets, {} base db size, {} max used size)",
            record_count,
            Self::NUM_BUCKET,
            BaseRespire::DB_SIZE,
            max_count
        );
        assert!(max_count <= BaseRespire::DB_SIZE);

        for b in bucket_layouts.iter_mut() {
            while b.len() < BaseRespire::DB_SIZE {
                b.push(None);
            }
        }

        let mut result = Vec::with_capacity(Self::NUM_BUCKET);
        let zero = Self::RecordBytes::default();
        for b in bucket_layouts.iter() {
            let bucket_records = b
                .iter()
                .map(|x| x.map_or(zero.clone(), |i| records[i].clone()));
            result.push(BaseRespire::encode_db(bucket_records));
        }
        result
    }

    fn setup() -> (Self::QueryKey, Self::PublicParams) {
        BaseRespire::setup()
    }

    fn query(qk: &Self::QueryKey, idxs: &[usize]) -> Self::Query {
        assert_eq!(BaseRespire::BATCH_SIZE, 1);
        let cuckooed = Self::cuckoo(idxs, 2usize.pow(16)).unwrap();
        assert_eq!(cuckooed.len(), Self::NUM_BUCKET);
        todo!()
    }

    fn answer(
        pp: &Self::PublicParams,
        db: &Self::Database,
        q: &Self::Query,
        qk: Option<&Self::QueryKey>,
    ) -> Self::Response {
        todo!()
    }

    fn extract(qk: &Self::QueryKey, r: &Self::Response) -> Vec<Self::RecordBytes> {
        todo!()
    }
}

impl<
        const BATCH_SIZE: usize,
        const NUM_BUCKET: usize,
        const NUM_RECORDS: usize,
        BaseRespire: PIR + Respire,
    > BatchRespireImpl<BATCH_SIZE, NUM_BUCKET, NUM_RECORDS, BaseRespire>
{
    fn idx_to_buckets(i: usize) -> (usize, usize, usize) {
        let modulus = Self::NUM_BUCKET as u64;
        assert!(modulus.checked_pow(3).is_some());
        // TODO: DefaultHasher is not stable
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let hashed = hasher.finish();
        let h1 = hashed % modulus;
        let h2 = (hashed / modulus) % modulus;
        let h3 = (hashed / modulus / modulus) % modulus;
        (h1 as usize, h2 as usize, h3 as usize)
    }

    fn cuckoo(idxs: &[usize], max_depth: usize) -> Option<Vec<Option<usize>>> {
        let mut result = vec![None; Self::NUM_BUCKET];
        let mut remaining = Vec::from_iter(idxs.iter().copied().map(|x| (x, 0usize)));
        let mut rng = thread_rng();
        while let Some((idx, depth)) = remaining.pop() {
            if depth >= max_depth {
                return None;
            }
            let (i1, i2, i3) = Self::idx_to_buckets(idx);
            match (result[i1], result[i2], result[i3]) {
                (None, _, _) => {
                    result[i1] = Some(idx);
                }
                (_, None, _) => {
                    result[i2] = Some(idx);
                }
                (_, _, None) => {
                    result[i3] = Some(idx);
                }
                (Some(curr1), Some(curr2), Some(curr3)) => match rng.gen_range(0..3) {
                    0 => {
                        remaining.push((curr1, depth + 1));
                        result[i1] = Some(idx);
                    }
                    1 => {
                        remaining.push((curr2, depth + 1));
                        result[i2] = Some(idx);
                    }
                    _ => {
                        remaining.push((curr3, depth + 1));
                        result[i3] = Some(idx);
                    }
                },
            }
        }
        Some(result)
    }

    // fn idx_to_bucket_pos(i: usize) -> (usize, usize) {
    //     let modulus = Self::BUCKET_SIZE as u64;
    //     assert!(modulus < 2_u64.pow(32)); // need two hashes from a u64
    //     // TODO: DefaultHasher is not stable
    //     let mut hasher = DefaultHasher::new();
    //     i.hash(&mut hasher);
    //     let hashed = hasher.finish();
    //     let h1 = hashed % modulus;
    //     let h2 = (hashed / modulus) % modulus;
    //     (h1 as usize, h2 as usize)
    // }

    // fn cuckoo(items: &Vec<usize>, bucket_count: usize) -> Option<Vec<Option<usize>>> {
    //     let mut result = vec![None; bucket_count];
    //     let mut remaining = Vec::from_iter(items.iter().copied());
    //     let mut rng = thread_rng();
    //     while let Some(idx) = remaining.pop() {
    //         let (i1, i2) = Self::idx_to_bucket_pos(idx);
    //         match (result[i1], result[i2]) {
    //             (None, _) => {
    //                 result[i1] = Some(idx);
    //             },
    //             (_, None) => {
    //                 result[i2] = Some(idx);
    //             },
    //             (Some(curr1), Some(curr2)) => {
    //                 if rng.gen() {
    //                     remaining.push(curr2);
    //                     result[i2] = Some(idx);
    //                 } else {
    //                     remaining.push(curr1);
    //                     result[i1] = Some(idx);
    //                 }
    //             }
    //         }
    //     }
    //     result
    // }
}
