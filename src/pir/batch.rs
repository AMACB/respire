use crate::pir::pir::{Respire, RespireAliases};
use rand::{thread_rng, Rng};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

pub trait BatchRespire: Respire {
    type BaseRespire: Respire + RespireAliases;
    const NUM_BUCKET: usize;
    const BUCKET_SIZE: usize;
    // fn encode_batch_db<I: ExactSizeIterator<Item = <Self::BaseRespire as Respire>::RecordBytes>>(
    //     records_iter: I,
    // ) -> Vec<<Self::BaseRespire as Respire>::Database>;
}

pub struct BatchRespireImpl<const NUM_DB: usize, BaseRespire: Respire + RespireAliases> {
    phantom: PhantomData<BaseRespire>,
}

impl<const NUM_BUCKET: usize, BaseRespire: Respire + RespireAliases> BatchRespire
    for BatchRespireImpl<NUM_BUCKET, BaseRespire>
{
    type BaseRespire = BaseRespire;
    const NUM_BUCKET: usize = NUM_BUCKET;
    const BUCKET_SIZE: usize = <Self::BaseRespire as RespireAliases>::DB_SIZE;
}

impl<const NUM_DB: usize, BaseRespire: Respire + RespireAliases> Respire
    for BatchRespireImpl<NUM_DB, BaseRespire>
{
    type QueryKey = BaseRespire::QueryKey;
    type PublicParams = BaseRespire::PublicParams;
    type Queries = ();
    type ResponseRaw = BaseRespire::ResponseRaw;
    type Response = BaseRespire::Response;
    type Record = BaseRespire::Record;
    type RecordPackedSmall = BaseRespire::RecordPackedSmall;
    type RecordPacked = BaseRespire::RecordPacked;
    type Database = Vec<<BaseRespire as Respire>::Database>;
    type RecordBytes = BaseRespire::RecordBytes;
    const NUM_RECORDS: usize = 0;

    fn encode_db<I: ExactSizeIterator<Item = Self::RecordBytes>>(
        records_iter: I,
    ) -> Self::Database {
        let mut buckets = vec![Vec::new(); Self::NUM_BUCKET];
        for (i, r) in records_iter.enumerate() {
            let (b1, b2, b3) = Self::idx_to_buckets(i);
            buckets[b1].push((i, r.clone()));
            buckets[b2].push((i, r.clone()));
            buckets[b3].push((i, r.clone()));
        }
        let max_count = buckets.iter().map(|b| b.len()).max().unwrap();
        assert!(max_count <= BaseRespire::DB_SIZE);

        let mut result = Vec::with_capacity(Self::NUM_BUCKET);
        for b in buckets.iter() {
            // TODO cuckoo inside bucket
            result.push(BaseRespire::encode_db(b.iter().map(|(_, r)| r.clone())));
        }
        result
    }

    fn setup() -> (Self::QueryKey, Self::PublicParams) {
        BaseRespire::setup()
    }

    fn query(qk: &Self::QueryKey, idx: &[usize]) -> Self::Queries {
        todo!()
    }

    fn answer(
        pp: &Self::PublicParams,
        db: &Self::Database,
        q: &Self::Queries,
    ) -> Self::ResponseRaw {
        todo!()
    }

    fn response_compress(pp: &Self::PublicParams, r: &Self::ResponseRaw) -> Self::Response {
        todo!()
    }

    fn response_extract(qk: &Self::QueryKey, r: &Self::Response) -> Self::RecordPackedSmall {
        todo!()
    }

    fn response_decode(r: &Self::RecordPackedSmall) -> Vec<Self::RecordBytes> {
        todo!()
    }

    fn response_raw_stats(
        qk: &<Self as Respire>::QueryKey,
        r: &<Self as Respire>::ResponseRaw,
    ) -> f64 {
        todo!()
    }
}

impl<const NUM_BUCKET: usize, BaseRespire: Respire + RespireAliases>
    BatchRespireImpl<NUM_BUCKET, BaseRespire>
{
    fn idx_to_buckets(i: usize) -> (usize, usize, usize) {
        let modulus = Self::NUM_BUCKET as u64;
        assert!(modulus < 2_u64.pow(21)); // need three hashes from a u64
                                          // TODO: DefaultHasher is not stable
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let hashed = hasher.finish();
        let h1 = hashed % modulus;
        let h2 = (hashed / modulus) % modulus;
        let h3 = (hashed / modulus / modulus) % modulus;
        (h1 as usize, h2 as usize, h3 as usize)
    }

    fn cuckoo(idxs: &Vec<usize>, max_depth: usize) -> Option<Vec<Option<usize>>> {
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
