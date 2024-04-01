use std::collections::HashMap;
use std::ops::AddAssign;
use std::time::Duration;

pub trait PIRRecordBytes: Clone + Default {
    fn from_bytes(bytes: &[u8]) -> Option<Self>;
    fn as_bytes(&self) -> &[u8];
}

pub struct Stats<T: AddAssign<T> + Copy + Default> {
    order: Vec<&'static str>,
    stats: HashMap<&'static str, T>,
}

impl<T: AddAssign<T> + Copy + Default> Stats<T> {
    pub fn add(&mut self, name: &'static str, value: T) {
        match self.stats.get_mut(name) {
            Some(it) => {
                *it += value;
            }
            None => {
                self.order.push(name);
                self.stats.insert(name, value);
            }
        };
    }
    pub fn new() -> Self {
        Self {
            order: Vec::new(),
            stats: HashMap::new(),
        }
    }

    pub fn as_vec(&self) -> Vec<(&'static str, T)> {
        let mut result = Vec::with_capacity(self.order.len());
        for name in self.order.iter().copied() {
            result.push((name, self.stats[name]));
        }
        result
    }
}

pub trait PIR {
    // Associated types
    type QueryKey;
    type PublicParams;
    type Query;
    type Response;
    type Database;
    type DatabaseHint;
    type State;

    // A single raw record
    type RecordBytes: PIRRecordBytes;
    const BYTES_PER_RECORD: usize;
    const NUM_RECORDS: usize;
    const BATCH_SIZE: usize;

    fn print_summary();

    fn encode_db<F: Fn(usize) -> Self::RecordBytes>(
        records_generator: F,
        time_stats: Option<&mut Stats<Duration>>,
    ) -> (Self::Database, Self::DatabaseHint);
    fn setup(time_stats: Option<&mut Stats<Duration>>) -> (Self::QueryKey, Self::PublicParams);
    fn query(
        qk: &Self::QueryKey,
        idx: &[usize],
        db_hint: &Self::DatabaseHint,
        time_stats: Option<&mut Stats<Duration>>,
    ) -> (Self::Query, Self::State);
    fn answer(
        pp: &Self::PublicParams,
        db: &Self::Database,
        q: &Self::Query,
        qk: Option<&Self::QueryKey>,
        time_stats: Option<&mut Stats<Duration>>,
    ) -> Self::Response;
    fn extract(
        qk: &Self::QueryKey,
        r: &Self::Response,
        st: &Self::State,
        time_stats: Option<&mut Stats<Duration>>,
    ) -> Vec<Self::RecordBytes>;
}
