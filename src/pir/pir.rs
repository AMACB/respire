use std::fmt::{Display, Formatter};
use std::time::Duration;

pub trait PIRRecordBytes: Clone + Default {
    fn from_bytes(bytes: &[u8]) -> Option<Self>;
    fn as_bytes(&self) -> &[u8];
}

#[derive(Debug)]
struct TimeStat {
    name: &'static str,
    time: Duration,
}

impl TimeStat {
    fn new(name: &'static str, time: Duration) -> Self {
        Self { name, time }
    }
}

#[derive(Debug)]
pub struct TimeStats {
    time_stats: Vec<TimeStat>,
}

impl TimeStats {
    pub fn add(&mut self, name: &'static str, time: Duration) {
        self.time_stats.push(TimeStat::new(name, time));
    }
    pub fn new() -> Self {
        Self {
            time_stats: Vec::new(),
        }
    }
    
    pub fn total(&self) -> Duration {
        self.time_stats.iter().fold(Duration::new(0, 0), |acc, x| acc + x.time)
    }
}

impl Display for TimeStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (i, time_stat) in self.time_stats.iter().enumerate() {
            write!(f, "{}: {:?}", time_stat.name, time_stat.time)?;
            if i < self.time_stats.len() - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
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
        time_stats: Option<&mut TimeStats>,
    ) -> (Self::Database, Self::DatabaseHint);
    fn setup(time_stats: Option<&mut TimeStats>) -> (Self::QueryKey, Self::PublicParams);
    fn query(
        qk: &Self::QueryKey,
        idx: &[usize],
        db_hint: &Self::DatabaseHint,
        time_stats: Option<&mut TimeStats>,
    ) -> (Self::Query, Self::State);
    fn answer(
        pp: &Self::PublicParams,
        db: &Self::Database,
        q: &Self::Query,
        qk: Option<&Self::QueryKey>,
        time_stats: Option<&mut TimeStats>,
    ) -> Self::Response;
    fn extract(
        qk: &Self::QueryKey,
        r: &Self::Response,
        st: &Self::State,
        time_stats: Option<&mut TimeStats>,
    ) -> Vec<Self::RecordBytes>;
}
