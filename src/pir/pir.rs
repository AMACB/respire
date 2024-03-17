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
    type RecordBytes: Clone + Default;

    const NUM_RECORDS: usize;
    const BATCH_SIZE: usize;

    fn print_summary();

    fn encode_db<I: ExactSizeIterator<Item = Self::RecordBytes>>(
        records_iter: I,
    ) -> (Self::Database, Self::DatabaseHint);
    fn setup() -> (Self::QueryKey, Self::PublicParams);
    fn query(
        qk: &Self::QueryKey,
        idx: &[usize],
        db_hint: &Self::DatabaseHint,
    ) -> (Self::Query, Self::State);
    fn answer(
        pp: &Self::PublicParams,
        db: &Self::Database,
        q: &Self::Query,
        qk: Option<&Self::QueryKey>,
    ) -> Self::Response;
    fn extract(qk: &Self::QueryKey, r: &Self::Response, st: &Self::State)
        -> Vec<Self::RecordBytes>;
}
