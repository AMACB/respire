pub trait EncodingScheme {
    type Plaintext: Clone;
    type Ciphertext: Clone;
    type SecretKey: Clone;

    fn keygen() -> Self::SecretKey;
    fn encode(sk: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext;
}
