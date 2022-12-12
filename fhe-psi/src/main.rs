use fhe_psi::matrix::*;
use fhe_psi::z_n::*;
use fhe_psi::ring_elem::*;
use rand::{SeedableRng};
use rand_chacha::ChaCha20Rng;

fn z_n_demo() {
    let x : Z_N<17> = Z_N { a: 4 };
    let y : Z_N<17> = Z_N { a: 9 };
    println!("x+y = {:?}", x+y);
    println!("x*y = {:?}", x*y);
    println!("x-y = {:?}", x-y);
    println!("-x, -y = {:?}, {:?}", -x, -y);
}
fn matrix_demo() {
    let mut identity: Matrix<2,2,Z_N<17>> = Matrix::zero();
    identity[(0,0)] = Z_N::one();
    identity[(1,1)] = Z_N::one() + Z_N::one();
    let mut rng = ChaCha20Rng::from_entropy();
    let rand_mat : Matrix<2,4,Z_N<17>> = Matrix::random_rng(&mut rng);
    let prod = &identity * &rand_mat;
    println!("I = {:?}", identity);
    println!("R = {:?}", rand_mat);
    println!("I*R = {:?}", prod);
}
fn main() {
    z_n_demo();
    matrix_demo();
}
