use fhe_psi::{matrix::*, discrete_gaussian::*};
use fhe_psi::z_n::*;
use fhe_psi::ring_elem::*;
use rand::{SeedableRng};
use rand_chacha::ChaCha20Rng;

fn z_n_demo() {
    let x : Z_N<17> = Z_N::new_u(4);
    let y : Z_N<17> = Z_N::new_u(7);
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
fn gsw_keygen_demo() {
    const N : usize = 5;
    const M : usize = 10;
    const Q : u64 = 268369921;

    let dg = DiscreteGaussian::init(6.4);
    let mut rng = ChaCha20Rng::from_entropy();
    println!("small = {:?}", dg.sample::<Q>(&mut rng));

    let a_bar : Matrix<N, M, Z_N<Q>> = Matrix::random_rng(&mut rng);
    let s_bar_T : Matrix<1, N, Z_N<Q>>= Matrix::random_rng(&mut rng);
    let neg_s_bar_T = -&s_bar_T;
    let e : Matrix<1, M, Z_N<Q>> = dg.sample_int_matrix(&mut rng);
    let mut s_bar_T_a_bar = &s_bar_T * &a_bar;
    let mut s_bar_T_a_bar_plus_e = &s_bar_T_a_bar + &e;
    let mut _1 : Matrix<1, 1, Z_N<Q>> = Matrix::zero();
    _1[(0,0)] = Z_N::one();

    let mut a : Matrix<{N+1}, M, Z_N<Q>> = stack(&a_bar, &s_bar_T_a_bar_plus_e);
    let mut s_T : Matrix<1, {N+1}, Z_N<Q>> = append(&neg_s_bar_T, &_1);

    let e_hopefully = &s_T * &a;

    println!("{a:?}");
    println!("{s_T:?}");
    println!("{e:?}");
    println!("{e_hopefully:?}");
}

fn main() {
    z_n_demo();
    matrix_demo();
    gsw_keygen_demo();
}
