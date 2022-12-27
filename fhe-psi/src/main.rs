use fhe_psi::{matrix::*, gsw::*, z_n::*, ring_elem::*, params::*};
use rand::{SeedableRng, Rng};
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
fn gsw_demo() {
    let (a, s_T) = test_keygen();
    let e_hopefully = &s_T * &a;

    println!("{a:?}");
    println!("{s_T:?}");
    println!("{e_hopefully:?}");

    let mu = 0;
    let ct = encrypt(&a, mu);
    println!("{mu:?}");
    println!("{ct:?}");
    let pt = decrypt(&s_T, &ct);
    println!("{pt:?}");

    let mu = 1;
    let ct = encrypt(&a, mu);
    println!("{mu:?}");
    println!("{ct:?}");
    let pt = decrypt(&s_T, &ct);
    println!("{pt:?}");

    for i in 0..100 {
        let mu = i % 2;
        let ct = encrypt(&a, mu);
        let pt = decrypt(&s_T, &ct);
        assert!(pt == mu);
        assert!(pt == mu);
    }
}
fn main() {
    z_n_demo();
    matrix_demo();
    gsw_demo();
}
