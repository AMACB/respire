use fhe_psi::{matrix::*, gsw::*, gadget::*, z_n::*, ring_elem::*, params::*};
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
    let (a, s_T) = keygen(DumbParams);
    // let e_hopefully = &s_T * &a;

    println!("{a:?}");
    println!("{s_T:?}");
    // println!("{e_hopefully:?}");

    for i in 0..10 {
        let mu = Z_N::new_u(i);
        let ct = encrypt(&a, mu);
        let pt = decrypt(&s_T, &ct);
        assert!(pt == mu);
    }
}
fn gadget_demo() {
    let mut rng = ChaCha20Rng::from_entropy();
    let R : Matrix<2,8,Z_N<13>> = Matrix::random_rng(&mut rng);
    let G = gen_G::<2,8,13,2,4>();
    let R_inv = G_inv::<2,8,8,13,2,4>(&R);
    let R_hopefully = &G * &R_inv;
    println!("{G:?}");
    println!("{R:?}");
    println!("{R_hopefully:?}");
}
fn fhe_demo() {
    let (a, s_T) = keygen(DumbParams);
    for i in 0..10 {
        for j in 0..10 {
            let mu1 = Z_N::new_u(i);
            let mu2 = Z_N::new_u(j);
            let ct1 = encrypt(&a, mu1);
            let ct2 = encrypt(&a, mu2);
            let pt_add = decrypt(&s_T, &(&ct1 + mu2));
            let pt_mul = decrypt(&s_T, &(&ct1 * mu2));
            let pt_add_ct = decrypt(&s_T, &(&ct1 + &ct2));
            let pt_mul_ct = decrypt(&s_T, &(&ct1 * &ct2));
            assert!(pt_add == mu1 + mu2);
            assert!(pt_mul == mu1 * mu2);
            assert!(pt_add_ct == mu1 + mu2);
            assert!(pt_mul_ct == mu1 * mu2);
        }
    }
}
fn main() {
    z_n_demo();
    matrix_demo();
    gsw_demo();
    gadget_demo();
    fhe_demo();
}

