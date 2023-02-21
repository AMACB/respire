// use fhe_psi::{z_n::*, gsw::*, params::*};

fn main() {
    // let (A, s_T) = gsw::keygen(TEST_PARAMS);
    // for i in 0..10 {
    //     for j in 0..10 {
    //         let mu1 = Z_N::new_u(i);
    //         let mu2 = Z_N::new_u(j);
    //         let ct1 = gsw::encrypt(&A, &mu1);
    //         let ct2 = gsw::encrypt(&A, &mu2);
    //         let pt_add = gsw::decrypt(&s_T, &(&ct1 + &mu2));
    //         let pt_mul = gsw::decrypt(&s_T, &(&ct1 * &mu2));
    //         let pt_add_ct = gsw::decrypt(&s_T, &(&ct1 + &ct2));
    //         let pt_mul_ct = gsw::decrypt(&s_T, &(&ct1 * &ct2));
    //         assert_eq!(pt_add, &mu1 + &mu2, "addition by scalar failed");
    //         assert_eq!(pt_add_ct, &mu1 + &mu2, "ciphertext addition failed");
    //
    //         assert_eq!(pt_mul, &mu1 * &mu2, "multiplication by scalar failed");
    //         assert_eq!(pt_mul_ct, &mu1 * &mu2, "ciphertext multiplication failed");
    //     }
    // }
}
