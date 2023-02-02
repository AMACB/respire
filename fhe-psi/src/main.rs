use fhe_psi::{z_n::*};
fn z_n_demo() {
    let x : Z_N<17> = Z_N::new_u(4);
    let y : Z_N<17> = Z_N::new_u(7);
    println!("x+y = {:?}", &x+&y);
    println!("x*y = {:?}", &x*&y);
    println!("x-y = {:?}", &x-&y);
    println!("-x, -y = {:?}, {:?}", -&x, -&y);
}

fn main() {
    z_n_demo();
}

