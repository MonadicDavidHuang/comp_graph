extern crate comp_graph;

// extern crate rand;
extern crate ndarray;
// extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate openblas_src; // or another backend of your choice

use comp_graph::graph::node_function::*;
use comp_graph::graph::node_variable::*;

// use rand::distributions::*;
use ndarray::*;
// use ndarray_rand::RandomExt;
use ndarray_linalg::*;

use std::time::{Duration, Instant};
use std::sync::{Arc, Weak, RwLock};

fn fuck() -> Arc<RwLock<CgVariable>> {
    let shape1 = (5 as usize, 2 as usize);
    let arr1 = Array2::<f32>::ones(shape1);
    let var1 = CgVariable::new_base(arr1);

    let shape2 = (5 as usize, 2 as usize);
    let arr2 = Array2::<f32>::ones(shape2);
    let var2 = CgVariable::new_base(arr2);

    let fun3 = CgPlus::new(var1, var2);
    let var3 = CgVariable::new(fun3);
    var3
}

fn main() {
    let var = fuck();

    let result = (*(*var).write().unwrap()).forward().to_owned();
    // let mut guard = (*var).write().unwrap();
    // let result = (*guard).forward().to_owned();

    println!("{:?}", result);

    /*
    let shape1 = (5 as usize, 5 as usize);
    let tmp1 = Array2::<f32>::ones(shape1);

    let data = CgData::new(shape1, tmp1);
    let data_ref = CgVariable::new(data);

    {
        // let mut result = (*data_ref).write().unwrap();
        let result = (*data_ref).read().unwrap();
        let dude: &(Array2<f32>) = (*result).see();
        let calc: Array2<f32> = 2.0 + dude;
        println!("{:?}", calc);
    }

    let tmp = Arc::new(RwLock::new(Array2::<f32>::ones(shape1)));

    let a = Some(1);
    let b = match a {Some(ref ina) => {*ina}, None => 114514,};
    println!("{:?}", b);
    */
}
