extern crate ndarray;

use std::sync::{Arc, RwLock};

use graph::node_variable::CgVariable;

use self::ndarray::*;

pub trait CgFunction {
    fn forward(&self) -> Array2<f32>;
    fn backward(&self, grad: Array2<f32>);
    fn set_child(&mut self, chi_v: Arc<RwLock<CgVariable>>);
    fn get_domain_shape(&self) -> (usize, usize);
    fn get_codomain_shape(&self) -> (usize, usize);
}
