use std::cell::RefCell;
use std::rc::Rc;

use super::node_variable::CgVariable;

use ndarray::*;

pub trait CgFunction {
    fn forward(&self) -> Array2<f32>;
    fn backward(&self, grad: &Array2<f32>);
    fn set_child(&mut self, chi_v: Rc<RefCell<CgVariable>>);
    fn get_domain_shape(&self) -> (usize, usize);
    fn get_codomain_shape(&self) -> (usize, usize);
}
