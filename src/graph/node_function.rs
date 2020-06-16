use std::ptr;

use std::ops::Deref;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use std::cell::RefCell;
use std::rc::Rc;

use super::node_variable::CgVariableWrapper;

use ndarray::*;

pub trait CgFunction {
    fn forward(&self) -> Array2<f32>;
    fn backward(&self, grad: &Array2<f32>);
    fn set_child(&mut self, chi_v: CgVariableWrapper);

    fn get_left_parent_wrapper(&self) -> CgVariableWrapper;
    fn get_right_parent_wrapper(&self) -> CgVariableWrapper;

    // Get variable "get_variable_ancestors", must add direct parents of this function.
    fn get_variable_ancestors(&self) -> HashSet<CgVariableWrapper> {
        let mut ret = HashSet::new();

        let left_parent = self.get_left_parent_wrapper();
        let right_parent = self.get_right_parent_wrapper();

        let left_variable_ancestors = (*left_parent).borrow().get_variable_ancestors();
        let right_variable_ancestors = (*left_parent).borrow().get_variable_ancestors();

        ret.union(&left_variable_ancestors);
        ret.union(&right_variable_ancestors);

        ret.insert(left_parent);
        ret.insert(right_parent);

        ret
    }

    fn get_domain_shape(&self) -> (usize, usize);
    fn get_codomain_shape(&self) -> (usize, usize);
}

#[derive(Clone)]
pub struct CgFunctionWrapper(Rc<RefCell<dyn CgFunction>>);

impl PartialEq for CgFunctionWrapper {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for CgFunctionWrapper {}

impl Hash for CgFunctionWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(self, state)
    }
}

impl Deref for CgFunctionWrapper {
    type Target = Rc<RefCell<dyn CgFunction>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl CgFunctionWrapper {
    fn from_reference(reference: Rc<RefCell<dyn CgFunction>>) -> Self {
        CgFunctionWrapper { 0: reference }
    }
}
