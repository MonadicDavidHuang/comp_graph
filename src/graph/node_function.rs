use std::ops::Deref;

use std::collections::HashSet;

use std::cell::RefCell;
use std::rc::Rc;

use super::node_variable::CgVariableWrapper;

use ndarray::*;

pub trait CgFunction {
    fn apply(
        left_parent_wrapper: CgVariableWrapper,
        right_parent_wrapper: CgVariableWrapper,
    ) -> CgVariableWrapper where Self: Sized;

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

        ret.extend(left_variable_ancestors);
        ret.extend(right_variable_ancestors);

        ret.insert(left_parent);
        ret.insert(right_parent);

        ret
    }

    fn get_domain_shape(&self) -> (usize, usize);
    fn get_codomain_shape(&self) -> (usize, usize);
}

#[derive(Clone)]
pub struct CgFunctionWrapper(pub Rc<RefCell<dyn CgFunction>>);

impl Deref for CgFunctionWrapper {
    type Target = Rc<RefCell<dyn CgFunction>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
