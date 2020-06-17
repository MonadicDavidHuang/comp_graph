use std::cell::RefCell;
use std::rc::Rc;

use super::super::node_function::{CgFunction, CgFunctionWrapper};
use super::super::node_variable::{CgVariableWeakWrapper, CgVariableWrapper};

use ndarray::*;

pub struct CgMse {
    domain_shape: (usize, usize),
    codomain_shape: (usize, usize),
    left_parent_wrapper: CgVariableWrapper,
    right_parent_wrapper: CgVariableWrapper,
    child_variable_reference_weak_optional: Option<CgVariableWeakWrapper>,
}

impl CgMse {
    pub fn from_wrapper_to_reference(
        left_parent_wrapper: CgVariableWrapper,
        right_parent_wrapper: CgVariableWrapper,
    ) -> Rc<RefCell<Self>> {
        let shape_left = (*left_parent_wrapper).borrow().get_shape();
        let shape_right = (*right_parent_wrapper).borrow().get_shape();

        assert_eq!(shape_left, shape_right);

        let domain_shape = shape_left;
        let codomain_shape = (1 as usize, 1 as usize); // Mse is always scalar
        let child_variable_reference_weak_optional = None;

        let data = CgMse {
            domain_shape,
            codomain_shape,
            left_parent_wrapper,
            right_parent_wrapper,
            child_variable_reference_weak_optional,
        };

        let reference = Rc::new(RefCell::new(data));

        reference
    }
}

impl CgFunction for CgMse {
    fn apply(
        left_parent_wrapper: CgVariableWrapper,
        right_parent_wrapper: CgVariableWrapper,
    ) -> CgVariableWrapper {
        let reference = Self::from_wrapper_to_reference(left_parent_wrapper, right_parent_wrapper);
        let wrapper = CgFunctionWrapper(reference);
        let wrapper = CgVariableWrapper::from_function_wrapper(wrapper);

        wrapper
    }

    fn forward(&self) -> Array2<f32> {
        {
            let mut guard_left = (*self.left_parent_wrapper).borrow_mut();
            (*guard_left).forward();
        }

        {
            let mut guard_right = (*self.right_parent_wrapper).borrow_mut();
            (*guard_right).forward();
        }

        let codomain_data = {
            let guard_left = (*self.left_parent_wrapper).borrow();
            let guard_right = (*self.right_parent_wrapper).borrow();

            let left_domain_reference = (*guard_left).get_ref();
            let right_domain_reference = (*guard_right).get_ref();

            let return_data = left_domain_reference + right_domain_reference;
            return_data
        };

        codomain_data
    }

    fn backward(&self, grad: &Array2<f32>) {}

    fn set_child(&mut self, child_variable_wrapper: CgVariableWrapper) {
        let child_variable_reference_weak = child_variable_wrapper.downgrade();
        self.child_variable_reference_weak_optional = Some(child_variable_reference_weak);
    }

    fn get_left_parent_wrapper(&self) -> CgVariableWrapper {
        self.left_parent_wrapper.clone()
    }

    fn get_right_parent_wrapper(&self) -> CgVariableWrapper {
        self.right_parent_wrapper.clone()
    }

    fn get_domain_shape(&self) -> (usize, usize) {
        self.domain_shape
    }

    fn get_codomain_shape(&self) -> (usize, usize) {
        self.codomain_shape
    }
}
