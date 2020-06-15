use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::super::node_function::CgFunction;
use super::super::node_variable::CgVariable;

use ndarray::*;

pub struct CgPlus {
    domain_shape: (usize, usize),
    codomain_shape: (usize, usize),
    left_parent_reference: Rc<RefCell<CgVariable>>,
    right_parent_reference: Rc<RefCell<CgVariable>>,
    child_variable_reference_weak_optional: Option<Weak<RefCell<CgVariable>>>,
}

impl CgPlus {
    pub fn from_ref(
        left_parent_reference: Rc<RefCell<CgVariable>>,
        right_parent_reference: Rc<RefCell<CgVariable>>,
    ) -> Rc<RefCell<CgPlus>> {
        let shape_left = (*(*left_parent_reference).borrow()).get_shape();
        let shape_right = (*(*right_parent_reference).borrow()).get_shape();

        assert_eq!(shape_left, shape_right);

        let domain_shape = shape_left;
        let codomain_shape = shape_left;
        let child_variable_reference_weak_optional = None;

        let data = CgPlus {
            domain_shape,
            codomain_shape, // `+` returns same shape
            left_parent_reference,
            right_parent_reference,
            child_variable_reference_weak_optional,
        };

        let reference = Rc::new(RefCell::new(data));
        reference
    }
}

impl CgFunction for CgPlus {
    fn forward(&self) -> Array2<f32> {
        {
            let mut guard_left = (*self.left_parent_reference).borrow_mut();
            (*guard_left).forward();
        }

        {
            let mut guard_right = (*self.right_parent_reference).borrow_mut();
            (*guard_right).forward();
        }

        let codomain_data = {
            let guard_left = (*self.left_parent_reference).borrow();
            let guard_right = (*self.right_parent_reference).borrow();

            let left_domain_reference = (*guard_left).get_ref();
            let right_domain_reference = (*guard_right).get_ref();

            let return_data = left_domain_reference + right_domain_reference;
            return_data
        };

        codomain_data
    }

    fn backward(&self, grad: &Array2<f32>) {
        {
            let mut guard_left = (*self.left_parent_reference).borrow_mut();
            (*guard_left).accumulate_grad(grad);
        }

        {
            let mut guard_right = (*self.right_parent_reference).borrow_mut();
            (*guard_right).accumulate_grad(grad);
        }

        {
            let guard_left = (*self.left_parent_reference).borrow();
            (*guard_left).backward(grad);
        }

        {
            let guard_right = (*self.right_parent_reference).borrow();
            (*guard_right).backward(grad);
        }
    }

    fn set_child(&mut self, child_variable_reference: Rc<RefCell<CgVariable>>) {
        let child_variable_reference_weak = Rc::downgrade(&child_variable_reference);
        self.child_variable_reference_weak_optional = Some(child_variable_reference_weak);
    }

    fn get_domain_shape(&self) -> (usize, usize) {
        self.domain_shape
    }

    fn get_codomain_shape(&self) -> (usize, usize) {
        self.codomain_shape
    }
}
