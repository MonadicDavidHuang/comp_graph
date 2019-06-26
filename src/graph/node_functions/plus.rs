extern crate ndarray;
extern crate ndarray_linalg;

use std::sync::{Arc, RwLock, Weak};

use graph::node_function::CgFunction;
use graph::node_variable::CgVariable;

use self::ndarray::*;

pub struct CgPlus {
    domain_shape: (usize, usize),
    codomain_shape: (usize, usize),
    left_parent_reference: Arc<RwLock<CgVariable>>,
    right_parent_reference: Arc<RwLock<CgVariable>>,
    child_variable_reference_weak_optional: Option<Weak<RwLock<CgVariable>>>,
}

impl CgPlus {
    pub fn from_ref(left_parent_reference: Arc<RwLock<CgVariable>>,
                    right_parent_reference: Arc<RwLock<CgVariable>>)
                    -> Arc<RwLock<CgPlus>> {
        let shape_left = (*(*left_parent_reference).read().unwrap()).get_shape();
        let shape_right = (*(*right_parent_reference).read().unwrap()).get_shape();

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

        let reference = Arc::new(RwLock::new(data));
        
        reference
    }
}

impl CgFunction for CgPlus {
    fn forward(&self) -> Array2<f32> {
        {
            let mut guard_left = (*self.left_parent_reference).write().unwrap();
            (*guard_left).forward();
        }

        {
            let mut guard_right = (*self.right_parent_reference).write().unwrap();
            (*guard_right).forward();
        }

        let codomain_data = {
            let mut guard_left = (*self.left_parent_reference).read().unwrap();
            let mut guard_right = (*self.right_parent_reference).read().unwrap();

            let left_domain_reference = (*guard_left).get_ref();
            let right_domain_reference = (*guard_right).get_ref();

            let return_data = left_domain_reference + right_domain_reference;
            return_data
        };

        codomain_data
    }

    fn backward(&self, grad: Array2<f32>) {
        let left_grad = grad.clone();
        let right_grad = grad.clone();

        {
            let mut guard_left = (*self.left_parent_reference).write().unwrap();
            let left_grad = grad.clone();
            (*guard_left).accumulate_grad(&left_grad);
        }

        {
            let mut guard_right = (*self.right_parent_reference).write().unwrap();
            let right_grad = grad.clone();
            (*guard_right).accumulate_grad(&right_grad);
        }

        {
            let guard_left = (*self.left_parent_reference).read().unwrap();
            (*guard_left).backward(left_grad);
        }

        {
            let guard_right = (*self.right_parent_reference).read().unwrap();
            (*guard_right).backward(right_grad);
        }
    }

    fn set_child(&mut self, child_variable_reference: Arc<RwLock<CgVariable>>) {
        let child_variable_reference_weak =
            Arc::downgrade(&child_variable_reference);
        self.child_variable_reference_weak_optional = Some(child_variable_reference_weak);
    }

    fn get_domain_shape(&self) -> (usize, usize) {
        self.domain_shape
    }

    fn get_codomain_shape(&self) -> (usize, usize) {
        self.codomain_shape
    }
}
