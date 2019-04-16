extern crate ndarray;
extern crate ndarray_linalg;

use std::sync::{Arc, RwLock, Weak};

use graph::node_function::CgFunction;
use graph::node_variable::*;

use self::ndarray::*;
use self::ndarray_linalg::*;

// ++++++++++++++++++++++++ CgPlus ++++++++++++++++++++++++ //
pub struct CgPlus {
    domain_shape: (usize, usize),
    codomain_shape: (usize, usize),
    left_parent_reference: Arc<RwLock<CgVariable>>,
    right_parent_reference: Arc<RwLock<CgVariable>>,
    child_variable_reference_weak_optional: Option<Weak<RwLock<CgVariable>>>,
}

impl CgPlus {
    pub fn new(left_parent_reference: Arc<RwLock<CgVariable>>,
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
        let mut guard_left = (*self.left_parent_reference).write().unwrap();
        let mut guard_right = (*self.right_parent_reference).write().unwrap();
        let left_domain_reference = (*guard_left).forward();
        let right_domain_reference = (*guard_right).forward();
        let codomain_data = left_domain_reference + right_domain_reference;
        codomain_data
    }

    // fn backward(&self) {}

    fn set_child(&mut self, child_variable_reference: Arc<RwLock<CgVariable>>) {
        let child_variable_reference_weak =
            Arc::downgrade(&child_variable_reference);
        self.child_variable_reference_weak_optional = Some(child_variable_reference_weak);
    }

    fn get_domain_shape(&self) -> (usize, usize) { self.domain_shape }
    fn get_codomain_shape(&self) -> (usize, usize) { self.codomain_shape }
}
// ++++++++++++++++++++++++ /CgPlus ++++++++++++++++++++++++ //

// ++++++++++++++++++++++++ CgMse ++++++++++++++++++++++++ //
pub struct CgMse {
    domain_shape: (usize, usize),
    codomain_shape: (usize, usize),
    left_parent_reference: Arc<RwLock<CgVariable>>,
    right_parent_reference: Arc<RwLock<CgVariable>>,
    child_variable_reference_weak_optional: Option<Weak<RwLock<CgVariable>>>,
}

impl CgMse {
    pub fn new(left_parent_reference: Arc<RwLock<CgVariable>>,
               right_parent_reference: Arc<RwLock<CgVariable>>)
               -> Arc<RwLock<CgMse>> {
        let shape_left = (*(*left_parent_reference).read().unwrap()).get_shape();
        let shape_right = (*(*right_parent_reference).read().unwrap()).get_shape();

        assert_eq!(shape_left, shape_right);

        let domain_shape = shape_left;
        let codomain_shape = (1 as usize, 1 as usize); // Mse is always scalar
        let child_variable_reference_weak_optional = None;

        let data = CgMse {
            domain_shape,
            codomain_shape,
            left_parent_reference,
            right_parent_reference,
            child_variable_reference_weak_optional,
        };

        let reference = Arc::new(RwLock::new(data));
        reference
    }
}

impl CgFunction for CgMse {
    fn forward(&self) -> Array2<f32> {
        let mut left_guard = (*self.left_parent_reference).write().unwrap();
        let mut right_guard = (*self.right_parent_reference).write().unwrap();
        let left_domain_ref = (*left_guard).forward();
        let right_domain_ref = (*right_guard).forward();
        let codomain_data = left_domain_ref + right_domain_ref;
        codomain_data
    }

    // fn backward(&self) {}

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
// ++++++++++++++++++++++++ /CgMse ++++++++++++++++++++++++ //
