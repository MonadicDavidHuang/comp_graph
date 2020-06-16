use std::ptr;

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use std::ops::Deref;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use super::node_function::CgFunctionWrapper;

use ndarray::*;
use ndarray_linalg::*;

pub fn slice2pair(slice: &[usize]) -> (usize, usize) {
    (slice[0], slice[1])
}

#[derive(Debug, Hash, PartialEq)]
pub enum VariableRole {
    FromArray,
    FromFunction,
}

pub struct CgVariable {
    role: VariableRole,
    shape: (usize, usize),
    data: Array2<f32>,
    grad: Array2<f32>,
    par_f_opt: Option<CgFunctionWrapper>,
    did: bool,
}

impl std::fmt::Debug for CgVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CgVariable")
            .field("address", &self)
            .field("role", &self.role)
            .field("shape", &self.shape)
            .field("did", &self.did)
            .finish()
    }
}

impl CgVariable {
    pub fn from_array_to_reference(data: Array2<f32>) -> Rc<RefCell<Self>> {
        let shape: (usize, usize) = slice2pair(data.shape());

        let mut variable = CgVariable {
            role: VariableRole::FromArray,
            shape,
            data,
            grad: Array2::<f32>::zeros(shape),
            par_f_opt: None,
            did: true,
        };

        Rc::new(RefCell::new(variable))
    }

    pub fn from_function_wrapper_to_reference(
        function_wrapper: CgFunctionWrapper,
    ) -> Rc<RefCell<Self>> {
        let cod_shape: (usize, usize) = (*function_wrapper).borrow().get_codomain_shape();

        let variable = CgVariable {
            role: VariableRole::FromFunction,
            shape: cod_shape,
            data: Array2::<f32>::ones(cod_shape), // TODO: consider initial value
            grad: Array2::<f32>::zeros(cod_shape),
            par_f_opt: Some(function_wrapper.clone()),
            did: false,
        };

        let reference = Rc::new(RefCell::new(variable));

        let variable_wrapper = CgVariableWrapper::from_reference(reference.clone());

        (*function_wrapper).borrow_mut().set_child(variable_wrapper);

        reference
    }

    pub fn get_ref(&self) -> &Array2<f32> {
        &(self.data)
    }

    pub fn forward(&mut self) {
        match self.par_f_opt {
            Some(ref par_f) => {
                if !self.did {
                    self.data = (**par_f).borrow().forward();
                    self.did = true;
                }
            }
            None => (),
        };
    }

    pub fn backward(&self, grad: &Array2<f32>) {
        assert_eq!(self.shape, slice2pair(grad.shape()));

        match self.par_f_opt {
            Some(ref par_f) => {
                if self.did {
                    let reference = (**par_f).borrow();
                    (*reference).backward(grad);
                }
            }
            None => (),
        };
    }

    pub fn accumulate_grad(&mut self, grad: &Array2<f32>) {
        assert_eq!(self.shape, slice2pair(grad.shape()));

        self.grad += grad;
    }

    pub fn set_data(&mut self, data: Array2<f32>) {
        let shape: (usize, usize) = slice2pair(data.shape());

        assert_eq!(self.shape, shape);

        self.data = data;
    }

    // Get variable "get_variable_ancestors", which means does not include self.
    pub fn get_variable_ancestors(&self) -> HashSet<CgVariableWrapper> {
        let mut ret = HashSet::new();

        match self.role {
            VariableRole::FromArray => {
                ret // empty set
            },
            VariableRole::FromFunction => {
                match self.par_f_opt {
                    Some(ref par_f) => {
                        let variable_ancestors = par_f.borrow().get_variable_ancestors();
                        ret.union(&variable_ancestors);
                        ret
                    },
                    None => panic!("CgVariable with VariableRole::FromFunction must NOT have Optional::None parent function."),
                }
            },
        }
    }

    pub fn get_shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn reset_did(&mut self) {
        self.did = false;
    }

    pub fn show_data(&self) {
        println!("{:?}", self.data);
    }

    pub fn show_grad(&self) {
        println!("{:?}", self.grad);
    }
}

#[derive(Clone)]
pub struct CgVariableWrapper(Rc<RefCell<CgVariable>>);

impl PartialEq for CgVariableWrapper {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for CgVariableWrapper {}

impl Hash for CgVariableWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(self, state)
    }
}

impl Deref for CgVariableWrapper {
    type Target = Rc<RefCell<CgVariable>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl CgVariableWrapper {
    pub fn from_reference(reference: Rc<RefCell<CgVariable>>) -> Self {
        CgVariableWrapper { 0: reference }
    }

    pub fn from_function_wrapper(function_wrapper: CgFunctionWrapper) -> Self {
        CgVariableWrapper::from_reference(CgVariable::from_function_wrapper_to_reference(
            function_wrapper,
        ))
    }

    pub fn downgrade(&self) -> CgVariableWeakWrapper {
        CgVariableWeakWrapper {
            0: Rc::downgrade(&self.0),
        }
    }
}

#[derive(Clone)]
pub struct CgVariableWeakWrapper(Weak<RefCell<CgVariable>>);

impl Deref for CgVariableWeakWrapper {
    type Target = Weak<RefCell<CgVariable>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
