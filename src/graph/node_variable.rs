use std::cell::RefCell;
use std::rc::Rc;

use super::node_function::CgFunction;

use ndarray::*;
use ndarray_linalg::*;

pub fn slice2pair(slice: &[usize]) -> (usize, usize) {
    (slice[0], slice[1])
}

pub struct CgVariable {
    role: String,
    shape: (usize, usize),
    data: Array2<f32>,
    grad: Array2<f32>,
    par_f_opt: Option<Rc<RefCell<CgFunction>>>,
    did: bool,
}

impl CgVariable {
    pub fn from_array(data: Array2<f32>) -> Rc<RefCell<CgVariable>> {
        let shape: (usize, usize) = slice2pair(data.shape());

        let mut variable_obj = CgVariable {
            role: "mono".to_string(),
            shape,
            data: Array2::<f32>::ones(shape),
            grad: Array2::<f32>::zeros(shape),
            par_f_opt: None,
            did: true,
        };

        variable_obj.set_data(data);
        let variable_ref = Rc::new(RefCell::new(variable_obj));
        variable_ref
    }

    pub fn from_ref(parent: Rc<RefCell<CgFunction>>, role: String) -> Rc<RefCell<CgVariable>> {
        let cod_shape: (usize, usize) = (*(*parent).borrow()).get_codomain_shape();

        let variable_obj = CgVariable {
            role,
            shape: cod_shape,
            data: Array2::<f32>::ones(cod_shape),
            grad: Array2::<f32>::zeros(cod_shape),
            par_f_opt: Some(parent.clone()),
            did: false,
        };

        let variable_ref = Rc::new(RefCell::new(variable_obj));
        (*(*parent).borrow_mut()).set_child(variable_ref.clone());
        variable_ref
    }

    pub fn get_ref(&self) -> &(Array2<f32>) {
        &(self.data)
    }

    pub fn forward(&mut self) {
        match self.par_f_opt {
            Some(ref par_f) => {
                if !self.did {
                    self.data = (*(**par_f).borrow()).forward();
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
                    let guard = (**par_f).borrow();
                    (*guard).backward(grad);
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
