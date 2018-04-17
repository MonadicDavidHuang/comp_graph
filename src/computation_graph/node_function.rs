extern crate ndarray;
extern crate ndarray_linalg;

use std::rc::Rc;
use std::cell::RefCell;

use ndarray::*;
use ndarray_linalg::*;

mod node_variable;
use node_variable::*;

// ++++++++++++++++++++++++ trait ++++++++++++++++++++++++ //
trait cg_function {
    fn forward(&self) -> Array2::<f32>;
    // fn backward(&self) -> ();
    fn showdata(&self) -> ();
    fn showgrad(&self) -> ();
    fn row_size() -> usize;
    fn col_size() -> usize;
}
// ++++++++++++++++++++++++ /trait ++++++++++++++++++++++++ //

// ++++++++++++++++++++++++ cg_data ++++++++++++++++++++++++ //
struct cg_data {
    shape: (usize, usize),
    data: Array2::<f32>,
    grad: Array2::<f32>,
    par_v_opt: Option<Rc<RefCell<cg_variable>>>,
}

impl cg_data {
    fn new(shape: (usize, usize),
           data: Array2::<f32>,
           par_v_opt: Option<Rc<RefCell<cg_variable>>>)
           -> Rc<RefCell<cg_data>> {
        let obj_data = cg_data {data: data,
                                grad: Array2::<f32>::zeros(shape),
                                par_v_opt: par_v_opt};
        let ref_data = Rc::new(RefCell::new(obj_data));
        ref_data
    }
}

impl cg_function for cg_data {
    fn forward(&self) -> Array2::<f32> { // since forward method calls from its child recursivelly,
        match self.par_v_opt {
            Some(ref par_v) => (**par_v).borrow_mut().forward(), // ** wont be needed due to implemention of deref
            None => self.data,
        }
    }
    fn showdata(&self) -> () {println!("{}", self.data);}
    fn showgrad(&self) -> () {println!("{}", self.grad);}
}
// ++++++++++++++++++++++++ /cg_data ++++++++++++++++++++++++ //
