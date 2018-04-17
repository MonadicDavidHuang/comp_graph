extern crate ndarray;
extern crate ndarray_linalg;

use std::rc::Rc;
use std::cell::RefCell;

use ndarray::*;
use ndarray_linalg::*;

mod node_function;
use node_function::*;

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
struct cg_variable {
    did: bool,
    data: Array2::<f32>,
    grad: Array2::<f32>,
    par_f: Rc<RefCell<cg_function>>,
}

impl cg_variable {
    fn new(parent: Rc<RefCell<cg_function>>) -> Rc<RefCell<cg_variable>> {
        let obj_variable = cg_variable {did: false,
                                        data: 114 as f64, grad: 810 as f64,
                                        par_f: parent};
        let ref_variable = Rc::new(RefCell::new(obj_variable));
        ref_variable
    }
    fn forward(&mut self) -> Array2::<f32> {
        self.data = (*self.parF).borrow().forward();
        self.data
    }
    // fn backward(&mut self) -> () {}
    fn setgrad(&mut self, grad: Array2::<f32>) -> () {
        self.grad = grad; // add shape check here
    }
    fn showdata(&self) -> () {println!("{}", self.data);}
    fn showgrad(&self) -> () {println!("{}", self.grad);}
}
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
