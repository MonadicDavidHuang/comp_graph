extern crate ndarray;
extern crate ndarray_linalg;

use std::rc::Rc;
use std::cell::RefCell;

use ndarray::*;
use ndarray_linalg::*;

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

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
// trait //
trait cg_function {
    fn forward(&self) -> f64;
    // fn backward(&self) -> ();
    fn showdata(&self) -> ();
    fn showgrad(&self) -> ();
    fn row_size() -> &i16;
    fn col_size() -> &i16;
}
// /trait //

// CG_Data //
struct CG_Data {
    data: f64,
    grad: f64,
    parVOpt: Option<Rc<RefCell<CG_Variable>>>,
}

impl CG_Data {
    fn new(datanum: f64, parentOpt: Option<Rc<RefCell<CG_Variable>>>) -> Rc<RefCell<CG_Data>> {
        let obj_Data = CG_Data {data: datanum, grad: 810931 as f64, parVOpt: parentOpt};
        let ref_Data = Rc::new(RefCell::new(obj_Data));
        ref_Data
    }
}

// impl CG_Data for CG_Function {
impl CG_Function for CG_Data {
    fn forward(&self) -> f64 { // since forward method calls from its child recursivelly,
        match self.parVOpt {
            Some(ref parent) => parent.borrow_mut().forward(),
            None => self.data,
        }
    }
    fn showdata(&self) -> () {println!("{}", self.data);}
    fn showgrad(&self) -> () {println!("{}", self.grad);}
}
// /CG_Data //
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
