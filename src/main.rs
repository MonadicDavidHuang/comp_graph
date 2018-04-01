use std::rc::Rc;
use std::cell::RefCell;

struct CG_Variable {
    did: bool,
    data: f64,
    grad: f64,
    parF: Rc<RefCell<CG_Function>>,
}

impl CG_Variable {
    fn new(parent: Rc<RefCell<CG_Function>>) -> CG_Variable {
        CG_Variable {did: false, data: 114 as f64, grad: 810 as f64, parF: parent}
    }
    fn forward(&mut self) -> f64 {
        self.data = (*self.parF).borrow().forward();
        self.data
    }
    // fn backward(&mut self, child: &f64) -> () {}
    fn showdata(&self) -> () {println!("{}", self.data);}
    fn showgrad(&self) -> () {println!("{}", self.grad);}
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
trait CG_Function {
    fn forward(&self) -> f64;
    // fn backward(&self, child: &f64) -> ();
    fn showdata(&self) -> ();
    fn showgrad(&self) -> ();
}

struct CG_Data {
    data: f64,
    grad: f64,
    parVOpt: Option<Rc<RefCell<CG_Variable>>>,
}

impl CG_Data {
    fn new(datanum: f64, parentOpt: Option<Rc<RefCell<CG_Variable>>>) -> CG_Data {
        CG_Data {data: datanum, grad: 810931 as f64, parVOpt: parentOpt}
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
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

fn main() {
    let d1 = Rc::new(RefCell::new(CG_Data::new(1919 as f64, None)));
    let v1 = Rc::new(RefCell::new(CG_Variable::new(d1.clone())));

    (*v1).borrow().showdata();

    (*v1).borrow_mut().forward();

    (*v1).borrow().showdata();
}
