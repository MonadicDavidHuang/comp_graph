// extern crate comp_graph;

extern crate ndarray;
extern crate ndarray_linalg;

use std::rc::Rc;
use std::cell::RefCell;

use ndarray::*;
use ndarray_linalg::*;

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
struct CG_Variable {
    did: bool,
    data: f64,
    grad: f64,
    parF: Rc<RefCell<CG_Function>>,
}

impl CG_Variable {
    fn new(parent: Rc<RefCell<CG_Function>>) -> Rc<RefCell<CG_Variable>> {
        let obj_Variable = CG_Variable {did: false, data: 114 as f64, grad: 810 as f64, parF: parent};
        let ref_Variable = Rc::new(RefCell::new(obj_Variable));
        ref_Variable
    }
    fn forward(&mut self) -> f64 {
        self.data = (*self.parF).borrow().forward();
        self.data
    }
    // fn backward(&mut self) -> () {}
    fn setgrad(&mut self, grad: f64) -> () {
        self.grad = grad; // add shape check here
    }
    fn showdata(&self) -> () {println!("{}", self.data);}
    fn showgrad(&self) -> () {println!("{}", self.grad);}
}
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
// trait //
trait CG_Function {
    fn forward(&self) -> f64;
    // fn backward(&self) -> ();
    fn showdata(&self) -> ();
    fn showgrad(&self) -> ();
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
            Some(ref parent) => (**parent).borrow_mut().forward(),
            None => self.data,
        }
    }
    fn showdata(&self) -> () {println!("{}", self.data);}
    fn showgrad(&self) -> () {println!("{}", self.grad);}
}
// /CG_Data //
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

/*
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
// inplement actual network (i.e. construct computation graph in this struct's new method)

pub struct Network {
    lastVOpt: Option<Rc<RefCell<CG_Variable>>>,
}

impl Network {
    fn new() -> Network {
        // write computation graph here
        // and return final variable (here we only restrict network output one variable)

        let d1 = CG_Data::new(1919 as f64, None);
        let v1 = CG_Variable::new(d1.clone());

        Network {lastVOpt: Some(v1)}
    }
}
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
*/

fn main() {
    /*
    let d1 = CG_Data::new(1919 as f64, None);
    let v1 = CG_Variable::new(d1.clone());

    let net = Network::new();

    let tmp = match net.lastVOpt {
        Some(ref last) => last.borrow_mut().forward(),
        None => 10 as f64,
    };

    // (*v1).borrow().showdata();
    // (*v1).borrow_mut().forward();
    // (*v1).borrow().showdata();
    */

    println!(":)");

    /*
    /**************************************************/
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs): (Array1<_>, Array2<_>) = a.clone().eigh(UPLO::Upper).unwrap();
    println!("eigenvalues = \n{:?}", e);
    println!("V = \n{:?}", vecs);
    let av = a.dot(&vecs);
    println!("AV = \n{:?}", av);
    /**************************************************/
    */

    // let a = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    // let b = arr2(&[[114.0, 514.0], [893.0, 931.0]]);
    // let c = b.dot(&a) * 10_f64;
    let a: Array2::<f32> = Array::from_elem((2, 2), 1.);
    let b = Array::from_elem((2, 2), 1.);

    let c = b.dot(&a);
    let d = c.fold(0 as f32, |b, a| {b + *a});

    println!("{:?}", a.map(|x| {*x + 1.0}));
    println!("{:?}", c);
    println!("{:?}", d);
    println!("{:?}", c.dim().0);

    let shape = (2 as usize, 3 as usize);
    let tmp = Array2::<f32>::zeros(shape);

    let from = tmp.shape();
    let shape2 = (from[0], from[1]);
    let tmp2 = Array2::<f32>::zeros(shape2);
    println!("{:?}", tmp2);
}
