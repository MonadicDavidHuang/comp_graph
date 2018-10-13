extern crate ndarray;
extern crate ndarray_linalg;
extern crate openblas_src; // or another backend of your choice

use self::ndarray::*;
use self::ndarray_linalg::*;

use std::sync::{Arc, Weak, RwLock};

use graph::node_variable::*;
use graph::associator::*;

pub trait CgFunction {
    fn forward(&self) -> Array2::<f32>;
    // fn backward(&self) -> ();
    fn set_child(&mut self, chi_v: Arc<RwLock<CgVariable>>) -> ();
    fn get_dom_shape(&self) -> (usize, usize);
    fn get_cod_shape(&self) -> (usize, usize);
}

// ++++++++++++++++++++++++ CgPlus ++++++++++++++++++++++++ //
pub struct CgPlus {
    dom_shape: (usize, usize),
    cod_shape: (usize, usize),
    par_v_left: Arc<RwLock<CgVariable>>,
    par_v_right: Arc<RwLock<CgVariable>>,
    chi_v_opt: Option<Weak<RwLock<CgVariable>>>,
}

impl CgPlus {
    pub fn new(par_v_left: Arc<RwLock<CgVariable>>,
               par_v_right: Arc<RwLock<CgVariable>>) -> Arc<RwLock<CgPlus>> {
        let shape_left = (*(*par_v_left).read().unwrap()).get_shape();
        let shape_right = (*(*par_v_right).read().unwrap()).get_shape();
        assert_eq!(shape_left, shape_right);

        let obj_data = CgPlus {
            dom_shape: shape_left,
            cod_shape: shape_left, // `+` returns same shape
            par_v_left: par_v_left,
            par_v_right: par_v_right,
            chi_v_opt: None
        };

        let ref_data = Arc::new(RwLock::new(obj_data));
        ref_data
    }
}

impl CgFunction for CgPlus {
    fn forward(&self) -> Array2::<f32> {
        let mut guard_left = (*self.par_v_left).write().unwrap();
        let mut guard_right = (*self.par_v_right).write().unwrap();
        let dom_left_ref = (*guard_left).forward();
        let dom_right_ref = (*guard_right).forward();
        let cod_obj = dom_left_ref + dom_right_ref;
        cod_obj
    }

    // fn backward(&self) -> () {}

    fn set_child(&mut self, chi_v: Arc<RwLock<CgVariable>>) -> () {
        let weak_chi_v: Weak<RwLock<CgVariable>> = Arc::downgrade(&chi_v);
        self.chi_v_opt = Some(weak_chi_v);
    }

    fn get_dom_shape(&self) -> (usize, usize) {self.dom_shape}
    fn get_cod_shape(&self) -> (usize, usize) {self.cod_shape}
}
// ++++++++++++++++++++++++ /CgPlus ++++++++++++++++++++++++ //

// ++++++++++++++++++++++++ CgMSE ++++++++++++++++++++++++ //
pub struct CgMSE {
    dom_shape: (usize, usize),
    cod_shape: (usize, usize),
    par_v_left: Arc<RwLock<CgVariable>>,
    par_v_right: Arc<RwLock<CgVariable>>,
    chi_v_opt: Option<Weak<RwLock<CgVariable>>>,
}

impl CgMSE {
    pub fn new(par_v_left: Arc<RwLock<CgVariable>>,
               par_v_right: Arc<RwLock<CgVariable>>) -> Arc<RwLock<CgMSE>> {
        let shape_left = (*(*par_v_left).read().unwrap()).get_shape();
        let shape_right = (*(*par_v_right).read().unwrap()).get_shape();
        assert_eq!(shape_left, shape_right);

        let obj_data = CgMSE {
            dom_shape: shape_left,
           cod_shape: shape_left, // `+` returns same shape
           par_v_left: par_v_left,
           par_v_right: par_v_right,
           chi_v_opt: None
       };

        let ref_data = Arc::new(RwLock::new(obj_data));
        ref_data
    }
}

impl CgFunction for CgMSE {
    fn forward(&self) -> Array2::<f32> {
        let mut guard_left = (*self.par_v_left).write().unwrap();
        let mut guard_right = (*self.par_v_right).write().unwrap();
        let dom_left_ref = (*guard_left).forward();
        let dom_right_ref = (*guard_right).forward();
        let cod_obj = dom_left_ref + dom_right_ref;
        cod_obj
    }

    // fn backward(&self) -> () {}

    fn set_child(&mut self, chi_v: Arc<RwLock<CgVariable>>) -> () {
        let weak_chi_v: Weak<RwLock<CgVariable>> = Arc::downgrade(&chi_v);
        self.chi_v_opt = Some(weak_chi_v);
    }

    fn get_dom_shape(&self) -> (usize, usize) {self.dom_shape}
    fn get_cod_shape(&self) -> (usize, usize) {self.cod_shape}
}
// ++++++++++++++++++++++++ /CgDot ++++++++++++++++++++++++ //
