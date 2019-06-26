use std::sync::Arc;
use std::sync::RwLock;

use graph::node_variable::CgVariable;

pub trait Optimizer {
    fn get_cg_variables() -> Vec<Arc<RwLock<CgVariable>>>;
    fn update();
}
