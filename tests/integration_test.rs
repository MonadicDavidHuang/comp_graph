extern crate computation_graph;
extern crate ndarray;
extern crate ndarray_linalg;

#[cfg(test)]
mod basic_tests {
    use std::sync::{Arc, RwLock, Weak};

    use ndarray::*;
    use ndarray_linalg::*;

    use computation_graph::graph::node_functions::*;
    use computation_graph::graph::node_variable::*;
    use computation_graph::graph::node_functions::cg_plus::CgPlus;

    fn make_plus(shape: (usize, usize)) -> Arc<RwLock<CgVariable>> {
        let array1 = Array2::<f32>::ones(shape);
        let variable1 = CgVariable::from_array(array1);

        let array2 = Array2::<f32>::ones(shape);
        let variable2 = CgVariable::from_array(array2);

        let function3 = CgPlus::from_ref(variable1, variable2);
        let variable3 = CgVariable::from_ref(function3, "mono".to_string());
        variable3
    }

    #[test]
    fn test_forward() {
        let shape = (5 as usize, 2 as usize);

        let var = make_plus(shape);
        let result: Array2<f32> = (*(*var).write().unwrap()).forward().to_owned();

        let array_twos = Array2::<f32>::ones(shape) + Array2::<f32>::ones(shape);

        assert_eq!(array_twos, result);
    }
}
