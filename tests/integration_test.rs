#[cfg(test)]
mod basic_tests {
    use std::sync::{Arc, RwLock};

    use ndarray::*;
    use ndarray_linalg::*;

    use computation_graph::graph::node_functions::plus::CgPlus;
    use computation_graph::graph::node_variable::CgVariable;

    fn make_plus(shape: (usize, usize)) -> (Arc<RwLock<CgVariable>>, Arc<RwLock<CgVariable>>) {
        let array1 = Array2::<f32>::ones(shape);
        let variable1 = CgVariable::from_array(array1);

        let array2 = Array2::<f32>::ones(shape);
        let variable2 = CgVariable::from_array(array2);

        let function3 = CgPlus::from_ref(variable1.clone(), variable2.clone());
        let variable3 = CgVariable::from_ref(function3, "mono".to_string());

        let function4 = CgPlus::from_ref(variable3, variable1.clone());
        let variable4 = CgVariable::from_ref(function4, "mono".to_string());

        (variable4, variable2)
    }

    #[test]
    fn test_forward() {
        let shape = (5 as usize, 2 as usize);

        let tup = make_plus(shape);

        let var = tup.0;

        {
            let mut guard = (*var).write().unwrap();
            guard.forward();
        }

        let result = {
            let guard = (*var).read().unwrap();
            let return_result = (*guard).get_ref().to_owned();
            return_result
        };

        {
            let guard = (*var).read().unwrap();
            guard.backward(Array2::<f32>::ones(shape));
        }

        {
            let hoge = tup.1;
            let guard = (*hoge).read().unwrap();
            (*guard).show_grad();
        }

        let array_twos =
            Array2::<f32>::ones(shape) + Array2::<f32>::ones(shape) + Array2::<f32>::ones(shape);

        assert_eq!(array_twos, result);
    }
}
