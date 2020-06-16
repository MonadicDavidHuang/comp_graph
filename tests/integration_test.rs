#[cfg(test)]
mod basic_tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use ndarray::*;
    use ndarray_linalg::*;

    use computation_graph::graph::node_function::CgFunctionWrapper;
    use computation_graph::graph::node_functions::plus::CgPlus;
    use computation_graph::graph::node_variable::CgVariableWrapper;

    fn make_plus(shape: (usize, usize)) -> (CgVariableWrapper, CgVariableWrapper) {
        let array1 = Array2::<f32>::ones(shape);
        let variable1 = CgVariableWrapper::from_array(array1);

        let array2 = Array2::<f32>::ones(shape);
        let variable2 = CgVariableWrapper::from_array(array2);

        let variable3 = CgPlus::from_wrapper_to_wrapper(variable1.clone(), variable2.clone());

        let variable4 = CgPlus::from_wrapper_to_wrapper(variable3.clone(), variable1.clone());

        let variable5 = CgPlus::from_wrapper_to_wrapper(variable3, variable4);

        (variable5, variable2)
    }

    //#[test]
    fn test_forward() {
        let shape = (5 as usize, 2 as usize);

        let tup = make_plus(shape);

        let var = tup.0;

        {
            let mut guard = (*var).borrow_mut();
            guard.forward();
        }

        let result = {
            let guard = (*var).borrow();
            let return_result = (*guard).get_ref().to_owned();
            return_result
        };

        {
            let guard = (*var).borrow();
            guard.backward(&Array2::<f32>::ones(shape));
        }

        {
            let hoge = tup.1;
            let guard = (*hoge).borrow();
            (*guard).show_grad();
        }

        let array_twos =
            Array2::<f32>::ones(shape) + Array2::<f32>::ones(shape) + Array2::<f32>::ones(shape);

        assert_eq!(array_twos, result);
    }

    #[test]
    fn test_ancestor() {
        let shape = (5 as usize, 2 as usize);

        let tup = make_plus(shape);

        let var = tup.0;

        {
            let guard = (*var).borrow();
            for e in guard.get_variable_ancestors() {
                println!("{:?}", (*e).borrow());
            }
        }
    }
}
