use std::collections::vec_deque::VecDeque;
use std::sync::Arc;
use std::sync::RwLock;

use either::Either;

use super::super::graph::node_function::CgFunction;
use super::super::graph::node_variable::CgVariable;

pub trait Optimizer {
    fn get_cg_variables() -> Vec<Arc<RwLock<CgVariable>>>;
    fn update();
}

pub struct GradientDecentOptimizer {
    variable_references: Vec<Arc<RwLock<CgVariable>>>,
}

impl GradientDecentOptimizer {
    pub fn new(terminal_variable_reference: Arc<RwLock<CgVariable>>) -> GradientDecentOptimizer {
        let variable_references: Vec<Arc<RwLock<CgVariable>>> = Vec::new();
        GradientDecentOptimizer {
            variable_references,
        }
    }

    fn bfs(terminal_variable_reference: Arc<RwLock<CgVariable>>) -> Vec<Arc<RwLock<CgVariable>>> {
        let mut variable_references = Vec::new();

        let mut deq: VecDeque<Either<Arc<RwLock<CgVariable>>, Arc<RwLock<CgFunction>>>> =
            VecDeque::new();

        while !deq.is_empty() {
            deq.push_front(Either::Left(terminal_variable_reference.clone()));
        }

        variable_references
    }
}

#[cfg(test)]
mod temp_tests {
    use std::collections::vec_deque::VecDeque;

    #[test]
    fn test_forward() {
        let mut deq: VecDeque<i32> = VecDeque::new();

        deq.push_front(10);

        while !deq.is_empty() {
            let cur = deq.pop_back().unwrap();

            if cur < 100 {
                deq.push_front(cur + 1);
            }
        }
    }
}
