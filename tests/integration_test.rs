extern crate computation_graph;

use computation_graph::graph;

#[cfg(test)]
mod tests {

    use graph::node_function::*;
    use graph::node_variable::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
