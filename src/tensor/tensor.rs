use std::fmt;

trait TensorInitialization<T> {
    fn new(arr_or_shape: T) -> Tensor
    where
        Self: Sized;
}

#[derive(Debug)]
struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl TensorInitialization<Vec<Vec<f32>>> for Tensor {
    fn new(arr2: Vec<Vec<f32>>) -> Tensor {
        let dim0 = arr2.len();
        let dim1 = arr2[0].len();
        let shape = vec![dim0, dim1];
        let mut data: Vec<f32> = vec![0.0; dim0 * dim1];
        for i in 0..dim0 {
            for j in 0..dim1 {
                data[i * dim1 + j] = arr2[i][j];
            }
        }
        Tensor { data, shape }
    }
}

impl TensorInitialization<Vec<Vec<Vec<f32>>>> for Tensor {
    fn new(arr3: Vec<Vec<Vec<f32>>>) -> Tensor {
        let dim0 = arr3.len();
        let dim1 = arr3[0].len();
        let dim2 = arr3[0][0].len();

        let shape = vec![dim0, dim1, dim2];
        let mut data: Vec<f32> = vec![0.0; dim0 * dim1 * dim2];
        for i in 0..dim0 {
            for j in 0..dim1 {
                for k in 0..dim2 {
                    data[i * dim1 + j * dim2 + k] = arr3[i][j][k];
                }
            }
        }
        Tensor { data, shape }
    }
}

impl TensorInitialization<&[usize]> for Tensor {
    fn new(shape: &[usize]) -> Tensor {
        let size: usize = shape.iter().product();
        let data = vec![0.0; size];
        let shape = shape.to_vec();
        Tensor { data, shape }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("");
        self.helper(0, 0, &mut result);
        write!(f, "{}", result)
    }
}

impl Tensor {
    fn helper(&self, index: usize, offset: usize, result: &mut String) {
        let spaces: String = Tensor::duplicate_space(index);

        if index == self.shape.len() - 1 as usize {
            result.push_str(&spaces);
            result.push_str(&String::from("[ "));
            for i in 0..self.shape[index] {
                let tmp = format!("{}, ", self.data[offset + i]);
                result.push_str(&tmp);
            }
            result.push_str(&String::from("],\n"));
        } else {
            result.push_str(&spaces);
            result.push_str(&(String::from("[\n")));

            for i in 0..self.shape[index] {
                self.helper(index + 1, offset + i * self.shape[index + 1], result);
            }

            result.push_str(&spaces);
            result.push_str(&(String::from("],\n")));
        }
    }

    fn duplicate_space(index: usize) -> String {
        let space = String::from("  ");
        let mut ret = String::from("");

        for _ in 0..index {
            ret += &space;
        }

        ret
    }
}

#[cfg(test)]
mod basic_tests {
    use super::{Tensor, TensorInitialization};

    #[test]
    fn test_hoge() {
        let shape_ref: &[usize] = &[2, 5, 5, 14];

        let tensor = Tensor::new(shape_ref);

        println!("{}", tensor);
    }
}
