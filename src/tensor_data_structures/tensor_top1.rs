// use super::top1::
// use rand_distr::num_traits::Float;
//
// pub struct TensorTop1 {
//     pub top1_structures: Vec<Top1>,
//     pub alpha: f64,
//     pub beta: f64,
//     pub threshold: f64,
// }
//
// impl TensorTop1 {
//     pub fn new(data: Vec<Vec<f64>>, alpha: f64, beta: f64, theta: f64) -> Self {
//         // Number of data points
//         let n = data.len() as f64;
//         // Number of Top1 structures
//         let t = ( n.ln().powf(1. / 8.) / (1. - alpha.powf(2.))).ceil() as usize;
//         // Update theta
//         let theta = theta / (t as f64);
//         //// Store t Top1 structures
//         let mut top1_structures = Vec::new();
//         for i in 0..t {
//             println!("Creating Top1 structure {}/{}", i, t);
//             let top1 = Top1::new(data.clone(), alpha, beta, theta);
//             top1_structures.push(top1);
//         }
//         // All the thresholds are the same, so we can just use the first one
//         let threshold = top1_structures[0].threshold;
//         TensorTop1 {
//             top1_structures,
//             alpha,
//             beta,
//             threshold,
//         }
//     }
// }
