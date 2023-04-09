#![allow(non_snake_case)]

use crate::math::{self, Matrix};
use crate::pgbart::PgBartState;
use crate::tree::Tree;

use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};

use rand::{self, Rng};

#[derive(Clone)]
pub struct ParticleParams {
    pub n_points: usize, // Number of points in the dataset
    pub kfactor: f64,    // Standard deviation of noise added during leaf value sampling
}

#[derive(Clone)]
struct Indices {
    leaf_nodes: HashSet<usize>,               // Set of leaf node indices
    expansion_nodes: VecDeque<usize>,         // Nodes that we still can expand
    data_indices: HashMap<usize, Vec<usize>>, // Indicies of points at each node
}

#[derive(Clone)]
pub struct Particle {
    params: ParticleParams,
    tree: Tree,
    indices: Indices,
    pub log_likelihood: f64,
}

impl Indices {
    // Creates a new struct for a given dataset size
    fn new(n_points: usize) -> Self {
        let data_indices = Vec::from_iter(0..n_points);
        Indices {
            leaf_nodes: HashSet::from([0]),
            expansion_nodes: VecDeque::from([0]),
            data_indices: HashMap::from([(0, data_indices)]),
        }
    }

    // Checks if there are any nodes left to expand
    fn is_empty(&self) -> bool {
        self.expansion_nodes.is_empty()
    }

    // Returns the indices of datapoints stored in node with index `idx`
    fn get_data_indices(&self, idx: usize) -> Result<&Vec<usize>, &str> {
        let ret = self.data_indices.get(&idx);
        ret.ok_or("Index not found in the data_indices map")
    }

    // Removes an index from the list of expansion nodes
    fn pop_expansion_index(&mut self) -> Option<usize> {
        self.expansion_nodes.pop_front()
    }

    // Removes the index from the set of leaves and from the data_indices map
    fn remove_index(&mut self, idx: usize) {
        self.leaf_nodes.remove(&idx);
        self.data_indices.remove(&idx);
    }

    // Adds an index of a new leaf to be expanded
    fn add_index(&mut self, idx: usize, data_rows: Vec<usize>) {
        self.leaf_nodes.insert(idx);
        self.expansion_nodes.push_back(idx);
        self.data_indices.insert(idx, data_rows);
    }

    // Removes everything from the expansion nodes
    fn clear(&mut self) {
        self.expansion_nodes.clear();
    }
}

impl Particle {
    // Creates a new Particle with specified Params and a single-node (root only) Tree
    pub fn new(params: ParticleParams, leaf_value: f64) -> Self {
        let n_points = params.n_points;

        Particle {
            params,
            tree: Tree::new(leaf_value),
            indices: Indices::new(n_points),
            log_likelihood: 0.0,
        }
    }

    pub fn frozen_copy(&self) -> Particle {
        let mut ret = self.clone();
        ret.indices.clear();

        ret
    }

    // Attempts to grow this particle (or, more precisely, the tree inside this particle)
    // Returns a boolean indicating if the tree structure was modified
    pub fn grow<R: Rng + ?Sized>(&mut self, rng: &mut R, X: &Matrix<f64>, state: &PgBartState) -> bool {
        // Check if there are any nodes left to expand
        let idx = match self.indices.pop_expansion_index() {
            Some(value) => value,
            None => {
                return false;
            }
        };

        // Stochastiaclly decide if the node should be split or not
        let msg = "Internal indices are not aligned with the tree";
        let expand = state.probabilities().sample_expand_flag(rng, Tree::depth(idx));
        if !expand {
            return false;
        }

        // Get the examples that were routed to this leaf
        let rows = self.indices.get_data_indices(idx).expect(msg);
        let split_idx = state.probabilities().sample_split_index(rng);
        let feature_values = X.select_rows(rows, &split_idx);

        // And see if we can split them into two groups
        if feature_values.len() == 0 {
            return false;
        }

        let split_value = state.probabilities().sample_split_value(rng, &feature_values);
        // Now we have everything (leaf_idx, split_idx, split_value)
        // So we can split the leaf into an internal node

        // Now route the data points into left / right child
        let data_inds = self.split_data(&rows, &feature_values, &split_value);

        // Ask the sampler to generate values for the new leaves to be added
        let leaf_vals = (
            self.leaf_value(rng, &data_inds.0, state),
            self.leaf_value(rng, &data_inds.1, state),
        );
        let (left_value, right_value) = leaf_vals;

        // Update the tree
        let msg = "Splitting a leaf failed, meaning the indices in particle were not consistent with the tree";
        let ret = self.tree.split_leaf_node(idx, split_idx, split_value, left_value, right_value);
        let new_inds = ret.expect(msg);

        // Remove the old index, we won't need it anymore
        self.indices.remove_index(idx);

        // Add the new leaves into expansion nodes etc.
        self.indices.add_index(new_inds.0, data_inds.0);
        self.indices.add_index(new_inds.1, data_inds.1);

        // Signal that the structure was updated
        true
    }

    // Generate predictions for this particle.
    // We do not need to traverse the tree, because during training
    // We simply keep track the leaf index where each data points lands
    pub fn predict(&self) -> Vec<f64> {
        let mut y_hat: Vec<f64> = vec![0.; self.params.n_points];

        for idx in &self.indices.leaf_nodes {
            let leaf_val = self.tree.get_leaf_value(*idx).unwrap();
            let row_inds = &self.indices.data_indices[idx];
            for i in row_inds {
                y_hat[*i] = leaf_val;
            }
        }

        y_hat
    }

    // For each pair of (row_index, feature_value) decide if that row will go to the left or right child
    fn split_data(
        &self,
        row_indices: &Vec<usize>,
        feature_values: &Vec<f64>,
        split_value: &f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices: Vec<usize> = vec![];
        let mut right_indices: Vec<usize> = vec![];

        for (idx, value) in std::iter::zip(row_indices, feature_values) {
            if value <= split_value {
                left_indices.push(*idx);
            } else {
                right_indices.push(*idx);
            }
        }

        (left_indices, right_indices)
    }

    // Returns a new sampled leaf value
    fn leaf_value<R: Rng + ?Sized>(&self, rng: &mut R, data_indices: &Vec<usize>, state: &PgBartState) -> f64 {
        // TODO: This feels a bit off
        // This function takes as input indices of data points that ended in a particular leaf
        // Then calls the Sampler to fetch the predicted values for those data points
        // Calculates the mean
        // And calls the state again to sample a value around that mean
        
        let mu = if data_indices.len() == 0 {
            0.
        } else {
            let node_preds = state.predictions_subset(data_indices);
            math::mean(&node_preds) / (self.params.n_points as f64)
        };
        
        let value = state
            .probabilities()
            .sample_leaf_value(rng, mu, self.params.kfactor);

        value
    }

    // --- Getters ---
    pub fn finished(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn split_variables(&self) -> Vec<usize> {
        self.tree.get_split_variables()
    }

    pub fn params(&self) -> &ParticleParams {
        &self.params
    }
}
