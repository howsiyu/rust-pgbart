use std::collections::HashMap;

#[derive(Clone)]
pub enum Node {
    Leaf(f64),
    Internal { split_idx: usize, split_value: f64 },
}

#[derive(Clone)]
pub struct Tree {
    nodes: HashMap<usize, Node>,
}

#[derive(Debug)]
pub enum TreeError {
    NotLeaf(usize),
    IndexNotFound(usize),
}

impl Tree {
    // Returns the index of the left child for this node
    fn left(index: usize) -> usize {
        index * 2 + 1
    }

    // Returns the index of the right child for this node
    fn right(index: usize) -> usize {
        index * 2 + 2
    }

    // Returns the depth at which this node lives
    pub fn depth(index: usize) -> u32 {
        (index + 1).ilog2()
    }

    // Creates a tree with a single root node
    pub fn new(root_value: f64) -> Self {
        let nodes = HashMap::from_iter([(0, Node::Leaf(root_value))]);
        Tree { nodes }
    }

    // If a leaf node exists at given index, returns its value
    pub fn get_leaf_value(&self, idx: usize) -> Result<f64, TreeError> {
        let node = self.nodes.get(&idx).ok_or(TreeError::IndexNotFound(idx))?;
        match node {
            Node::Leaf(v) => Ok(*v),
            _ => Err(TreeError::NotLeaf(idx))
        }
    }

    // Assigns a new node at a given index
    fn add_leaf_node(&mut self, idx: usize, value: f64) {
        self.nodes.insert(idx, Node::Leaf(value));
    }

    // Turns a leaf node into an internal node with two children
    // Returns the indices of newly created leaves
    pub fn split_leaf_node(
        &mut self,
        idx: usize,
        split_idx: usize,
        split_value: f64,
        left_value: f64,
        right_value: f64,
    ) -> Result<(usize, usize), TreeError> {
        // Check is Leaf
        let node = self.nodes.get_mut(&idx).ok_or(TreeError::IndexNotFound(idx))?;
        let Node::Leaf(_) = node else {
            return Err(TreeError::NotLeaf(idx));
        };

        // Get the children indices
        let lix = Tree::left(idx);
        let rix = Tree::right(idx);

        // Set the nodes
        *node = Node::Internal { split_idx, split_value };
        self.add_leaf_node(lix, left_value);
        self.add_leaf_node(rix, right_value);

        // Done
        Ok((lix, rix))
    }

    // Returns a prediction for a given example
    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        // We start with the root node
        let mut idx = 0usize;

        // And keep searching the tree
        loop {
            // We should panic if idx points to nothing
            let node = self.nodes.get(&idx).unwrap();

            match *node {
                // Until we find a leaf
                Node::Leaf(v) => {
                    return v;
                }

                // Otherwise we go left or right
                Node::Internal { split_idx, split_value } => {
                    // Depending on the value of the feature
                    idx = if x[split_idx] <= split_value {
                        Tree::left(idx)
                    } else {
                        Tree::right(idx)
                    };
                }
            }
        }
    }

    // Returns a list of split variables (covariates, features) used by this tree
    pub fn get_split_variables(&self) -> Vec<usize> {
        let mut ret: Vec<usize> = Vec::new();

        for item in self.nodes.values() {
            if let Node::Internal { split_idx, .. } = item {
                ret.push(*split_idx);
            }
        }

        ret
    }
}
