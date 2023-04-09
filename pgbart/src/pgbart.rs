#![allow(non_snake_case)]

use rand::distributions::WeightedIndex;
use rand_distr::{Distribution, Normal, Uniform};

use crate::math::{self, Matrix};
use crate::particle::{Particle, ParticleParams};

use rand::{self, Rng};

// Settings for the Particle Gibbs Sampler
pub struct PgBartSettings {
    n_trees: usize,             // Number of trees in the ensemble
    n_particles: usize,         // Number of particles to spawn in each iteration
    alpha: f64,                 // Prior split probability
    default_kf: f64,            // Standard deviation of noise added during leaf value sampling
    batch: (f64, f64),          // How many trees to update in tuning / final phase
    intial_alpha_vec: Vec<f64>, // Prior on covariates to use as splits
}

// Struct with helpers and settings for most (all?) random things in the algorithm
pub struct Probabilities {
    normal: Normal<f64>,      // distro for sampling unit gaussian.
    uniform: Uniform<f64>,    // distro for sampling uniformly from a pre-defined range
    alpha_vec: Vec<f64>,      // prior for variable selection
    spliting_probs: Vec<f64>, // posterior for variable selection
    alpha: f64,               // prior split probability
}

// We had to use the trait here, because otherwise
// it seems impossible to have a clean callback into Python
pub trait ExternalData {
    fn X(&self) -> &Matrix<f64>;
    fn y(&self) -> &Vec<f64>;
    fn model_logp(&self, v: &Vec<f64>) -> f64;
}

// The core of the algorithm
pub struct PgBartState {
    data: Box<dyn ExternalData>,  // dataset we're training on
    params: PgBartSettings,       // hyperparams
    probabilities: Probabilities, // helpers and settings for most (all?) random things in the algorithm
    predictions: Vec<f64>,        // current bart predictions, one per data point
    particles: Vec<Particle>,     // m particles, one per tree
    variable_inclusion: Vec<u32>, // feature importance
    tune: bool,                   // tuning phase indicator
}

impl PgBartSettings {
    // I think we either need to implement this dummy `new`, or make all the fields `pub`?
    pub fn new(
        n_trees: usize,
        n_particles: usize,
        alpha: f64,
        default_kf: f64,
        batch: (f64, f64),
        intial_alpha_vec: Vec<f64>,
    ) -> Self {
        Self {
            n_trees,
            n_particles,
            alpha,
            default_kf,
            batch,
            intial_alpha_vec,
        }
    }
}

impl Probabilities {
    // Sample a boolean flag indicating if a node should be split or not
    pub fn sample_expand_flag<R: Rng + ?Sized>(&self, rng: &mut R, depth: u32) -> bool {
        let p = self.alpha.powi(depth as i32);
        rng.gen_bool(p)
    }

    // Sample a new value for a leaf node
    pub fn sample_leaf_value<R: Rng + ?Sized>(&self, rng: &mut R, mu: f64, kfactor: f64) -> f64 {
        mu + kfactor * self.normal.sample(rng)
    }

    // Sample the index of a feature to split on
    pub fn sample_split_index<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let p: f64 = rng.gen();
        for (idx, value) in self.spliting_probs.iter().enumerate() {
            if p < *value {
                return idx;
            }
        }

        unreachable!();
    }

    // Sample a boolean flag indicating if a node should be split or not
    pub fn sample_split_value<R: Rng + ?Sized>(&self, rng: &mut R, candidates: &Vec<f64>) -> f64 {
        let idx = rng.gen_range(0..candidates.len());
        candidates[idx]
    }

    // Sample a new kf
    pub fn sample_kf<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.uniform.sample(rng)
    }

    // Sample an index according to normalized weights
    pub fn select_particle<R: Rng + ?Sized>(&self, rng: &mut R, mut particles: Vec<Particle>, weights: &Vec<f64>) -> Particle {
        let dist = WeightedIndex::new(weights).unwrap();
        let idx = dist.sample(rng);
        let selected = particles.swap_remove(idx);

        selected
    }

    // Resample the particles according to the weights vector
    fn resample_particles<R: Rng + ?Sized>(&self, rng: &mut R, particles: Vec<Particle>, weights: &Vec<f64>) -> Vec<Particle> {
        let dist = WeightedIndex::new(weights).unwrap();
        let mut ret: Vec<Particle> = Vec::with_capacity(particles.len());

        if weights.len() != (particles.len() - 1) {
            panic!("Weights and particles mismatch");
        }

        // TODO: could this be optimized? Keep in mind that borrow checker
        // will not let us "move" any item out of a vector
        // using "remove" is slow
        // and using "swap_remove" will mess up the alignment between weights and particles
        // so "cloning" everything might be the best choice actually?
        ret.push(particles[0].clone());
        for _ in 1..particles.len() {
            let idx = dist.sample(rng) + 1;
            ret.push(particles[idx].clone());
        }

        ret
    }
}

impl PgBartState {
    // Initialize the Particle Gibbs sampler
    pub fn new(params: PgBartSettings, data: Box<dyn ExternalData>) -> Self {
        // Unpack
        let X = data.X();
        let y = data.y();
        let m = params.n_trees as f64;
        let mu = math::mean(y);
        let leaf_value = mu / m;

        // Standard deviation for binary / real data
        let binary = y.iter().all(|v| (*v == 0.) || (*v == 1.));
        let std = if binary {
            3. / m.powf(0.5)
        } else {
            math::stdev(y) / m.powf(0.5)
        };

        // Initialize the predictions at first iteration. Also initialize feat importance
        let predictions: Vec<f64> = vec![mu; X.n_rows];
        let variable_inclusion: Vec<u32> = vec![0; X.n_cols];

        // Initilize the trees (m trees with root nodes only)
        // We store the trees wrapped with Particle structs since it simplifies the code
        let mut particles: Vec<Particle> = Vec::with_capacity(params.n_trees);
        for _ in 0..params.n_trees {
            let p_params = ParticleParams { n_points: X.n_rows, kfactor: params.default_kf };
            let p = Particle::new(p_params, leaf_value);
            particles.push(p);
        }

        // Sampling probabilities
        let alpha_vec: Vec<f64> = params.intial_alpha_vec.clone(); // We will be updating those, hence the clone
        let spliting_probs: Vec<f64> = math::normalized_cumsum(&alpha_vec);
        let probabilities = Probabilities {
            alpha_vec,
            spliting_probs,
            alpha: params.alpha,
            normal: Normal::new(0., std).unwrap(),
            uniform: Uniform::new(0.33, 0.75), // TODO: parametrize this?
        };

        // Done
        PgBartState {
            params,
            data,
            particles,
            probabilities,
            predictions,
            variable_inclusion,
            tune: true,
        }
    }

    pub fn step(&mut self) {
        // Setup
        let rng = &mut rand::thread_rng();

        // Get the indices of the trees we'll be modifying
        let amount = self.num_to_update();
        let length = self.params.n_trees;
        let indices = rand::seq::index::sample(rng, length, amount);

        // Get the default prediction for a new particle
        let y = self.data.y();
        let mu = math::mean(y) / (self.params.n_particles as f64);

        // Modify each tree sequentially
        for particle_index in indices {
            // Fetch the tree to modify. We store the trees wrapped with Particle structs.
            let selected_p = &self.particles[particle_index];
            let local_preds = math::sub(&self.predictions, &selected_p.predict());

            // Initialize local particles
            // note that while self.particles has size n_trees
            // local_particles has size n_particles
            // and all particles in local_particles
            // essentially are modifications of a single tree
            let local_particles = self.initialize_particles(rng, selected_p, &local_preds, mu);

            // Now we run the inner loop
            // where we grow + resample the particles multiple times
            let local_particles = self.grow_particles(rng, local_particles, &local_preds);

            let log_weights = local_particles.iter().map(|p| p.log_likelihood).collect();
            let weights = self.normalize_weights(log_weights);

            // Sample a single tree (particle) to be kept for the next iteration of the PG sampler
            let selected = {
                self.probabilities
                    .select_particle(rng, local_particles, &weights)
            };

            // Update the probabilities of sampling each covariate if we're in the tuning phase
            // Otherwise update the feature importance counter
            self.update_sampling_probs(&selected);

            // Update the predictions
            self.predictions = math::add(&local_preds, &selected.predict());

            // Update the tree
            self.particles[particle_index] = selected;
        }
    }

    fn initialize_particles<R: Rng + ?Sized>(&self, rng: &mut R, p: &Particle, local_preds: &Vec<f64>, mu: f64) -> Vec<Particle> {
        // The first particle is the exact copy of the selected tree
        let p0 = p.frozen_copy();

        // Initialize the vector
        let mut local_particles = vec![p0];

        // Reset the weight of the first particle
        {
                let item = &mut local_particles[0];
                let preds = math::add(&local_preds, &item.predict());
                let log_lik = self.data.model_logp(&preds);
                item.log_likelihood = log_lik;
        }

        // The rest of the particles starts as empty trees (root node only);
        for _ in 1..self.params.n_particles {
            // Change the kf if we're in the tuning phase
            let params = {
                let mut params = p.params().clone();
                if self.tune {
                    params.kfactor = self.probabilities.sample_kf(rng);
                }
                params
            };

            // Create and add to the list
            let new_p = Particle::new(params, mu);
            local_particles.push(new_p);
        }

        // Done
        local_particles
    }

    fn grow_particles<R: Rng + ?Sized>(
        &self,
        rng : &mut R,
        mut particles: Vec<Particle>,
        local_preds: &Vec<f64>,
    ) -> Vec<Particle> {
        // We'll need the data to grow the particles
        let X = self.data.X();

        // Now we can start growing the local_particles
        loop {
            // Break if there is nothing to update anymore
            if particles.iter().all(|p| p.finished()) {
                break;
            }

            let mut log_weights = vec![0.0; particles.len() - 1];

            // We iterate over to_update, keeping the first unchanged
            for (i , p) in particles[1..].iter_mut().enumerate() {
                // Update the tree inside it
                let needs_update = p.grow(rng, X, self);

                // Update the weight if needed
                if needs_update {
                    let preds = math::add(&local_preds, &p.predict());
                    let loglik = self.data.model_logp(&preds);
                    log_weights[i] = loglik - p.log_likelihood;
                    p.log_likelihood = loglik;
                }
            }

            let weights = self.normalize_weights(log_weights);

            // Note: the weights are of size (n_particles - 1)
            // That's because resample() will keep the first particle anyway
            particles = self.probabilities.resample_particles(rng, particles, &weights);
        }

        // Done
        particles
    }

    fn update_sampling_probs(&mut self, p: &Particle) {
        // Get the indices of covariates used by this particle
        let used_variates = p.split_variables();

        // During tuning phase, we update the probabilities
        if self.tune {
            let probs = math::normalized_cumsum(&self.probabilities.alpha_vec);
            self.probabilities.spliting_probs = probs;
            for idx in used_variates {
                self.probabilities.alpha_vec[idx] += 1.;
            }

        // Otherwise we just record the counts
        } else {
            for idx in used_variates {
                self.variable_inclusion[idx] += 1;
            }
        }
    }

    fn normalize_weights(&self, log_weights: Vec<f64>) -> Vec<f64> {
        let max_log_weight = math::max(&log_weights);

        let scaled_weights: Vec<f64> = {
            log_weights
            .into_iter()
            .map(|x| (x - max_log_weight).exp() + 1e-12)
            .collect()
        };

        let w_sum: f64 = scaled_weights.iter().sum();

        scaled_weights.into_iter().map(|x| x / w_sum).collect()
    }

    // Returns the number of trees we should modify in the current phase
    fn num_to_update(&self) -> usize {
        let fraction = if self.tune {
            self.params.batch.0
        } else {
            self.params.batch.1
        };

        ((self.params.n_trees as f64) * fraction).floor() as usize
    }

    // Get predictions for a subset of data points
    pub fn predictions_subset(&self, indices: &Vec<usize>) -> Vec<f64> {
        let all_preds = &self.predictions;
        let mut ret = Vec::<f64>::new();

        for row_idx in indices {
            ret.push(all_preds[*row_idx]);
        }

        ret
    }

    // --- Getters ---
    pub fn probabilities(&self) -> &Probabilities {
        &self.probabilities
    }

    pub fn predictions(&self) -> &Vec<f64> {
        &self.predictions
    }

    // --- Setters ---
    pub fn set_tune(&mut self, tune: bool) {
        self.tune = tune;
    }
}
