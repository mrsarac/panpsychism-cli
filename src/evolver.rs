//! Evolver Agent module for Project Panpsychism.
//!
//! 🧬 **The Self-Improver** — Evolves and improves the system through learning.
//!
//! This module implements Agent 37: the Evolver, which uses genetic algorithms
//! and evolutionary optimization to improve prompts, responses, and system behavior.
//! Like a living organism that adapts to its environment, the Evolver Agent
//! continuously refines and optimizes through selection, mutation, and crossover.
//!
//! ## Philosophy
//!
//! In the Spinoza framework, evolution embodies CONATUS at its purest - the
//! fundamental drive toward self-preservation and improvement:
//!
//! - **CONATUS**: Self-improvement drive through evolutionary optimization
//! - **RATIO**: Logical fitness evaluation and selection
//! - **NATURA**: Natural selection of the fittest candidates
//! - **LAETITIA**: Joy through continuous improvement and excellence
//!
//! ## Architecture
//!
//! ```text
//! +------------------+     +------------------+     +------------------+
//! |  Initial Pool    | --> | Fitness Eval     | --> |   Selection      |
//! |  (Candidates)    |     | (Metric-Based)   |     | (Best Survive)   |
//! +------------------+     +------------------+     +------------------+
//!                                                          |
//!                                                          v
//!                          +------------------+     +------------------+
//!                          |   Next Gen       | <-- |    Mutation      |
//!                          |  (Improved)      |     | & Crossover      |
//!                          +------------------+     +------------------+
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::evolver::{EvolverAgent, EvolutionGoal, EvolutionMetric};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let evolver = EvolverAgent::builder()
//!         .population_size(20)
//!         .mutation_rate(0.1)
//!         .build();
//!
//!     let goal = EvolutionGoal::new(EvolutionMetric::ResponseQuality)
//!         .with_target_improvement(0.2)
//!         .with_max_generations(10);
//!
//!     let initial_content = "Explain quantum computing";
//!     let report = evolver.evolve(initial_content, &goal).await?;
//!
//!     println!("Best candidate: {}", report.best_candidate.content);
//!     println!("Improvement: {:.2}%", report.improvement * 100.0);
//!     Ok(())
//! }
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

// =============================================================================
// EVOLUTION METRIC
// =============================================================================

/// Metrics for evaluating evolutionary fitness.
///
/// Each metric represents a different dimension of quality that the
/// evolutionary process can optimize for. Like different survival
/// pressures in nature, these metrics guide selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum EvolutionMetric {
    /// Quality of the response content.
    /// Measures clarity, relevance, and usefulness.
    #[default]
    ResponseQuality,

    /// Response latency optimization.
    /// Measures how quickly the prompt can be processed.
    Latency,

    /// Token efficiency optimization.
    /// Measures information density per token.
    TokenEfficiency,

    /// User satisfaction optimization.
    /// Based on implicit or explicit feedback signals.
    UserSatisfaction,

    /// Accuracy score optimization.
    /// Measures correctness and precision of outputs.
    AccuracyScore,
}

impl std::fmt::Display for EvolutionMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResponseQuality => write!(f, "ResponseQuality"),
            Self::Latency => write!(f, "Latency"),
            Self::TokenEfficiency => write!(f, "TokenEfficiency"),
            Self::UserSatisfaction => write!(f, "UserSatisfaction"),
            Self::AccuracyScore => write!(f, "AccuracyScore"),
        }
    }
}

impl std::str::FromStr for EvolutionMetric {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "quality" | "response_quality" | "responsequality" => Ok(Self::ResponseQuality),
            "latency" | "speed" | "time" => Ok(Self::Latency),
            "tokens" | "token_efficiency" | "tokenefficiency" | "efficiency" => {
                Ok(Self::TokenEfficiency)
            }
            "satisfaction" | "user_satisfaction" | "usersatisfaction" | "user" => {
                Ok(Self::UserSatisfaction)
            }
            "accuracy" | "accuracy_score" | "accuracyscore" | "precision" => {
                Ok(Self::AccuracyScore)
            }
            _ => Err(Error::Config(format!(
                "Unknown evolution metric: '{}'. Valid metrics: quality, latency, tokens, satisfaction, accuracy",
                s
            ))),
        }
    }
}

impl EvolutionMetric {
    /// Get all evolution metrics.
    pub fn all() -> Vec<Self> {
        vec![
            Self::ResponseQuality,
            Self::Latency,
            Self::TokenEfficiency,
            Self::UserSatisfaction,
            Self::AccuracyScore,
        ]
    }

    /// Get a description of this metric.
    pub fn description(&self) -> &'static str {
        match self {
            Self::ResponseQuality => "Measures clarity, relevance, and usefulness of responses",
            Self::Latency => "Optimizes for faster processing time",
            Self::TokenEfficiency => "Maximizes information density per token",
            Self::UserSatisfaction => "Optimizes based on user feedback signals",
            Self::AccuracyScore => "Measures correctness and precision",
        }
    }

    /// Get the weight for this metric in multi-objective optimization.
    pub fn default_weight(&self) -> f64 {
        match self {
            Self::ResponseQuality => 0.35,
            Self::Latency => 0.15,
            Self::TokenEfficiency => 0.15,
            Self::UserSatisfaction => 0.20,
            Self::AccuracyScore => 0.15,
        }
    }
}

// =============================================================================
// MUTATION TYPE
// =============================================================================

/// Types of mutations that can be applied to candidates.
///
/// Each mutation type represents a different way to introduce
/// variation into the population, driving evolutionary exploration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MutationType {
    /// Insert new content at a position.
    Insertion,

    /// Delete content from a position.
    Deletion,

    /// Substitute content at a position.
    Substitution,

    /// Swap two segments of content.
    Swap,

    /// Combine material from two parents.
    Crossover,
}

impl std::fmt::Display for MutationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Insertion => write!(f, "Insertion"),
            Self::Deletion => write!(f, "Deletion"),
            Self::Substitution => write!(f, "Substitution"),
            Self::Swap => write!(f, "Swap"),
            Self::Crossover => write!(f, "Crossover"),
        }
    }
}

impl Default for MutationType {
    fn default() -> Self {
        Self::Substitution
    }
}

impl MutationType {
    /// Get all mutation types.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Insertion,
            Self::Deletion,
            Self::Substitution,
            Self::Swap,
            Self::Crossover,
        ]
    }

    /// Get the probability weight for this mutation type.
    pub fn probability_weight(&self) -> f64 {
        match self {
            Self::Insertion => 0.20,
            Self::Deletion => 0.15,
            Self::Substitution => 0.35,
            Self::Swap => 0.15,
            Self::Crossover => 0.15,
        }
    }
}

// =============================================================================
// MUTATION
// =============================================================================

/// A mutation applied to a candidate.
///
/// Records the specific change made during evolution, allowing
/// tracking of what transformations led to improvements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mutation {
    /// Type of mutation applied.
    pub mutation_type: MutationType,

    /// Position in the content where mutation occurred.
    pub position: usize,

    /// Description of the change made.
    pub change: String,
}

impl Mutation {
    /// Create a new mutation record.
    pub fn new(mutation_type: MutationType, position: usize, change: impl Into<String>) -> Self {
        Self {
            mutation_type,
            position,
            change: change.into(),
        }
    }

    /// Check if this is a structural mutation (affects content structure).
    pub fn is_structural(&self) -> bool {
        matches!(
            self.mutation_type,
            MutationType::Insertion | MutationType::Deletion | MutationType::Swap
        )
    }

    /// Get a brief summary of this mutation.
    pub fn summary(&self) -> String {
        format!("{} at position {}", self.mutation_type, self.position)
    }
}

impl Default for Mutation {
    fn default() -> Self {
        Self {
            mutation_type: MutationType::default(),
            position: 0,
            change: String::new(),
        }
    }
}

// =============================================================================
// CANDIDATE
// =============================================================================

/// A candidate solution in the evolutionary process.
///
/// Represents an individual in the population, with its content,
/// fitness score, and lineage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    /// The content of this candidate.
    pub content: String,

    /// Fitness score for this candidate (0.0 - 1.0).
    pub fitness: f64,

    /// ID of the parent candidate (if any).
    pub parent_id: Option<String>,

    /// Unique identifier for this candidate.
    #[serde(default)]
    pub id: String,

    /// Generation this candidate was created in.
    #[serde(default)]
    pub generation: usize,

    /// Mutations applied to create this candidate from parent.
    #[serde(default)]
    pub mutations: Vec<Mutation>,
}

impl Candidate {
    /// Create a new candidate with content.
    pub fn new(content: impl Into<String>) -> Self {
        let content_str = content.into();
        let id = Self::generate_id(&content_str, 0);
        Self {
            content: content_str,
            fitness: 0.0,
            parent_id: None,
            id,
            generation: 0,
            mutations: Vec::new(),
        }
    }

    /// Generate a deterministic ID based on content and generation.
    fn generate_id(content: &str, generation: usize) -> String {
        let hash: u64 = content.bytes().fold(generation as u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        format!("cand-{:016x}", hash)
    }

    /// Set the fitness score.
    pub fn with_fitness(mut self, fitness: f64) -> Self {
        self.fitness = fitness.clamp(0.0, 1.0);
        self
    }

    /// Set the parent ID.
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    /// Set the generation number.
    pub fn with_generation(mut self, generation: usize) -> Self {
        self.generation = generation;
        self.id = Self::generate_id(&self.content, generation);
        self
    }

    /// Add a mutation record.
    pub fn with_mutation(mut self, mutation: Mutation) -> Self {
        self.mutations.push(mutation);
        self
    }

    /// Check if this is a high-fitness candidate (>= 0.7).
    pub fn is_elite(&self) -> bool {
        self.fitness >= 0.7
    }

    /// Check if this is a low-fitness candidate (< 0.3).
    pub fn is_weak(&self) -> bool {
        self.fitness < 0.3
    }

    /// Get the number of mutations in this candidate's history.
    pub fn mutation_count(&self) -> usize {
        self.mutations.len()
    }

    /// Get a brief summary of this candidate.
    pub fn summary(&self) -> String {
        let content_preview = if self.content.len() > 30 {
            format!("{}...", &self.content[..27])
        } else {
            self.content.clone()
        };
        format!(
            "Candidate {} (gen {}, fitness: {:.3}): {}",
            &self.id[..12],
            self.generation,
            self.fitness,
            content_preview
        )
    }
}

impl Default for Candidate {
    fn default() -> Self {
        Self {
            content: String::new(),
            fitness: 0.0,
            parent_id: None,
            id: "cand-0000000000000000".to_string(),
            generation: 0,
            mutations: Vec::new(),
        }
    }
}

// =============================================================================
// GENERATION
// =============================================================================

/// A generation in the evolutionary process.
///
/// Contains all candidates in a generation along with statistics
/// about the evolutionary progress.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Generation {
    /// Generation number (0-indexed).
    pub number: usize,

    /// Candidates in this generation.
    pub candidates: Vec<Candidate>,

    /// Best fitness score in this generation.
    pub best_fitness: f64,

    /// Mutations applied in this generation.
    pub mutations: Vec<Mutation>,

    /// Average fitness of the generation.
    #[serde(default)]
    pub average_fitness: f64,

    /// Number of elite candidates preserved.
    #[serde(default)]
    pub elite_count: usize,
}

impl Generation {
    /// Create a new generation.
    pub fn new(number: usize) -> Self {
        Self {
            number,
            candidates: Vec::new(),
            best_fitness: 0.0,
            mutations: Vec::new(),
            average_fitness: 0.0,
            elite_count: 0,
        }
    }

    /// Add a candidate to this generation.
    pub fn add_candidate(&mut self, candidate: Candidate) {
        if candidate.fitness > self.best_fitness {
            self.best_fitness = candidate.fitness;
        }
        if candidate.is_elite() {
            self.elite_count += 1;
        }
        self.candidates.push(candidate);
        self.recalculate_average();
    }

    fn recalculate_average(&mut self) {
        if self.candidates.is_empty() {
            self.average_fitness = 0.0;
        } else {
            self.average_fitness = self.candidates.iter().map(|c| c.fitness).sum::<f64>()
                / self.candidates.len() as f64;
        }
    }

    /// Get the best candidate in this generation.
    pub fn best_candidate(&self) -> Option<&Candidate> {
        self.candidates.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get elite candidates (fitness >= 0.7).
    pub fn elite_candidates(&self) -> Vec<&Candidate> {
        self.candidates.iter().filter(|c| c.is_elite()).collect()
    }

    /// Get the fitness improvement from the previous generation's best.
    pub fn improvement_from(&self, previous_best: f64) -> f64 {
        if previous_best <= 0.0 {
            self.best_fitness
        } else {
            (self.best_fitness - previous_best) / previous_best
        }
    }

    /// Check if evolution has converged (all candidates similar fitness).
    pub fn has_converged(&self, threshold: f64) -> bool {
        if self.candidates.len() < 2 {
            return true;
        }
        let min = self.candidates.iter().map(|c| c.fitness).fold(f64::INFINITY, f64::min);
        let max = self.candidates.iter().map(|c| c.fitness).fold(f64::NEG_INFINITY, f64::max);
        (max - min) < threshold
    }

    /// Get a summary of this generation.
    pub fn summary(&self) -> String {
        format!(
            "Generation {}: {} candidates, best: {:.3}, avg: {:.3}, elite: {}",
            self.number,
            self.candidates.len(),
            self.best_fitness,
            self.average_fitness,
            self.elite_count
        )
    }
}

// =============================================================================
// EVOLUTION GOAL
// =============================================================================

/// Goal configuration for the evolutionary process.
///
/// Defines what metric to optimize, target improvement, and
/// constraints on the evolutionary process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionGoal {
    /// Primary metric to optimize.
    pub metric: EvolutionMetric,

    /// Target improvement ratio (e.g., 0.2 = 20% improvement).
    pub target_improvement: f64,

    /// Maximum number of generations to run.
    pub max_generations: usize,

    /// Optional secondary metrics with weights.
    #[serde(default)]
    pub secondary_metrics: HashMap<EvolutionMetric, f64>,

    /// Minimum acceptable fitness to continue evolution.
    #[serde(default = "default_min_fitness")]
    pub min_fitness: f64,

    /// Convergence threshold to stop early.
    #[serde(default = "default_convergence_threshold")]
    pub convergence_threshold: f64,
}

fn default_min_fitness() -> f64 {
    0.1
}

fn default_convergence_threshold() -> f64 {
    0.01
}

impl EvolutionGoal {
    /// Create a new evolution goal with a primary metric.
    pub fn new(metric: EvolutionMetric) -> Self {
        Self {
            metric,
            target_improvement: 0.1,
            max_generations: 10,
            secondary_metrics: HashMap::new(),
            min_fitness: default_min_fitness(),
            convergence_threshold: default_convergence_threshold(),
        }
    }

    /// Set the target improvement ratio.
    pub fn with_target_improvement(mut self, target: f64) -> Self {
        self.target_improvement = target.clamp(0.0, 10.0);
        self
    }

    /// Set the maximum number of generations.
    pub fn with_max_generations(mut self, max: usize) -> Self {
        self.max_generations = max.max(1);
        self
    }

    /// Add a secondary metric with weight.
    pub fn with_secondary_metric(mut self, metric: EvolutionMetric, weight: f64) -> Self {
        self.secondary_metrics.insert(metric, weight.clamp(0.0, 1.0));
        self
    }

    /// Set the minimum fitness threshold.
    pub fn with_min_fitness(mut self, min: f64) -> Self {
        self.min_fitness = min.clamp(0.0, 1.0);
        self
    }

    /// Set the convergence threshold.
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Check if a given improvement meets the target.
    pub fn target_met(&self, improvement: f64) -> bool {
        improvement >= self.target_improvement
    }

    /// Get a summary of this goal.
    pub fn summary(&self) -> String {
        format!(
            "Optimize {} for {:.1}% improvement over {} generations",
            self.metric,
            self.target_improvement * 100.0,
            self.max_generations
        )
    }
}

impl Default for EvolutionGoal {
    fn default() -> Self {
        Self::new(EvolutionMetric::default())
    }
}

// =============================================================================
// EVOLUTION REPORT
// =============================================================================

/// Report from an evolutionary optimization run.
///
/// Contains the results of evolution including the best candidate,
/// improvement achieved, and full evolutionary history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionReport {
    /// Number of generations evolved.
    pub generations: usize,

    /// Best candidate found during evolution.
    pub best_candidate: Candidate,

    /// Overall improvement achieved (ratio).
    pub improvement: f64,

    /// History of all generations.
    pub evolution_history: Vec<Generation>,

    /// Time taken in milliseconds.
    #[serde(default)]
    pub duration_ms: u64,

    /// Whether the target was achieved.
    #[serde(default)]
    pub target_achieved: bool,

    /// Whether evolution converged early.
    #[serde(default)]
    pub converged: bool,

    /// Initial fitness before evolution.
    #[serde(default)]
    pub initial_fitness: f64,

    /// Final fitness after evolution.
    #[serde(default)]
    pub final_fitness: f64,
}

impl EvolutionReport {
    /// Create a new evolution report.
    pub fn new(best_candidate: Candidate, generations: usize, improvement: f64) -> Self {
        Self {
            generations,
            best_candidate,
            improvement,
            evolution_history: Vec::new(),
            duration_ms: 0,
            target_achieved: false,
            converged: false,
            initial_fitness: 0.0,
            final_fitness: 0.0,
        }
    }

    /// Set the evolution history.
    pub fn with_history(mut self, history: Vec<Generation>) -> Self {
        self.evolution_history = history;
        self
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = duration_ms;
        self
    }

    /// Set whether target was achieved.
    pub fn with_target_achieved(mut self, achieved: bool) -> Self {
        self.target_achieved = achieved;
        self
    }

    /// Set whether evolution converged.
    pub fn with_converged(mut self, converged: bool) -> Self {
        self.converged = converged;
        self
    }

    /// Set initial and final fitness.
    pub fn with_fitness_range(mut self, initial: f64, final_fitness: f64) -> Self {
        self.initial_fitness = initial;
        self.final_fitness = final_fitness;
        self
    }

    /// Get the fitness progression over generations.
    pub fn fitness_progression(&self) -> Vec<f64> {
        self.evolution_history.iter().map(|g| g.best_fitness).collect()
    }

    /// Get total mutations applied across all generations.
    pub fn total_mutations(&self) -> usize {
        self.evolution_history.iter().map(|g| g.mutations.len()).sum()
    }

    /// Check if evolution was successful (improvement > 0).
    pub fn is_successful(&self) -> bool {
        self.improvement > 0.0
    }

    /// Get a summary of this report.
    pub fn summary(&self) -> String {
        format!(
            "Evolution: {} generations, {:.1}% improvement, fitness {:.3} -> {:.3}{}{}",
            self.generations,
            self.improvement * 100.0,
            self.initial_fitness,
            self.final_fitness,
            if self.target_achieved { " [TARGET MET]" } else { "" },
            if self.converged { " [CONVERGED]" } else { "" }
        )
    }
}

impl Default for EvolutionReport {
    fn default() -> Self {
        Self {
            generations: 0,
            best_candidate: Candidate::default(),
            improvement: 0.0,
            evolution_history: Vec::new(),
            duration_ms: 0,
            target_achieved: false,
            converged: false,
            initial_fitness: 0.0,
            final_fitness: 0.0,
        }
    }
}

// =============================================================================
// EVOLVER CONFIGURATION
// =============================================================================

/// Configuration for the Evolver Agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolverConfig {
    /// Size of the population in each generation.
    #[serde(default = "default_population_size")]
    pub population_size: usize,

    /// Probability of mutation occurring (0.0 - 1.0).
    #[serde(default = "default_mutation_rate")]
    pub mutation_rate: f64,

    /// Probability of crossover occurring (0.0 - 1.0).
    #[serde(default = "default_crossover_rate")]
    pub crossover_rate: f64,

    /// Number of elite candidates to preserve each generation.
    #[serde(default = "default_elite_count")]
    pub elite_count: usize,

    /// Tournament size for selection.
    #[serde(default = "default_tournament_size")]
    pub tournament_size: usize,

    /// Seed for deterministic evolution (for testing).
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_population_size() -> usize {
    20
}

fn default_mutation_rate() -> f64 {
    0.1
}

fn default_crossover_rate() -> f64 {
    0.7
}

fn default_elite_count() -> usize {
    2
}

fn default_tournament_size() -> usize {
    3
}

impl Default for EvolverConfig {
    fn default() -> Self {
        Self {
            population_size: default_population_size(),
            mutation_rate: default_mutation_rate(),
            crossover_rate: default_crossover_rate(),
            elite_count: default_elite_count(),
            tournament_size: default_tournament_size(),
            seed: None,
        }
    }
}

impl EvolverConfig {
    /// Create a fast configuration with smaller population.
    pub fn fast() -> Self {
        Self {
            population_size: 10,
            mutation_rate: 0.15,
            crossover_rate: 0.6,
            elite_count: 1,
            tournament_size: 2,
            seed: None,
        }
    }

    /// Create a thorough configuration with larger population.
    pub fn thorough() -> Self {
        Self {
            population_size: 50,
            mutation_rate: 0.08,
            crossover_rate: 0.8,
            elite_count: 5,
            tournament_size: 5,
            seed: None,
        }
    }

    /// Create a configuration optimized for exploration.
    pub fn exploratory() -> Self {
        Self {
            population_size: 30,
            mutation_rate: 0.25,
            crossover_rate: 0.5,
            elite_count: 1,
            tournament_size: 2,
            seed: None,
        }
    }

    /// Set a seed for deterministic behavior.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

// =============================================================================
// FITNESS EVALUATOR (Internal)
// =============================================================================

#[derive(Debug, Clone)]
struct FitnessEvaluator {
    metric: EvolutionMetric,
    secondary: HashMap<EvolutionMetric, f64>,
    seed: u64,
}

impl FitnessEvaluator {
    fn new(metric: EvolutionMetric, seed: u64) -> Self {
        Self {
            metric,
            secondary: HashMap::new(),
            seed,
        }
    }

    fn with_secondary(mut self, secondary: HashMap<EvolutionMetric, f64>) -> Self {
        self.secondary = secondary;
        self
    }

    fn evaluate(&self, candidate: &Candidate) -> f64 {
        let base_fitness = self.evaluate_content(&candidate.content);
        let primary_score = self.evaluate_metric(&candidate.content, self.metric);
        let secondary_score: f64 = self
            .secondary
            .iter()
            .map(|(m, w)| self.evaluate_metric(&candidate.content, *m) * w)
            .sum();
        let total_weight: f64 = 1.0 + self.secondary.values().sum::<f64>();
        let combined = (base_fitness * 0.3 + primary_score * 0.7 + secondary_score) / total_weight;
        combined.clamp(0.0, 1.0)
    }

    fn evaluate_content(&self, content: &str) -> f64 {
        if content.is_empty() {
            return 0.0;
        }
        let length_score = (content.len() as f64 / 500.0).min(1.0);
        let word_count = content.split_whitespace().count();
        let word_score = (word_count as f64 / 50.0).min(1.0);
        let has_structure = content.contains('.') || content.contains('\n');
        let structure_score = if has_structure { 0.8 } else { 0.5 };
        let hash: u64 = content.bytes().fold(self.seed, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        let variation = (hash % 100) as f64 / 500.0;
        ((length_score + word_score + structure_score) / 3.0 + variation).clamp(0.0, 1.0)
    }

    fn evaluate_metric(&self, content: &str, metric: EvolutionMetric) -> f64 {
        let base = self.evaluate_content(content);
        match metric {
            EvolutionMetric::ResponseQuality => {
                let word_count = content.split_whitespace().count();
                let quality = if word_count > 10 && word_count < 200 {
                    0.8
                } else if word_count >= 200 {
                    0.6
                } else {
                    0.4
                };
                (base * 0.5 + quality * 0.5).clamp(0.0, 1.0)
            }
            EvolutionMetric::Latency => {
                let penalty = (content.len() as f64 / 1000.0).min(0.5);
                (1.0 - penalty).max(0.3)
            }
            EvolutionMetric::TokenEfficiency => {
                let unique_words: std::collections::HashSet<&str> =
                    content.split_whitespace().collect();
                let total_words = content.split_whitespace().count().max(1);
                let uniqueness = unique_words.len() as f64 / total_words as f64;
                (base * 0.4 + uniqueness * 0.6).clamp(0.0, 1.0)
            }
            EvolutionMetric::UserSatisfaction => {
                let avg_word_len = content
                    .split_whitespace()
                    .map(|w| w.len())
                    .sum::<usize>() as f64
                    / content.split_whitespace().count().max(1) as f64;
                let readability = if avg_word_len > 4.0 && avg_word_len < 8.0 {
                    0.8
                } else {
                    0.5
                };
                (base * 0.5 + readability * 0.5).clamp(0.0, 1.0)
            }
            EvolutionMetric::AccuracyScore => {
                let has_numbers = content.chars().any(|c| c.is_numeric());
                let has_specifics = content.contains("example") || content.contains("such as");
                let specificity = if has_numbers && has_specifics {
                    0.9
                } else if has_numbers || has_specifics {
                    0.7
                } else {
                    0.5
                };
                (base * 0.4 + specificity * 0.6).clamp(0.0, 1.0)
            }
        }
    }
}

// =============================================================================
// MUTATION ENGINE (Internal)
// =============================================================================

#[derive(Debug, Clone)]
struct MutationEngine {
    mutation_rate: f64,
    crossover_rate: f64,
    seed: u64,
    counter: u64,
}

impl MutationEngine {
    fn new(mutation_rate: f64, crossover_rate: f64, seed: u64) -> Self {
        Self {
            mutation_rate,
            crossover_rate,
            seed,
            counter: 0,
        }
    }

    fn next_value(&mut self) -> f64 {
        self.counter = self.counter.wrapping_add(1);
        let hash = self.seed.wrapping_mul(self.counter).wrapping_add(12345);
        (hash % 10000) as f64 / 10000.0
    }

    fn should_mutate(&mut self) -> bool {
        self.next_value() < self.mutation_rate
    }

    fn should_crossover(&mut self) -> bool {
        self.next_value() < self.crossover_rate
    }

    fn select_mutation_type(&mut self) -> MutationType {
        let value = self.next_value();
        let mut cumulative = 0.0;
        for mt in MutationType::all() {
            cumulative += mt.probability_weight();
            if value < cumulative {
                return mt;
            }
        }
        MutationType::Substitution
    }

    fn mutate(&mut self, content: &str) -> (String, Option<Mutation>) {
        if !self.should_mutate() || content.is_empty() {
            return (content.to_string(), None);
        }
        let mutation_type = self.select_mutation_type();
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.is_empty() {
            return (content.to_string(), None);
        }
        let position = (self.next_value() * words.len() as f64) as usize;
        let position = position.min(words.len().saturating_sub(1));

        let (new_content, change) = match mutation_type {
            MutationType::Insertion => {
                let insert_words = ["additionally", "furthermore", "also", "importantly"];
                let idx = (self.next_value() * insert_words.len() as f64) as usize;
                let insert_word = insert_words[idx.min(insert_words.len() - 1)];
                let mut new_words = words.clone();
                new_words.insert(position.min(new_words.len()), insert_word);
                (new_words.join(" "), format!("Inserted '{}'", insert_word))
            }
            MutationType::Deletion => {
                if words.len() > 3 {
                    let mut new_words: Vec<&str> = words.clone();
                    let deleted = new_words.remove(position.min(new_words.len() - 1));
                    (new_words.join(" "), format!("Deleted '{}'", deleted))
                } else {
                    (content.to_string(), "No deletion (too short)".to_string())
                }
            }
            MutationType::Substitution => {
                let replacements = ["enhanced", "improved", "optimized", "refined"];
                let idx = (self.next_value() * replacements.len() as f64) as usize;
                let replacement = replacements[idx.min(replacements.len() - 1)];
                let mut new_words = words.clone();
                let old_word = new_words[position];
                new_words[position] = replacement;
                (
                    new_words.join(" "),
                    format!("Substituted '{}' with '{}'", old_word, replacement),
                )
            }
            MutationType::Swap => {
                if words.len() > 1 {
                    let mut new_words = words.clone();
                    let other_pos = ((position + 1) % words.len()).max(0);
                    new_words.swap(position, other_pos);
                    (new_words.join(" "), format!("Swapped positions {} and {}", position, other_pos))
                } else {
                    (content.to_string(), "No swap (single word)".to_string())
                }
            }
            MutationType::Crossover => {
                if words.len() > 3 {
                    let mid = words.len() / 2;
                    let (first, second) = words.split_at(mid);
                    let new_words: Vec<&str> = second.iter().chain(first.iter()).copied().collect();
                    (new_words.join(" "), "Crossover: segments swapped".to_string())
                } else {
                    (content.to_string(), "No crossover (too short)".to_string())
                }
            }
        };
        let mutation = Mutation::new(mutation_type, position, change);
        (new_content, Some(mutation))
    }

    fn crossover(&mut self, parent1: &str, parent2: &str) -> (String, Option<Mutation>) {
        if !self.should_crossover() {
            return (parent1.to_string(), None);
        }
        let words1: Vec<&str> = parent1.split_whitespace().collect();
        let words2: Vec<&str> = parent2.split_whitespace().collect();
        if words1.is_empty() || words2.is_empty() {
            return (parent1.to_string(), None);
        }
        let crossover_point1 = (self.next_value() * words1.len() as f64) as usize;
        let crossover_point2 = (self.next_value() * words2.len() as f64) as usize;
        let mut child_words: Vec<&str> = Vec::new();
        child_words.extend(&words1[..crossover_point1.min(words1.len())]);
        child_words.extend(&words2[crossover_point2.min(words2.len())..]);
        let mutation = Mutation::new(
            MutationType::Crossover,
            crossover_point1,
            format!("Crossover at points {} and {}", crossover_point1, crossover_point2),
        );
        (child_words.join(" "), Some(mutation))
    }
}

// =============================================================================
// EVOLVER AGENT
// =============================================================================

/// The Self-Improver — Agent 37 of Project Panpsychism.
///
/// The Evolver Agent uses genetic algorithms to evolve and improve
/// prompts and responses. Like natural evolution, it selects the
/// fittest candidates and applies mutations to explore the solution space.
#[derive(Debug, Clone)]
pub struct EvolverAgent {
    config: EvolverConfig,
}

impl Default for EvolverAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl EvolverAgent {
    /// Create a new Evolver Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: EvolverConfig::default(),
        }
    }

    /// Create a new Evolver Agent with custom configuration.
    pub fn with_config(config: EvolverConfig) -> Self {
        Self { config }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> EvolverAgentBuilder {
        EvolverAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &EvolverConfig {
        &self.config
    }

    /// Evolve content toward an optimization goal.
    pub async fn evolve(&self, initial_content: &str, goal: &EvolutionGoal) -> Result<EvolutionReport> {
        let start = Instant::now();

        if initial_content.trim().is_empty() {
            return Err(Error::Validation("Cannot evolve empty content".to_string()));
        }

        let seed = self.config.seed.unwrap_or(42);
        let evaluator = FitnessEvaluator::new(goal.metric, seed)
            .with_secondary(goal.secondary_metrics.clone());
        let mut mutation_engine =
            MutationEngine::new(self.config.mutation_rate, self.config.crossover_rate, seed);

        debug!(
            "Starting evolution: population={}, max_gen={}",
            self.config.population_size, goal.max_generations
        );

        let mut population = self.initialize_population(initial_content, &evaluator);
        let initial_fitness = population
            .iter()
            .map(|c| c.fitness)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let mut history: Vec<Generation> = Vec::new();
        let mut best_overall = population
            .iter()
            .max_by(|a, b| {
                a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .unwrap_or_else(|| Candidate::new(initial_content));

        let mut converged = false;

        for gen_num in 0..goal.max_generations {
            let mut generation = Generation::new(gen_num);

            for candidate in &population {
                generation.add_candidate(candidate.clone());
            }

            if generation.has_converged(goal.convergence_threshold) {
                converged = true;
                history.push(generation);
                info!("Evolution converged at generation {}", gen_num);
                break;
            }

            if let Some(gen_best) = generation.best_candidate() {
                if gen_best.fitness > best_overall.fitness {
                    best_overall = gen_best.clone();
                }
            }

            let improvement = if initial_fitness > 0.0 {
                (best_overall.fitness - initial_fitness) / initial_fitness
            } else {
                best_overall.fitness
            };

            if goal.target_met(improvement) {
                history.push(generation);
                info!(
                    "Target improvement of {:.1}% achieved at generation {}",
                    goal.target_improvement * 100.0,
                    gen_num
                );
                break;
            }

            history.push(generation);
            population = self.create_next_generation(
                &population,
                &evaluator,
                &mut mutation_engine,
                gen_num + 1,
            );
        }

        let final_fitness = best_overall.fitness;
        let improvement = if initial_fitness > 0.0 {
            (final_fitness - initial_fitness) / initial_fitness
        } else {
            final_fitness
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        let report = EvolutionReport::new(best_overall, history.len(), improvement)
            .with_history(history)
            .with_duration(duration_ms)
            .with_target_achieved(goal.target_met(improvement))
            .with_converged(converged)
            .with_fitness_range(initial_fitness, final_fitness);

        info!(
            "Evolution complete: {} generations, {:.1}% improvement in {}ms",
            report.generations,
            improvement * 100.0,
            duration_ms
        );

        Ok(report)
    }

    fn initialize_population(
        &self,
        initial_content: &str,
        evaluator: &FitnessEvaluator,
    ) -> Vec<Candidate> {
        let mut population = Vec::with_capacity(self.config.population_size);
        let original = Candidate::new(initial_content).with_generation(0);
        let fitness = evaluator.evaluate(&original);
        population.push(original.with_fitness(fitness));

        let mut seed_counter = 1u64;
        while population.len() < self.config.population_size {
            let content = self.create_variation(initial_content, seed_counter);
            let mut candidate = Candidate::new(&content).with_generation(0);
            candidate.fitness = evaluator.evaluate(&candidate);
            population.push(candidate);
            seed_counter += 1;
        }
        population
    }

    fn create_variation(&self, content: &str, seed: u64) -> String {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.len() < 3 {
            return content.to_string();
        }
        let variation_type = seed % 4;
        match variation_type {
            0 => {
                let prefixes = ["Essentially,", "Basically,", "In essence,", "Simply put,"];
                let idx = (seed as usize / 4) % prefixes.len();
                format!("{} {}", prefixes[idx], content)
            }
            1 => {
                let suffixes = ["This is fundamental.", "This is key.", "This is important."];
                let idx = (seed as usize / 4) % suffixes.len();
                format!("{} {}", content, suffixes[idx])
            }
            2 => {
                if words.len() > 2 {
                    let mut new_words = words.clone();
                    new_words.swap(0, 1);
                    new_words.join(" ")
                } else {
                    content.to_string()
                }
            }
            _ => content.to_string(),
        }
    }

    fn create_next_generation(
        &self,
        population: &[Candidate],
        evaluator: &FitnessEvaluator,
        mutation_engine: &mut MutationEngine,
        generation: usize,
    ) -> Vec<Candidate> {
        let mut next_gen = Vec::with_capacity(self.config.population_size);
        let mut sorted_pop = population.to_vec();
        sorted_pop.sort_by(|a, b| {
            b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
        });

        for elite in sorted_pop.iter().take(self.config.elite_count) {
            let mut preserved = elite.clone();
            preserved.generation = generation;
            next_gen.push(preserved);
        }

        while next_gen.len() < self.config.population_size {
            let parent1 = self.tournament_select(population, mutation_engine);
            let parent2 = self.tournament_select(population, mutation_engine);
            let (child_content, crossover_mutation) =
                mutation_engine.crossover(&parent1.content, &parent2.content);
            let (final_content, point_mutation) = mutation_engine.mutate(&child_content);
            let mut child = Candidate::new(&final_content)
                .with_generation(generation)
                .with_parent(parent1.id.clone());

            if let Some(m) = crossover_mutation {
                child = child.with_mutation(m);
            }
            if let Some(m) = point_mutation {
                child = child.with_mutation(m);
            }
            child.fitness = evaluator.evaluate(&child);
            next_gen.push(child);
        }
        next_gen
    }

    fn tournament_select<'a>(
        &self,
        population: &'a [Candidate],
        engine: &mut MutationEngine,
    ) -> &'a Candidate {
        let mut best: Option<&Candidate> = None;
        for _ in 0..self.config.tournament_size {
            let idx = (engine.next_value() * population.len() as f64) as usize;
            let idx = idx.min(population.len().saturating_sub(1));
            let candidate = &population[idx];
            if best.is_none() || candidate.fitness > best.unwrap().fitness {
                best = Some(candidate);
            }
        }
        best.unwrap_or(&population[0])
    }

    /// Evaluate fitness of a single candidate without evolution.
    pub fn evaluate_fitness(&self, content: &str, metric: EvolutionMetric) -> f64 {
        let seed = self.config.seed.unwrap_or(42);
        let evaluator = FitnessEvaluator::new(metric, seed);
        let candidate = Candidate::new(content);
        evaluator.evaluate(&candidate)
    }

    /// Apply a single mutation to content.
    pub fn mutate_once(&self, content: &str) -> (String, Option<Mutation>) {
        let seed = self.config.seed.unwrap_or(42);
        let mut engine = MutationEngine::new(1.0, 0.0, seed);
        engine.mutate(content)
    }

    /// Perform crossover between two contents.
    pub fn crossover(&self, parent1: &str, parent2: &str) -> (String, Option<Mutation>) {
        let seed = self.config.seed.unwrap_or(42);
        let mut engine = MutationEngine::new(0.0, 1.0, seed);
        engine.crossover(parent1, parent2)
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom EvolverAgent configuration.
#[derive(Debug, Default)]
pub struct EvolverAgentBuilder {
    config: Option<EvolverConfig>,
}

impl EvolverAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: EvolverConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the population size.
    pub fn population_size(mut self, size: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.population_size = size.max(2);
        self.config = Some(config);
        self
    }

    /// Set the mutation rate.
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.mutation_rate = rate.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set the crossover rate.
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.crossover_rate = rate.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set the elite count.
    pub fn elite_count(mut self, count: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.elite_count = count;
        self.config = Some(config);
        self
    }

    /// Set the tournament size.
    pub fn tournament_size(mut self, size: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.tournament_size = size.max(1);
        self.config = Some(config);
        self
    }

    /// Set the seed for deterministic evolution.
    pub fn seed(mut self, seed: u64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.seed = Some(seed);
        self.config = Some(config);
        self
    }

    /// Build the EvolverAgent.
    pub fn build(self) -> EvolverAgent {
        EvolverAgent {
            config: self.config.unwrap_or_default(),
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // EvolutionMetric Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_evolution_metric_default() {
        let metric = EvolutionMetric::default();
        assert_eq!(metric, EvolutionMetric::ResponseQuality);
    }

    #[test]
    fn test_evolution_metric_all() {
        let metrics = EvolutionMetric::all();
        assert_eq!(metrics.len(), 5);
        assert!(metrics.contains(&EvolutionMetric::ResponseQuality));
        assert!(metrics.contains(&EvolutionMetric::Latency));
        assert!(metrics.contains(&EvolutionMetric::TokenEfficiency));
        assert!(metrics.contains(&EvolutionMetric::UserSatisfaction));
        assert!(metrics.contains(&EvolutionMetric::AccuracyScore));
    }

    #[test]
    fn test_evolution_metric_display() {
        assert_eq!(EvolutionMetric::ResponseQuality.to_string(), "ResponseQuality");
        assert_eq!(EvolutionMetric::Latency.to_string(), "Latency");
        assert_eq!(EvolutionMetric::TokenEfficiency.to_string(), "TokenEfficiency");
    }

    #[test]
    fn test_evolution_metric_from_str() {
        assert_eq!("quality".parse::<EvolutionMetric>().unwrap(), EvolutionMetric::ResponseQuality);
        assert_eq!("latency".parse::<EvolutionMetric>().unwrap(), EvolutionMetric::Latency);
        assert_eq!("tokens".parse::<EvolutionMetric>().unwrap(), EvolutionMetric::TokenEfficiency);
        assert_eq!("satisfaction".parse::<EvolutionMetric>().unwrap(), EvolutionMetric::UserSatisfaction);
        assert_eq!("accuracy".parse::<EvolutionMetric>().unwrap(), EvolutionMetric::AccuracyScore);
    }

    #[test]
    fn test_evolution_metric_from_str_invalid() {
        let result = "invalid".parse::<EvolutionMetric>();
        assert!(result.is_err());
    }

    #[test]
    fn test_evolution_metric_description() {
        let desc = EvolutionMetric::ResponseQuality.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("clarity") || desc.contains("relevance"));
    }

    #[test]
    fn test_evolution_metric_default_weight() {
        let weight = EvolutionMetric::ResponseQuality.default_weight();
        assert!(weight > 0.0 && weight <= 1.0);
        let total: f64 = EvolutionMetric::all().iter().map(|m| m.default_weight()).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // MutationType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mutation_type_default() {
        let mt = MutationType::default();
        assert_eq!(mt, MutationType::Substitution);
    }

    #[test]
    fn test_mutation_type_all() {
        let types = MutationType::all();
        assert_eq!(types.len(), 5);
        assert!(types.contains(&MutationType::Insertion));
        assert!(types.contains(&MutationType::Deletion));
        assert!(types.contains(&MutationType::Substitution));
        assert!(types.contains(&MutationType::Swap));
        assert!(types.contains(&MutationType::Crossover));
    }

    #[test]
    fn test_mutation_type_display() {
        assert_eq!(MutationType::Insertion.to_string(), "Insertion");
        assert_eq!(MutationType::Deletion.to_string(), "Deletion");
        assert_eq!(MutationType::Crossover.to_string(), "Crossover");
    }

    #[test]
    fn test_mutation_type_probability_weight() {
        let total: f64 = MutationType::all().iter().map(|m| m.probability_weight()).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Mutation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mutation_new() {
        let mutation = Mutation::new(MutationType::Insertion, 5, "added word");
        assert_eq!(mutation.mutation_type, MutationType::Insertion);
        assert_eq!(mutation.position, 5);
        assert_eq!(mutation.change, "added word");
    }

    #[test]
    fn test_mutation_is_structural() {
        assert!(Mutation::new(MutationType::Insertion, 0, "").is_structural());
        assert!(Mutation::new(MutationType::Deletion, 0, "").is_structural());
        assert!(Mutation::new(MutationType::Swap, 0, "").is_structural());
        assert!(!Mutation::new(MutationType::Substitution, 0, "").is_structural());
        assert!(!Mutation::new(MutationType::Crossover, 0, "").is_structural());
    }

    #[test]
    fn test_mutation_summary() {
        let mutation = Mutation::new(MutationType::Substitution, 3, "replaced word");
        let summary = mutation.summary();
        assert!(summary.contains("Substitution"));
        assert!(summary.contains("3"));
    }

    #[test]
    fn test_mutation_default() {
        let mutation = Mutation::default();
        assert_eq!(mutation.mutation_type, MutationType::Substitution);
        assert_eq!(mutation.position, 0);
        assert!(mutation.change.is_empty());
    }

    // -------------------------------------------------------------------------
    // Candidate Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_candidate_new() {
        let candidate = Candidate::new("test content");
        assert_eq!(candidate.content, "test content");
        assert_eq!(candidate.fitness, 0.0);
        assert!(candidate.parent_id.is_none());
        assert_eq!(candidate.generation, 0);
        assert!(candidate.id.starts_with("cand-"));
    }

    #[test]
    fn test_candidate_with_fitness() {
        let candidate = Candidate::new("test").with_fitness(0.8);
        assert_eq!(candidate.fitness, 0.8);
    }

    #[test]
    fn test_candidate_fitness_clamped() {
        let candidate = Candidate::new("test").with_fitness(1.5);
        assert_eq!(candidate.fitness, 1.0);
        let candidate2 = Candidate::new("test").with_fitness(-0.5);
        assert_eq!(candidate2.fitness, 0.0);
    }

    #[test]
    fn test_candidate_with_parent() {
        let candidate = Candidate::new("test").with_parent("parent-123");
        assert_eq!(candidate.parent_id, Some("parent-123".to_string()));
    }

    #[test]
    fn test_candidate_with_generation() {
        let candidate = Candidate::new("test").with_generation(5);
        assert_eq!(candidate.generation, 5);
    }

    #[test]
    fn test_candidate_with_mutation() {
        let mutation = Mutation::new(MutationType::Insertion, 0, "added");
        let candidate = Candidate::new("test").with_mutation(mutation);
        assert_eq!(candidate.mutations.len(), 1);
    }

    #[test]
    fn test_candidate_is_elite() {
        let elite = Candidate::new("test").with_fitness(0.8);
        let non_elite = Candidate::new("test").with_fitness(0.5);
        assert!(elite.is_elite());
        assert!(!non_elite.is_elite());
    }

    #[test]
    fn test_candidate_is_weak() {
        let weak = Candidate::new("test").with_fitness(0.2);
        let strong = Candidate::new("test").with_fitness(0.5);
        assert!(weak.is_weak());
        assert!(!strong.is_weak());
    }

    #[test]
    fn test_candidate_mutation_count() {
        let candidate = Candidate::new("test")
            .with_mutation(Mutation::default())
            .with_mutation(Mutation::default());
        assert_eq!(candidate.mutation_count(), 2);
    }

    #[test]
    fn test_candidate_summary() {
        let candidate = Candidate::new("short test").with_fitness(0.5).with_generation(2);
        let summary = candidate.summary();
        assert!(summary.contains("gen 2"));
        assert!(summary.contains("0.500"));
    }

    #[test]
    fn test_candidate_default() {
        let candidate = Candidate::default();
        assert!(candidate.content.is_empty());
        assert_eq!(candidate.fitness, 0.0);
    }

    // -------------------------------------------------------------------------
    // Generation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_generation_new() {
        let gen = Generation::new(3);
        assert_eq!(gen.number, 3);
        assert!(gen.candidates.is_empty());
        assert_eq!(gen.best_fitness, 0.0);
    }

    #[test]
    fn test_generation_add_candidate() {
        let mut gen = Generation::new(0);
        gen.add_candidate(Candidate::new("test1").with_fitness(0.5));
        gen.add_candidate(Candidate::new("test2").with_fitness(0.8));
        assert_eq!(gen.candidates.len(), 2);
        assert_eq!(gen.best_fitness, 0.8);
    }

    #[test]
    fn test_generation_average_fitness() {
        let mut gen = Generation::new(0);
        gen.add_candidate(Candidate::new("a").with_fitness(0.4));
        gen.add_candidate(Candidate::new("b").with_fitness(0.6));
        assert!((gen.average_fitness - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_generation_best_candidate() {
        let mut gen = Generation::new(0);
        gen.add_candidate(Candidate::new("a").with_fitness(0.3));
        gen.add_candidate(Candidate::new("best").with_fitness(0.9));
        gen.add_candidate(Candidate::new("c").with_fitness(0.5));
        let best = gen.best_candidate().unwrap();
        assert_eq!(best.content, "best");
    }

    #[test]
    fn test_generation_elite_candidates() {
        let mut gen = Generation::new(0);
        gen.add_candidate(Candidate::new("a").with_fitness(0.3));
        gen.add_candidate(Candidate::new("b").with_fitness(0.75));
        gen.add_candidate(Candidate::new("c").with_fitness(0.8));
        let elites = gen.elite_candidates();
        assert_eq!(elites.len(), 2);
    }

    #[test]
    fn test_generation_improvement_from() {
        let mut gen = Generation::new(1);
        gen.add_candidate(Candidate::new("test").with_fitness(0.6));
        let improvement = gen.improvement_from(0.5);
        assert!((improvement - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_generation_has_converged() {
        let mut gen = Generation::new(0);
        gen.add_candidate(Candidate::new("a").with_fitness(0.50));
        gen.add_candidate(Candidate::new("b").with_fitness(0.51));
        assert!(gen.has_converged(0.05));
        assert!(!gen.has_converged(0.005));
    }

    #[test]
    fn test_generation_summary() {
        let mut gen = Generation::new(2);
        gen.add_candidate(Candidate::new("test").with_fitness(0.7));
        let summary = gen.summary();
        assert!(summary.contains("Generation 2"));
        assert!(summary.contains("1 candidates"));
    }

    // -------------------------------------------------------------------------
    // EvolutionGoal Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_evolution_goal_new() {
        let goal = EvolutionGoal::new(EvolutionMetric::Latency);
        assert_eq!(goal.metric, EvolutionMetric::Latency);
        assert_eq!(goal.target_improvement, 0.1);
        assert_eq!(goal.max_generations, 10);
    }

    #[test]
    fn test_evolution_goal_with_target_improvement() {
        let goal = EvolutionGoal::new(EvolutionMetric::ResponseQuality)
            .with_target_improvement(0.25);
        assert_eq!(goal.target_improvement, 0.25);
    }

    #[test]
    fn test_evolution_goal_with_max_generations() {
        let goal = EvolutionGoal::new(EvolutionMetric::ResponseQuality)
            .with_max_generations(20);
        assert_eq!(goal.max_generations, 20);
    }

    #[test]
    fn test_evolution_goal_with_secondary_metric() {
        let goal = EvolutionGoal::new(EvolutionMetric::ResponseQuality)
            .with_secondary_metric(EvolutionMetric::Latency, 0.3);
        assert!(goal.secondary_metrics.contains_key(&EvolutionMetric::Latency));
        assert_eq!(goal.secondary_metrics[&EvolutionMetric::Latency], 0.3);
    }

    #[test]
    fn test_evolution_goal_target_met() {
        let goal = EvolutionGoal::new(EvolutionMetric::ResponseQuality)
            .with_target_improvement(0.2);
        assert!(goal.target_met(0.25));
        assert!(!goal.target_met(0.15));
    }

    #[test]
    fn test_evolution_goal_summary() {
        let goal = EvolutionGoal::new(EvolutionMetric::TokenEfficiency)
            .with_target_improvement(0.15)
            .with_max_generations(5);
        let summary = goal.summary();
        assert!(summary.contains("TokenEfficiency"));
        assert!(summary.contains("15.0%"));
        assert!(summary.contains("5 generations"));
    }

    #[test]
    fn test_evolution_goal_default() {
        let goal = EvolutionGoal::default();
        assert_eq!(goal.metric, EvolutionMetric::ResponseQuality);
    }

    // -------------------------------------------------------------------------
    // EvolutionReport Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_evolution_report_new() {
        let candidate = Candidate::new("best").with_fitness(0.9);
        let report = EvolutionReport::new(candidate, 5, 0.3);
        assert_eq!(report.generations, 5);
        assert_eq!(report.improvement, 0.3);
        assert_eq!(report.best_candidate.content, "best");
    }

    #[test]
    fn test_evolution_report_with_history() {
        let report = EvolutionReport::new(Candidate::default(), 2, 0.1)
            .with_history(vec![Generation::new(0), Generation::new(1)]);
        assert_eq!(report.evolution_history.len(), 2);
    }

    #[test]
    fn test_evolution_report_with_duration() {
        let report = EvolutionReport::new(Candidate::default(), 1, 0.0)
            .with_duration(150);
        assert_eq!(report.duration_ms, 150);
    }

    #[test]
    fn test_evolution_report_fitness_progression() {
        let mut gen0 = Generation::new(0);
        gen0.best_fitness = 0.3;
        let mut gen1 = Generation::new(1);
        gen1.best_fitness = 0.5;
        let report = EvolutionReport::new(Candidate::default(), 2, 0.66)
            .with_history(vec![gen0, gen1]);
        let progression = report.fitness_progression();
        assert_eq!(progression, vec![0.3, 0.5]);
    }

    #[test]
    fn test_evolution_report_is_successful() {
        let successful = EvolutionReport::new(Candidate::default(), 1, 0.1);
        let unsuccessful = EvolutionReport::new(Candidate::default(), 1, -0.1);
        assert!(successful.is_successful());
        assert!(!unsuccessful.is_successful());
    }

    #[test]
    fn test_evolution_report_summary() {
        let report = EvolutionReport::new(Candidate::default(), 3, 0.15)
            .with_fitness_range(0.5, 0.65)
            .with_target_achieved(true);
        let summary = report.summary();
        assert!(summary.contains("3 generations"));
        assert!(summary.contains("15.0%"));
        assert!(summary.contains("[TARGET MET]"));
    }

    // -------------------------------------------------------------------------
    // EvolverConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_evolver_config_default() {
        let config = EvolverConfig::default();
        assert_eq!(config.population_size, 20);
        assert!((config.mutation_rate - 0.1).abs() < 0.01);
        assert!((config.crossover_rate - 0.7).abs() < 0.01);
        assert_eq!(config.elite_count, 2);
        assert_eq!(config.tournament_size, 3);
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_evolver_config_fast() {
        let config = EvolverConfig::fast();
        assert_eq!(config.population_size, 10);
        assert_eq!(config.elite_count, 1);
    }

    #[test]
    fn test_evolver_config_thorough() {
        let config = EvolverConfig::thorough();
        assert_eq!(config.population_size, 50);
        assert_eq!(config.elite_count, 5);
    }

    #[test]
    fn test_evolver_config_exploratory() {
        let config = EvolverConfig::exploratory();
        assert!(config.mutation_rate > 0.2);
    }

    #[test]
    fn test_evolver_config_with_seed() {
        let config = EvolverConfig::default().with_seed(12345);
        assert_eq!(config.seed, Some(12345));
    }

    // -------------------------------------------------------------------------
    // EvolverAgentBuilder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_evolver_builder_default() {
        let evolver = EvolverAgentBuilder::default().build();
        assert_eq!(evolver.config().population_size, 20);
    }

    #[test]
    fn test_evolver_builder_population_size() {
        let evolver = EvolverAgent::builder().population_size(30).build();
        assert_eq!(evolver.config().population_size, 30);
    }

    #[test]
    fn test_evolver_builder_mutation_rate() {
        let evolver = EvolverAgent::builder().mutation_rate(0.2).build();
        assert!((evolver.config().mutation_rate - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_evolver_builder_crossover_rate() {
        let evolver = EvolverAgent::builder().crossover_rate(0.5).build();
        assert!((evolver.config().crossover_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_evolver_builder_elite_count() {
        let evolver = EvolverAgent::builder().elite_count(4).build();
        assert_eq!(evolver.config().elite_count, 4);
    }

    #[test]
    fn test_evolver_builder_tournament_size() {
        let evolver = EvolverAgent::builder().tournament_size(5).build();
        assert_eq!(evolver.config().tournament_size, 5);
    }

    #[test]
    fn test_evolver_builder_seed() {
        let evolver = EvolverAgent::builder().seed(42).build();
        assert_eq!(evolver.config().seed, Some(42));
    }

    #[test]
    fn test_evolver_builder_chained() {
        let evolver = EvolverAgent::builder()
            .population_size(15)
            .mutation_rate(0.15)
            .crossover_rate(0.6)
            .elite_count(3)
            .seed(999)
            .build();
        assert_eq!(evolver.config().population_size, 15);
        assert!((evolver.config().mutation_rate - 0.15).abs() < 0.01);
        assert_eq!(evolver.config().elite_count, 3);
        assert_eq!(evolver.config().seed, Some(999));
    }

    // -------------------------------------------------------------------------
    // EvolverAgent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_evolver_new() {
        let evolver = EvolverAgent::new();
        assert_eq!(evolver.config().population_size, 20);
    }

    #[test]
    fn test_evolver_with_config() {
        let config = EvolverConfig::fast();
        let evolver = EvolverAgent::with_config(config);
        assert_eq!(evolver.config().population_size, 10);
    }

    #[test]
    fn test_evolver_evaluate_fitness() {
        let evolver = EvolverAgent::builder().seed(42).build();
        let fitness = evolver.evaluate_fitness(
            "This is a well-structured sentence with good content.",
            EvolutionMetric::ResponseQuality,
        );
        assert!(fitness > 0.0 && fitness <= 1.0);
    }

    #[test]
    fn test_evolver_evaluate_fitness_empty() {
        let evolver = EvolverAgent::new();
        let fitness = evolver.evaluate_fitness("", EvolutionMetric::ResponseQuality);
        // Empty content gets low but non-zero score due to quality baseline
        // base_fitness=0.0, quality=0.4 (word_count<10), combined = 0.0*0.3 + 0.2*0.7 = 0.14
        assert!(fitness < 0.2, "Empty content should have low fitness");
        assert!(fitness >= 0.0, "Fitness should be non-negative");
    }

    #[test]
    fn test_evolver_mutate_once() {
        let evolver = EvolverAgent::builder().seed(42).build();
        let (mutated, mutation) = evolver.mutate_once("The quick brown fox jumps over the lazy dog");
        assert!(mutation.is_some());
        assert!(!mutated.is_empty());
    }

    #[test]
    fn test_evolver_crossover() {
        let evolver = EvolverAgent::builder().seed(42).build();
        let (child, mutation) = evolver.crossover(
            "The first parent sentence here",
            "The second parent sentence there",
        );
        assert!(mutation.is_some());
        assert!(!child.is_empty());
    }

    #[tokio::test]
    async fn test_evolver_evolve_basic() {
        let evolver = EvolverAgent::builder()
            .population_size(10)
            .seed(42)
            .build();
        let goal = EvolutionGoal::new(EvolutionMetric::ResponseQuality)
            .with_max_generations(3)
            .with_target_improvement(0.5);
        let report = evolver
            .evolve("Explain the concept of evolution in simple terms.", &goal)
            .await
            .unwrap();
        assert!(report.generations > 0);
        assert!(report.generations <= 3);
        assert!(!report.best_candidate.content.is_empty());
    }

    #[tokio::test]
    async fn test_evolver_evolve_empty_content() {
        let evolver = EvolverAgent::new();
        let goal = EvolutionGoal::default();
        let result = evolver.evolve("", &goal).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_evolver_evolve_deterministic() {
        let evolver = EvolverAgent::builder()
            .population_size(5)
            .seed(12345)
            .build();
        let goal = EvolutionGoal::new(EvolutionMetric::TokenEfficiency)
            .with_max_generations(2);
        let report1 = evolver.evolve("Sample text for evolution", &goal).await.unwrap();
        let report2 = evolver.evolve("Sample text for evolution", &goal).await.unwrap();
        assert_eq!(report1.best_candidate.fitness, report2.best_candidate.fitness);
    }

    #[tokio::test]
    async fn test_evolver_evolve_with_secondary_metrics() {
        let evolver = EvolverAgent::builder()
            .population_size(8)
            .seed(42)
            .build();
        let goal = EvolutionGoal::new(EvolutionMetric::ResponseQuality)
            .with_secondary_metric(EvolutionMetric::Latency, 0.3)
            .with_secondary_metric(EvolutionMetric::TokenEfficiency, 0.2)
            .with_max_generations(2);
        let report = evolver
            .evolve("Multi-objective optimization test content", &goal)
            .await
            .unwrap();
        assert!(report.generations > 0);
    }

    #[tokio::test]
    async fn test_evolver_evolve_convergence() {
        let evolver = EvolverAgent::builder()
            .population_size(5)
            .mutation_rate(0.01)
            .seed(42)
            .build();
        let goal = EvolutionGoal::new(EvolutionMetric::ResponseQuality)
            .with_max_generations(20)
            .with_convergence_threshold(0.5);
        let report = evolver
            .evolve("Test convergence detection mechanism", &goal)
            .await
            .unwrap();
        assert!(report.generations <= 20);
    }

    #[tokio::test]
    async fn test_evolver_evolve_report_history() {
        let evolver = EvolverAgent::builder()
            .population_size(5)
            .seed(42)
            .build();
        let goal = EvolutionGoal::new(EvolutionMetric::ResponseQuality)
            .with_max_generations(3);
        let report = evolver
            .evolve("Test evolution history tracking", &goal)
            .await
            .unwrap();
        assert!(!report.evolution_history.is_empty());
        for (i, gen) in report.evolution_history.iter().enumerate() {
            assert_eq!(gen.number, i);
        }
    }

    #[tokio::test]
    async fn test_evolver_evolve_fitness_improves() {
        let evolver = EvolverAgent::builder()
            .population_size(15)
            .mutation_rate(0.2)
            .seed(42)
            .build();
        let goal = EvolutionGoal::new(EvolutionMetric::ResponseQuality)
            .with_max_generations(5);
        let report = evolver
            .evolve("This is the initial prompt that will be evolved over multiple generations to test fitness improvement.", &goal)
            .await
            .unwrap();
        assert!(report.final_fitness >= report.initial_fitness * 0.9);
    }

    // -------------------------------------------------------------------------
    // FitnessEvaluator Tests (via EvolverAgent)
    // -------------------------------------------------------------------------

    #[test]
    fn test_fitness_evaluator_latency_metric() {
        let evolver = EvolverAgent::builder().seed(42).build();
        let short_fitness = evolver.evaluate_fitness("Short", EvolutionMetric::Latency);
        let long_fitness = evolver.evaluate_fitness(
            &"Long content. ".repeat(100),
            EvolutionMetric::Latency,
        );
        assert!(short_fitness > long_fitness);
    }

    #[test]
    fn test_fitness_evaluator_token_efficiency() {
        let evolver = EvolverAgent::builder().seed(42).build();
        let unique_fitness = evolver.evaluate_fitness(
            "Each word here is completely different unique special",
            EvolutionMetric::TokenEfficiency,
        );
        let repetitive_fitness = evolver.evaluate_fitness(
            "the the the the the the the the the the",
            EvolutionMetric::TokenEfficiency,
        );
        assert!(unique_fitness > repetitive_fitness);
    }

    #[test]
    fn test_fitness_evaluator_accuracy_with_numbers() {
        let evolver = EvolverAgent::builder().seed(42).build();
        let with_numbers = evolver.evaluate_fitness(
            "The example shows 42 items, such as these",
            EvolutionMetric::AccuracyScore,
        );
        let without = evolver.evaluate_fitness(
            "The items are shown here",
            EvolutionMetric::AccuracyScore,
        );
        assert!(with_numbers > without);
    }
}
