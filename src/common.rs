use serde::{Deserialize, Serialize};
use std::mem::size_of;
pub const MAX_F32_CHUNK_SIZE: usize = 62500 / size_of::<f32>();

pub type ParameterVersion = u32;
pub struct ParameterTransfer {
    pub parameter_version: ParameterVersion,
    pub transfer_offset: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Episode {
    pub parameter_version: ParameterVersion,
    pub noise_offset: usize,
    pub noise_size: usize,
    pub reward: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum MessageFromWorker {
    Init,
    EpisodeCompleted(Episode),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum MessageFromLearner {
    ParameterChunk {
        parameter_version: ParameterVersion,
        chunk: Vec<f32>,
        chunk_offset: usize,
        chunk_hash: u64,
    },
    InitialiseWorker {
        // Noise
        seed: u64,
        block_size: usize,
        block_count: usize,
        // Model
        parameter_count: usize,
    },
}
