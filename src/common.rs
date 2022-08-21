use serde::{Deserialize, Serialize};
use std::mem::size_of;
pub const MAX_F32_CHUNK_SIZE: usize = 62500 / size_of::<f32>();

pub enum TransferCompletion {
    NeedsMoreData {
        transfer: ModelTransfer,
        received: usize,
        total: usize,
    },
    Complete {
        model: Vec<f32>,
        model_version: ModelVersion,
    },
}

pub type ModelVersion = u32;
pub struct ModelTransfer {
    pub model_version: ModelVersion,
    pub transfer_offset: usize,
    pub buffer: Vec<f32>,
}

impl ModelTransfer {
    pub fn new(model_version: ModelVersion, parameter_count: usize) -> ModelTransfer {
        ModelTransfer {
            model_version,
            transfer_offset: 0,
            buffer: vec![0.0; parameter_count],
        }
    }

    pub fn receive_chunk(self, chunk: &[f32]) -> TransferCompletion {
        // Destruction self
        let ModelTransfer {
            model_version,
            mut transfer_offset,
            mut buffer,
        } = self;
        let buffer_len = buffer.len();
        buffer[self.transfer_offset..self.transfer_offset + chunk.len()].copy_from_slice(chunk);
        transfer_offset += chunk.len();
        if self.transfer_offset == buffer.len() {
            TransferCompletion::Complete {
                model: buffer,
                model_version,
            }
        } else {
            TransferCompletion::NeedsMoreData {
                transfer: ModelTransfer {
                    model_version,
                    transfer_offset,
                    buffer,
                },
                received: transfer_offset,
                total: buffer_len,
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Episode {
    pub model_version: ModelVersion,
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
pub struct ParameterChunkData {
    pub chunk: Vec<f32>,
    pub chunk_offset: usize,
    pub chunk_hash: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum MessageFromLearner {
    ParameterChunk {
        model_version: ModelVersion,
        data: ParameterChunkData,
    },
    InitialiseWorker {
        parameter_count: usize,
    },
}
