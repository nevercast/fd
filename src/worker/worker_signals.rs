use crate::common::ModelVersion;

pub enum ThreadSignal {
    SendInit,
    Stop,
}

pub enum WorkerSignal {
    ModelUpdate(ModelVersion, Vec<f32>),
    ConfigureBuffer(usize),
}
