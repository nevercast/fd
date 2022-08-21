mod worker_signals;
mod worker_thread;

use crate::common::ModelVersion;
use message_io::network::Transport;
use pyo3::pyclass;
use std::io;
use worker_signals::WorkerSignal;
use worker_thread::WorkerThread;

#[pyclass]
pub struct Worker {
    thread: WorkerThread,
    buffer_size: Option<usize>,
    pub buffer: Option<Vec<f32>>,
    pub model_version: Option<ModelVersion>,
}

impl Worker {
    pub fn new(transport: Transport, addr: String) -> io::Result<Worker> {
        let thread = WorkerThread::new(transport, addr)?;
        Ok(Worker {
            thread,
            buffer_size: None,
            buffer: None,
            model_version: None,
        })
    }

    pub fn process_signals(&mut self) {
        let signal_receiver = &mut self.thread.receiver;
        while let Some(signal) = signal_receiver.try_receive() {
            match signal {
                WorkerSignal::ConfigureBuffer(size) => {
                    self.buffer_size = Some(size);
                    self.buffer = Some(vec![0.0; size]);
                }
                WorkerSignal::ModelUpdate(version, data) => {
                    if let Some(buffer) = &mut self.buffer {
                        if buffer.len() != data.len() {
                            panic!(
                                "Illegal state: ModelUpdate size {} does not match buffer size {}",
                                data.len(),
                                buffer.len()
                            );
                        }
                        buffer.copy_from_slice(&data);
                    } else {
                        panic!("Illegal state: ModelUpdate received before buffer configured");
                    }
                    self.model_version = Some(version);
                }
            }
        }
    }
}
