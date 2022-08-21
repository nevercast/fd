mod collect_slice;
pub mod common;
pub mod model;
mod noise;
mod worker;

use noise::par_fill_noise_standard;
use numpy::PyArray1;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use message_io::network::Transport;

use rand::SeedableRng;
use rand_xoshiro::Xoroshiro128Plus;

use crate::worker::Worker;

#[pymethods]
impl Worker {
    fn get_parameters(&mut self, py: Python) -> PyResult<PyObject> {
        self.process_signals();
        match self.buffer {
            Some(ref mut buffer) => {
                par_fill_noise_standard(Xoroshiro128Plus::seed_from_u64(0x1337), buffer);
                Ok(PyArray1::from_slice(py, buffer).as_ref().to_object(py))
            }
            None => Ok(py.None()),
        }
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn create_worker(connection_string: String) -> PyResult<Worker> {
    // connection_string can start with tcp:// or ws://
    // parse the connection string

    let (transport, addr) = match connection_string.split("://").next() {
        Some("tcp") => (Transport::FramedTcp, connection_string.split("://").nth(1).unwrap().to_string()),
        Some("ws") => (Transport::Ws, connection_string),
        Some("wss") => (Transport::Ws, connection_string),
        _ => return Err(PyValueError::new_err(format!("Invalid connection string: {}, expected scheme://host:port where scheme is tcp, ws or wss.", connection_string))),
    };
    match Worker::new(transport, addr) {
        Ok(worker) => Ok(worker),
        Err(err) => Err(PyIOError::new_err(format!("{}", err))),
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn fdlib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Worker>()?;
    m.add_function(wrap_pyfunction!(create_worker, m)?)?;
    // m.add_function(wrap_pyfunction!(get_buffer, m)?)?;

    Ok(())
}
