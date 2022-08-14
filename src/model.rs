use bincode::{deserialize, serialize, Result};

pub struct Model {
    model: Vec<f32>,
}

impl Model {
    pub fn new(model: Vec<f32>) -> Model {
        Model { model }
    }

    pub fn as_bytes(&self) -> Result<Vec<u8>> {
        serialize(&self.model)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Model> {
        Ok(Model {
            model: deserialize::<Vec<f32>>(bytes)?,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.model.len()
    }

    pub fn get_ref(&self) -> &[f32] {
        &self.model
    }

    pub fn update_chunk(&mut self, chunk: &[f32], offset: usize) {
        self.model[offset..offset + chunk.len()].copy_from_slice(chunk);
    }
}
