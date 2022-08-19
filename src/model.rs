use crate::noise::par_fill_noise_standard;
use bincode::{deserialize, serialize, Result};
use rand::{thread_rng, Rng, SeedableRng};
use rand_xoshiro::Xoroshiro128Plus;
use rayon::{
    prelude::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

const PAR_CHUNK_SIZE: usize = 100_000;

pub fn serialize_parameters(parameters: &[f32]) -> Result<Vec<u8>> {
    serialize(parameters)
}

pub fn deserialize_parameters(parameters: &[u8]) -> Result<Vec<f32>> {
    deserialize(parameters)
}

pub fn permute_parameters(policy: &[f32], buffer: &mut [f32], step_size: f32) -> u64 {
    let seed = thread_rng().gen();
    let rng = Xoroshiro128Plus::seed_from_u64(seed);
    par_fill_noise_standard(rng, buffer);

    buffer
        .par_chunks_mut(PAR_CHUNK_SIZE)
        .zip(policy.par_chunks(PAR_CHUNK_SIZE))
        .for_each(|(param_chunk, policy_chunk)| {
            for (param, policy) in param_chunk.iter_mut().zip(policy_chunk) {
                *param = policy + *param * step_size;
            }
        });

    seed
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128Plus;
    use rayon::prelude::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    };

    const TEST_BUFFER_SIZE: usize = 1_000_000;

    fn create_sized_buffer(size: usize) -> Vec<f32> {
        let mut buffer = Vec::with_capacity(size);
        for _ in 0..size {
            buffer.push(0.0);
        }
        buffer
    }

    #[test]
    fn test_parameter_permutation() {
        // Create a "policy", it's just empty values.
        let policy = create_sized_buffer(TEST_BUFFER_SIZE);
        // Create a buffer to produce the permuted parameters.
        let mut permutation_buffer = create_sized_buffer(TEST_BUFFER_SIZE);
        // Permute the parameters, and return the seed for use later.
        let seed = super::permute_parameters(&policy, &mut permutation_buffer, 1.0);

        // Create an RNG from the seed to verify the seed is correct.
        let rng = Xoroshiro128Plus::seed_from_u64(seed);
        // Create a test buffer to verify the permutation is correct.
        let mut test_buffer = create_sized_buffer(TEST_BUFFER_SIZE);
        // Fill the test buffer with random values, these should be the same as the permutation buffer.
        super::par_fill_noise_standard(rng, &mut test_buffer);

        // Verify the permutation is correct.
        assert_eq!(
            permutation_buffer, test_buffer,
            "Permutations should be equal with same seed."
        );

        // Create a new permutation, the seed should be different.
        let seed_2 = super::permute_parameters(&policy, &mut permutation_buffer, 0.5);
        assert_ne!(seed, seed_2, "Seeds should be different");
        let rng = Xoroshiro128Plus::seed_from_u64(seed_2);
        let mut test_buffer2 = create_sized_buffer(TEST_BUFFER_SIZE);
        super::par_fill_noise_standard(rng, &mut test_buffer2);

        // Check that the test buffers are different, verifying that the different seed had an effect.
        assert_ne!(
            test_buffer, test_buffer2,
            "Permutations should be different with different seed."
        );

        // Divide the test_buffer by 2 to check that step_size works.
        test_buffer2.par_iter_mut().for_each(|x| *x /= 2.0);
        assert_eq!(
            permutation_buffer, test_buffer2,
            "test_buffer * 0.5 != permutation_buffer"
        );
    }

    #[test]
    fn test_parameters_biased_by_policy() {
        // Create a "policy", it's just empty values.
        let mut policy = create_sized_buffer(TEST_BUFFER_SIZE);
        // Assign the value 1.0 to every element in the policy.
        policy.par_iter_mut().for_each(|x| *x = 1.0);
        // Create a buffer to produce the permuted parameters.
        let mut permutation_buffer = create_sized_buffer(TEST_BUFFER_SIZE);
        // Permute the parameters, and capture the seed for recreation. Use a small step size
        let seed = super::permute_parameters(&policy, &mut permutation_buffer, 0.01);

        // Create an RNG from the seed to verify the seed is correct.
        let rng = Xoroshiro128Plus::seed_from_u64(seed);
        // Create a test buffer to verify the permutation is correct.
        let mut test_buffer = create_sized_buffer(TEST_BUFFER_SIZE);
        // Fill the test buffer with random values, these should be the same as the permutation buffer.
        super::par_fill_noise_standard(rng, &mut test_buffer);

        // Apply the expected transform of the permutation.
        test_buffer = test_buffer.par_iter().map(|x| 1.0 + x / 100.0).collect();

        // Verify that permutation_buffer and test_buffer are equal within f32::EPSILON.
        permutation_buffer
            .par_iter()
            .zip(test_buffer.par_iter())
            .for_each(|(x, y)| {
                assert!(
                    (*x - *y).abs() < 2.0 * f32::EPSILON,
                    "Policy permutation transformations should work."
                );
            });
    }
}
