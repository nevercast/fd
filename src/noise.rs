use rand::distributions::Open01;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::Rng;
use rand_xoshiro::Xoroshiro128Plus;
use rayon::prelude::ParallelBridge;
use rayon::prelude::ParallelIterator;

use crate::collect_slice::collect_slice;

/// Uniform distribution between -1.0 and 1.0, both exclusive.
struct UniformNoise;
impl Distribution<f32> for UniformNoise {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        let sample: f32 = Open01::sample(&Open01, rng);
        -1.0 + sample * 2.0
    }
}

fn jump_and_clone(rng: &mut Xoroshiro128Plus) -> Xoroshiro128Plus {
    rng.jump();
    rng.clone()
}

#[allow(dead_code)]
pub fn par_fill_noise_uniform(mut rng: Xoroshiro128Plus, buffer: &mut [f32]) {
    buffer
        .chunks_mut(100_000)
        .map(|chunk| (jump_and_clone(&mut rng).sample_iter(UniformNoise), chunk))
        .par_bridge() // This one line makes the whole thing parallel, and it's still as safe as being sequential.
        .for_each(|(mut rng, chunk)| {
            collect_slice(&mut rng, chunk);
        });
}

pub fn par_fill_noise_standard(mut rng: Xoroshiro128Plus, buffer: &mut [f32]) {
    // Check that buffer length divides by 2.
    assert!(buffer.len() % 2 == 0);

    const MU: f32 = 0.0;
    const SIGMA: f32 = 1.0;

    buffer
        .chunks_mut(100_000)
        .map(|chunk| {
            (
                jump_and_clone(&mut rng).sample_iter(Uniform::new(f32::EPSILON, 1.0)),
                chunk,
            )
        })
        .par_bridge()
        .for_each(|(mut rng, chunk)| {
            collect_slice(&mut rng, chunk);
            chunk.chunks_mut(2).for_each(|pair| {
                // Capture 2 uniform values, and make them into a standard normal.
                let (u1, u2) = (pair[0], pair[1]);
                // As u1 approaches 0.0, log(u1) approaches infinity, the uniform distribution is lower clamped to 0 + EPSILON.
                // For f32 the min/max expected values could be as large as sqrt(-2 * log(1.19209290e-07)) = 5.64666
                let mag = SIGMA * (-2.0 * u1.ln()).sqrt();
                let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos() + MU;
                let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin() + MU;

                pair[0] = z0;
                pair[1] = z1;
            });
        });
}

#[cfg(test)]
mod tests {
    use super::par_fill_noise_uniform;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128Plus;

    fn create_sized_buffer(size: usize) -> Vec<f32> {
        let mut buffer = Vec::with_capacity(size);
        for _ in 0..size {
            buffer.push(0.0);
        }
        buffer
    }

    #[test]
    fn rng_is_deterministic() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(0)
            .build()
            .expect("Creating thread pool")
            .install(|| {
                let rng1 = Xoroshiro128Plus::seed_from_u64(0);
                let mut buf1 = create_sized_buffer(100);
                let rng2 = Xoroshiro128Plus::seed_from_u64(0);
                let mut buf2 = create_sized_buffer(100);
                par_fill_noise_uniform(rng1, &mut buf1);
                par_fill_noise_uniform(rng2, &mut buf2);
                assert_eq!(buf1, buf2);

                let rng1 = Xoroshiro128Plus::seed_from_u64(0x12345678);
                let mut buf1 = create_sized_buffer(1_000_000);
                let rng2 = Xoroshiro128Plus::seed_from_u64(0x12345678);
                let mut buf2 = create_sized_buffer(1_000_000);
                par_fill_noise_uniform(rng1, &mut buf1);
                par_fill_noise_uniform(rng2, &mut buf2);
                assert_eq!(buf1, buf2);

                let rng1 = Xoroshiro128Plus::seed_from_u64(0xFEDCBA98);
                let mut buf1 = create_sized_buffer(1_999_007);
                let rng2 = Xoroshiro128Plus::seed_from_u64(0xFEDCBA98);
                let mut buf2 = create_sized_buffer(1_999_007);
                par_fill_noise_uniform(rng1, &mut buf1);
                par_fill_noise_uniform(rng2, &mut buf2);
                assert_eq!(buf1, buf2);
            });
    }

    #[test]
    fn rng_is_deterministic_one_thread() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("Creating thread pool")
            .install(|| {
                let rng1 = Xoroshiro128Plus::seed_from_u64(0);
                let mut buf1 = create_sized_buffer(100);
                let rng2 = Xoroshiro128Plus::seed_from_u64(0);
                let mut buf2 = create_sized_buffer(100);
                par_fill_noise_uniform(rng1, &mut buf1);
                par_fill_noise_uniform(rng2, &mut buf2);
                assert_eq!(buf1, buf2);

                let rng1 = Xoroshiro128Plus::seed_from_u64(0x12345678);
                let mut buf1 = create_sized_buffer(1_000_000);
                let rng2 = Xoroshiro128Plus::seed_from_u64(0x12345678);
                let mut buf2 = create_sized_buffer(1_000_000);
                par_fill_noise_uniform(rng1, &mut buf1);
                par_fill_noise_uniform(rng2, &mut buf2);
                assert_eq!(buf1, buf2);

                let rng1 = Xoroshiro128Plus::seed_from_u64(0xFEDCBA98);
                let mut buf1 = create_sized_buffer(1_999_007);
                let rng2 = Xoroshiro128Plus::seed_from_u64(0xFEDCBA98);
                let mut buf2 = create_sized_buffer(1_999_007);
                par_fill_noise_uniform(rng1, &mut buf1);
                par_fill_noise_uniform(rng2, &mut buf2);
                assert_eq!(buf1, buf2);
            });
    }

    #[test]
    fn rng_is_deterministic_prime_threads() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .expect("Creating thread pool")
            .install(|| {
                let rng1 = Xoroshiro128Plus::seed_from_u64(0);
                let mut buf1 = create_sized_buffer(100);
                let rng2 = Xoroshiro128Plus::seed_from_u64(0);
                let mut buf2 = create_sized_buffer(100);
                par_fill_noise_uniform(rng1, &mut buf1);
                par_fill_noise_uniform(rng2, &mut buf2);
                assert_eq!(buf1, buf2);

                let rng1 = Xoroshiro128Plus::seed_from_u64(0x12345678);
                let mut buf1 = create_sized_buffer(1_000_000);
                let rng2 = Xoroshiro128Plus::seed_from_u64(0x12345678);
                let mut buf2 = create_sized_buffer(1_000_000);
                par_fill_noise_uniform(rng1, &mut buf1);
                par_fill_noise_uniform(rng2, &mut buf2);
                assert_eq!(buf1, buf2);

                let rng1 = Xoroshiro128Plus::seed_from_u64(0xFEDCBA98);
                let mut buf1 = create_sized_buffer(1_999_007);
                let rng2 = Xoroshiro128Plus::seed_from_u64(0xFEDCBA98);
                let mut buf2 = create_sized_buffer(1_999_007);
                par_fill_noise_uniform(rng1, &mut buf1);
                par_fill_noise_uniform(rng2, &mut buf2);
                assert_eq!(buf1, buf2);
            });
    }

    #[test]
    fn rng_is_open_neg1_pos1_range() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(0)
            .build()
            .expect("Creating thread pool")
            .install(|| {
                let rng = Xoroshiro128Plus::seed_from_u64(0);
                let mut buf = create_sized_buffer(1_000_000);
                par_fill_noise_uniform(rng, &mut buf);
                for x in buf {
                    assert!(x.abs() < 1.0, "assert!(x.abs() < 1.0), x = {}", x);
                }
            });
    }
}
