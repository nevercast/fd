use std::thread;

use rand::{distributions, Rng, SeedableRng};
use rand_xoshiro::Xoshiro128Plus;
use rayon::{
    prelude::{IntoParallelIterator, ParallelExtend, ParallelIterator},
    ThreadPoolBuilder,
};

pub struct WorkerReturn {
    pub policy_version: u32,
    pub noise_scale: f32,
    pub reward: f64,
    pub seed: u64,
}

struct WorkerInitialise {
    pub seed: [u8; 16],
}

const MODEL_SIZE: usize = 1_000_000;
const STACK_SIZE: usize = 32 * 1024 * 1024;

fn noise(prng: &mut Xoshiro128Plus) -> [f32; MODEL_SIZE] {
    let mut noise = [0.0; MODEL_SIZE];
    prng.sample_iter(distributions::Open01)
        .take(MODEL_SIZE)
        .collect::<Vec<f32>>()
        .into_iter()
        .enumerate()
        .for_each(|(i, v)| noise[i] = v);
    noise
}

fn combine_noise(a: &mut [f32; MODEL_SIZE], b: &[f32; MODEL_SIZE]) {
    a.iter_mut()
        .zip(b.iter())
        .for_each(|(a, b)| *a = (*a + *b) / 2.0);
}

fn big_stack_energy(noise_buf: &Box<Vec<[f32; MODEL_SIZE]>>) {
    let start = std::time::Instant::now();
    let noise_product = noise_buf.iter().fold([0.0; MODEL_SIZE], |mut acc, noise| {
        combine_noise(&mut acc, noise);
        acc
    });
    let end = start.elapsed();
    println!("{:?}", end);
    // Slice the end of the accumulator and print it
    let end_slice = &noise_product[MODEL_SIZE - 100..];
    println!("{:?}", end_slice);
}

struct NoiseGen(Xoshiro128Plus);

impl Clone for NoiseGen {
    fn clone(&self) -> Self {
        let mut prng = self.0.clone();
        prng.jump();
        println!("Produced noise generator");
        NoiseGen(prng)
    }
}

fn create_noise_buf() -> Box<Vec<[f32; MODEL_SIZE]>> {
    let mut noise_buf = Box::new(Vec::<[f32; MODEL_SIZE]>::with_capacity(1_000));
    let prng_source = Xoshiro128Plus::from_seed([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]);
    let noise_gen = NoiseGen(prng_source);

    // Use rayon iterator to parallelize the noise generation
    noise_buf.par_extend(
        (0..100)
            .into_par_iter()
            .map_with(noise_gen, |job_prng, _| noise(&mut job_prng.0)),
    );

    noise_buf
}

fn main() {
    ThreadPoolBuilder::new()
        .stack_size(STACK_SIZE)
        .build_global()
        .unwrap();

    // Fill the noise buf in separate thread
    let noise_buf: Box<Vec<[f32; MODEL_SIZE]>> = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(create_noise_buf)
        .unwrap()
        .join()
        .unwrap();

    // Time the function call to noise for benchmarking
    let child = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || {
            big_stack_energy(&noise_buf);
        })
        .unwrap();

    // Wait for thread to join
    child.join().unwrap();
}
