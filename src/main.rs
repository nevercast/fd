use nalgebra::SMatrix;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoroshiro128Plus;

const N: usize = 50;

fn next_random(rng: &mut Xoroshiro128Plus) -> f32 {
    rng.gen_range(-1.0_f32..1.0_f32)
}

fn sgd(
    model: SMatrix<f32, N, N>,
    step_size: f32,
    gradient: &SMatrix<f32, N, N>,
) -> SMatrix<f32, N, N> {
    let norm = gradient.norm();
    let gradient = gradient / norm;
    let gradient = step_size * gradient;
    model + gradient
}

fn perturbations(prng: &mut Xoroshiro128Plus, perturb_scale: f32) -> SMatrix<f32, N, N> {
    let noise = SMatrix::<f32, N, N>::from_fn(|_, _| perturb_scale * next_random(prng));
    let noise = noise / noise.norm(); // unit ball this bitch
    noise
}

fn forward_pass(model: &SMatrix<f32, N, N>, x: &SMatrix<f32, N, 1>) -> SMatrix<f32, N, 1> {
    model * x
}

fn mse(y_hat: &SMatrix<f32, N, 1>, y: &SMatrix<f32, N, 1>) -> f32 {
    let vector = y_hat - y;
    vector.component_mul(&vector).mean()
}

fn training_sample(prng: &mut Xoroshiro128Plus) -> (SMatrix<f32, N, 1>, SMatrix<f32, N, 1>) {
    let x = SMatrix::<f32, N, 1>::from_fn(|_, _| next_random(prng));
    // Compute the reverse of x, for example if x = [1, 2, 3] then y = [3, 2, 1]
    let y = SMatrix::<f32, N, 1>::from_fn(|i, _| x[(N - i) - 1]);
    (x, y)
}

const EXPERIENCE_SIZE: usize = 1000;

fn main() {
    let mut prng = Xoroshiro128Plus::seed_from_u64(12345);
    let mut model = SMatrix::<f32, N, N>::from_fn(|_, _| next_random(&mut prng));
    let mut gradients = Vec::<SMatrix<f32, N, N>>::with_capacity(EXPERIENCE_SIZE);
    let step_size = 0.20;

    for step in 0..2000 {
        let (x, y) = training_sample(&mut prng);
        let y_hat = forward_pass(&model, &x);
        let policy_error = mse(&y_hat, &y);
        if policy_error < 0.005 {
            println!("Converged after {} steps", step);
            break;
        }
        if step == 0 {
            println!("Initial error: {}", policy_error);
        } else {
            println!("Step {}: error {}", step, policy_error);
        }

        // Collect "experience"
        gradients.clear();
        for _ in 0..EXPERIENCE_SIZE {
            let noise = perturbations(&mut prng, 0.1);
            let perturbed_model = model + noise;
            let y_hat = forward_pass(&perturbed_model, &x);
            let error = mse(&y_hat, &y);
            // policy_error - error instead of reward - policy_reward because reward and error are inverted ??
            let gradient = (policy_error - error) * noise / noise.norm();
            gradients.push(gradient);
        }

        let mut average_gradient = SMatrix::<f32, N, N>::zeros();
        for gradient in &gradients {
            average_gradient += gradient / 10.0_f32;
        }

        model = sgd(model, step_size, &average_gradient);
    }

    let (x, y) = training_sample(&mut prng);
    let y_hat = forward_pass(&model, &x);
    println!("Final error: {}", mse(&y_hat, &y));

    let x = SMatrix::<f32, N, 1>::from_iterator((0..N).map(|i| 1.0 + i as f32));
    let y = forward_pass(&model, &x);
    println!("y = {:?}", y);
}
