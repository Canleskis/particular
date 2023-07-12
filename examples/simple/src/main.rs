use glam::*;
use particular::prelude::*;

const DT: f32 = 1.0 / 50.0;

#[derive(Debug, Particle)]
struct Body {
    velocity: Vec2,
    position: Vec2,
    mu: f32,
}

fn main() {
    // Collection (vector) of two bodies.
    let mut bodies = vec![
        Body {
            velocity: Vec2::ZERO,
            position: Vec2::ZERO,
            mu: 1E6,
        },
        Body {
            velocity: Vec2::new(0.0, 100.0),
            position: Vec2::new(100.0, 0.0),
            mu: 0.0,
        },
    ];

    loop {
        let orbiting_body = &bodies[1];
        print!("\x1B[2J\x1B[1;1H");
        println!(
            "Body {{\n\
            \x20   velocity: {:?}\n\
            \x20   position: {:?}\n\
            }}",
            orbiting_body.velocity, orbiting_body.position,
        );

        bodies
            .iter()
            // Calling accelerations returns an iterator over the acceleration of each body.
            .accelerations(sequential::BruteForce)
            // Zipping the accelerations with a mutable reference to the bodies 
            // to change the state of each body using their computed acceleration.
            .zip(&mut bodies)
            .for_each(|(acceleration, body)| {
                // Integrating using the semi-implicit Euler method https://en.wikipedia.org/wiki/Semi-implicit_Euler_method.
                body.velocity += acceleration * DT;
                body.position += body.velocity * DT;
            });

        std::thread::sleep(std::time::Duration::from_millis(100))
    }
}
