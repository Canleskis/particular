use glam::*;
use particular::prelude::*;

use particular::gravity::newtonian::Acceleration;

const DT: f32 = 1.0 / 50.0;

// `Body` needs to implement the `Position` and `Mass` traits to allow for gravitational 
// interactions to be computed.
#[derive(Debug, Position, Mass)]
struct Body {
    velocity: Vec2,
    position: Vec2,
    // This attribute is optional. If it is missing, the value of `G` is 1.0.
    #[G = 1.0]
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
            // Calling `brute_force` returns an iterator over the calculated interaction of each body.
            .brute_force(Acceleration::checked())
            // Collecting the iterator to stop the borrow of `bodies` and allow subsequent mutation.
            // There are other ways to handle this, but this is the simplest.
            .collect::<Vec<_>>()
            .into_iter()
            // We can use the computed interaction to update the state of a body by zipping the
            // computed interactions with the mutable reference to the bodies.
            .zip(&mut bodies)
            .for_each(|(acceleration, body)| {
                // Integrating using the semi-implicit Euler method https://en.wikipedia.org/wiki/Semi-implicit_Euler_method.
                body.velocity += acceleration * DT;
                body.position += body.velocity * DT;
            });

        std::thread::sleep(std::time::Duration::from_millis(100))
    }
}
