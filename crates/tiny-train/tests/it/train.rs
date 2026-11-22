//! The loss and one optimizer step behave sensibly on the released weights.

use std::path::Path;

use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use tiny_train::checkpoint::Checkpoint;
use tiny_train::eval::load;
use tiny_train::model::Model;
use tiny_train::train::{Batch, Example, clip_gradients, loss};

/// 64 question/answer pairs from the frozen test set.
fn batch() -> Batch {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval.jsonl");
    let examples: Vec<Example> = load(&path)
        .expect("fixture")
        .iter()
        .take(64)
        .map(|item| Example::new(&item.question, &item.refs[0]))
        .collect();

    Batch::new(&examples, &Device::Cpu).expect("batch")
}

/// The released network, loaded from `tiny2.pt`.
fn released() -> Model {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/tiny2.pt");
    Model::from_checkpoint(&Checkpoint::load_pt(&path).expect("tiny2.pt"), &Device::Cpu)
        .expect("model")
}

/// The trained network predicts answers far better than chance (ln 97 ≈ 4.57),
/// and a clipped `AdamW` step on the batch lowers its loss on that batch.
#[test]
fn loss_is_low_and_a_step_lowers_it() {
    let model = released();
    let batch = batch();
    let before = loss(&model, &batch).expect("loss");
    let before_value = before.to_scalar::<f32>().expect("scalar");

    let vars = model.vars.all_vars();
    let mut grads = before.backward().expect("backward");
    let norm = clip_gradients(&mut grads, &vars, 1.0).expect("clip");
    let params = ParamsAdamW {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.98,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    AdamW::new(vars, params)
        .expect("optimizer")
        .step(&grads)
        .expect("step");
    let after = loss(&model, &batch)
        .expect("loss")
        .to_scalar::<f32>()
        .expect("scalar");

    println!("loss {before_value:.4} -> {after:.4}, gradient norm {norm:.3}");
    assert!(before_value < 1.5, "{before_value}");
    assert!(after < before_value, "{after} >= {before_value}");
}

/// Every parameter receives a non-zero gradient (attention included: an
/// operation without a backward pass would silently freeze its inputs).
#[test]
fn every_parameter_gets_a_gradient() {
    let model = released();
    let grads = loss(&model, &batch())
        .expect("loss")
        .backward()
        .expect("backward");
    let data = model.vars.data().lock().expect("var map");

    for (name, var) in data.iter() {
        let grad = grads
            .get(var)
            .unwrap_or_else(|| panic!("no gradient for {name}"));
        let size = grad
            .abs()
            .expect("abs")
            .sum_all()
            .expect("sum")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(size > 0.0, "zero gradient for {name}");
    }
}
