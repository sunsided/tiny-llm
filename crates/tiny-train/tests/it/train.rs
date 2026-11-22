//! The loss, gradients and one optimizer step behave sensibly on the released weights.

use std::path::Path;

use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::Device;
use tiny_train::checkpoint::Checkpoint;
use tiny_train::eval::load;
use tiny_train::model::Model;
use tiny_train::train::{Batch, Example, clip_gradients, loss};

/// 64 question/answer pairs from the frozen test set.
fn batch(device: &Device) -> Batch {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval.jsonl");
    let examples: Vec<Example> = load(&path)
        .expect("fixture")
        .iter()
        .take(64)
        .map(|item| Example::new(&item.question, &item.refs[0]))
        .collect();

    Batch::new(&examples, device)
}

/// The released network, loaded from `tiny2.pt` onto an autodiff CPU device.
fn released(device: &Device) -> Model {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/tiny2.pt");
    Model::from_checkpoint(&Checkpoint::load_pt(&path).expect("tiny2.pt"), device).expect("model")
}

/// The trained network predicts answers far better than chance (ln 97 ≈ 4.57),
/// and a clipped `AdamW` step on the batch lowers its loss on that batch.
#[test]
fn loss_is_low_and_a_step_lowers_it() {
    let device = Device::flex().autodiff();
    let model = released(&device);
    let batch = batch(&device);
    let before = loss(&model, &batch);

    let mut grads = GradientsParams::from_grads(before.backward(), &model);
    let norm = clip_gradients(&mut grads, &model.matrix_ids(), &model.gain_ids(), 1.0);
    let before = before.into_scalar::<f32>();
    let mut optimizer = AdamWConfig::new().with_beta_2(0.98).init();
    let model = optimizer.step(1e-3, model, grads);
    let after = loss(&model, &batch).into_scalar::<f32>();

    println!("loss {before:.4} -> {after:.4}, gradient norm {norm:.3}");
    assert!(before < 1.5, "{before}");
    assert!(after < before, "{after} >= {before}");
}

/// Every parameter receives a non-zero gradient (attention included: an
/// operation without a backward pass would silently freeze its inputs).
#[test]
fn every_parameter_gets_a_gradient() {
    let device = Device::flex().autodiff();
    let model = released(&device);
    let grads = GradientsParams::from_grads(loss(&model, &batch(&device)).backward(), &model);

    for id in model.matrix_ids() {
        let grad = grads
            .get::<2>(id)
            .unwrap_or_else(|| panic!("no gradient for matrix {id}"));
        assert!(
            grad.abs().sum().into_scalar::<f32>() > 0.0,
            "zero gradient for matrix {id}"
        );
    }
    for id in model.gain_ids() {
        let grad = grads
            .get::<1>(id)
            .unwrap_or_else(|| panic!("no gradient for gain {id}"));
        assert!(
            grad.abs().sum().into_scalar::<f32>() > 0.0,
            "zero gradient for gain {id}"
        );
    }
}
