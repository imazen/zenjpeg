//! Demo: structure a finite-budget sweep over the encoder knob space.
//!
//! Run: cargo run --release --example sweep_plan --features __expert

use zenjpeg::encode::sweep::{QualityGrid, SweepAxes, SweepBuilder};

fn main() {
    for (name, axes, grid, budget) in [
        (
            "rd_core / step5",
            SweepAxes::rd_core(),
            QualityGrid::Step5,
            None,
        ),
        (
            "modes_full / step5, budget 2000",
            SweepAxes::modes_full(),
            QualityGrid::Step5,
            Some(2000),
        ),
        (
            "modes_full / training-dense, budget 2000",
            SweepAxes::modes_full(),
            QualityGrid::TrainingDense,
            Some(2000),
        ),
    ] {
        let mut b = SweepBuilder::new(axes, grid);
        if let Some(n) = budget {
            b = b.with_budget(n);
        }
        let plan = b.plan();
        println!("== {name}");
        println!(
            "   cells={} merged_aliases={} invalid={} q_coarsenings={} over_budget={}",
            plan.cells.len(),
            plan.duplicates_merged,
            plan.invalid_skipped.len(),
            plan.q_coarsenings,
            plan.over_budget
        );
        for d in &plan.dropped {
            println!(
                "   dropped axis {}: kept [{}], dropped {:?}",
                d.axis, d.kept, d.dropped
            );
        }
        println!(
            "   encodes for 50 images x 4 sizes = {}",
            plan.encodes(50, 4)
        );
        for cell in plan.cells.iter().take(5) {
            println!(
                "   {}\tq={}\tfp={:016x}\taliases={}",
                cell.id,
                cell.quality,
                cell.fingerprint,
                cell.aliases.len()
            );
        }
    }
}
