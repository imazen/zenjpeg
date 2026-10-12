//! The fuzzer and stable regression gate share these plan invariants.
pub fn check(data: &[u8]) {
    use zencodec::metadata_rewrite::Request;
    let request = Request::new(data)
        .with_byte_limits(4 << 20, 8 << 20)
        .with_metadata_limit(1 << 20)
        .with_carrier_limit(4096);
    if let Ok(plan) = zenjpeg::metadata::rewrite(&request) {
        let output = plan.to_vec().unwrap();
        assert_eq!(output.len(), plan.output_len());
        assert_eq!(output, plan.chunks().flatten().copied().collect::<Vec<_>>());
        let next = Request::new(&output)
            .with_byte_limits(8 << 20, 8 << 20)
            .with_metadata_limit(1 << 20)
            .with_carrier_limit(4096);
        let again = zenjpeg::metadata::rewrite(&next)
            .expect("a completed rewrite must be readable by its planner")
            .to_vec()
            .unwrap();
        assert_eq!(
            output, again,
            "default metadata filtering must be idempotent"
        );
    }
}
