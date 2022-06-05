use drhall_bigdecimal::*;
#[test]
fn bigint_api_1() {
	let factorial_degree = 20;
	let mut factorial = BigInt::one();
	for n in 0..factorial_degree{
		factorial *= n;
	}
	assert_eq!(
		factorial,
		BigInt::from_str("1405006117752879898543142606244511569936384000000000").unwrap()
	);
}