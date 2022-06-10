//#![feature(backtrace)] // uncomment for debugging
use std::fmt::{Debug, Formatter, Display};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::*;
use std::borrow::Borrow;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BigInt {
	digits: Vec<u8>,
	positive: bool
}

impl BigInt {

	pub fn zero() -> BigInt {BigInt{digits: vec![0u8], positive:true }}
	pub fn one() -> BigInt {BigInt{digits: vec![1u8], positive:true }}
	pub fn neg_one() -> BigInt {BigInt{digits: vec![1u8], positive:true }}

	pub fn is_zero(&self) -> bool {
		for i in (0..self.digits.len()).rev(){
			if self.digits[i] != 0u8 {return false;}
		}
		return true;
	}

	pub fn get_sign_num(&self) -> isize {
		match self.positive {
			true => 1,
			false => -1
		}
	}

	pub fn abs(&self) -> Self {
		return self.convert_sign(true);
	}

	fn convert_sign(&self, positive: bool) -> Self{
		return BigInt{digits: self.digits.clone(), positive: positive};
	}

	/// for internal use only, returns slice with leading zeroes trimmed off
	fn get_nonzero_slice(&self) -> &[u8] {
		if self.digits.len() == 0 {
			return &self.digits;
		}
		for i in (0..self.digits.len()).rev(){
			if self.digits[i] != 0u8 {
				return &self.digits[0..i+1];
			}
		}
		// all zeroes
		return &self.digits[0..1];
	}

	pub fn pow(self, exponent: u64) -> Self {
		match exponent {
			0 => Self::one(),
			1 => self.clone(),
			2 => self.clone() * self.clone(),
			3 => self.clone() * self.clone() * self.clone(),
			_ => {
				// algorithm: keep squaring until squaring again would be too much
				// then recursively call with remaining exponent
				// then linearly multiple with the remainder
				let mut total = self.clone();
				let mut accum_exp = 1;
				while (accum_exp * 2) <= exponent {
					total = total.clone() * total.clone();
					accum_exp *= 2;
				}
				let remaining_exp = exponent.checked_sub(accum_exp).unwrap();
				return total * self.pow(remaining_exp);
			}
		}
	}
	pub fn from_str(s: &str) -> Result<BigInt, errors::ParseError> {
		return Self::from_localized_str(s, '.', vec![',', '_']);
	}
	pub fn from_localized_str(s: &str, decimal_char: char, separator_chars: Vec<char>)
			-> Result<BigInt, errors::ParseError> {
		let text = s.trim();
		let mut digits: Vec<u8> = Vec::with_capacity(text.len());
		let mut sign: Option<bool> = None;
		for c in text.chars().rev() {
			if separator_chars.contains(&c){
				// do nothing
				continue;
			} else if c == decimal_char {
				// decimal place!
				return Err(errors::ParseError::new(text, "BigInt"));
			} else if sign.is_some() {
				// more digits in front of minus sign?
				return Err(errors::ParseError::new(text, "BigInt"));
			}
			match c {
				'0' => digits.push(0u8),
				'1' => digits.push(1u8),
				'2' => digits.push(2u8),
				'3' => digits.push(3u8),
				'4' => digits.push(4u8),
				'5' => digits.push(5u8),
				'6' => digits.push(6u8),
				'7' => digits.push(7u8),
				'8' => digits.push(8u8),
				'9' => digits.push(9u8),
				'-' => sign = Some(false), // if not the end of the iterator, next iter will err
				'–' => sign = Some(false), // figure dash, in case you copy-paste from Word
				_ => return Err(errors::ParseError::new(text, "BigInt"))
			}
		}
		return Ok(BigInt{digits:digits, positive: sign.unwrap_or(true)});
	}

	pub fn from_u8(i: u8) -> BigInt {
		return Self::from_i32(i as i32);
	}
	pub fn from_i8(i: i8) -> BigInt {
		return Self::from_i32(i as i32);
	}
	pub fn from_u16(i: u16) -> BigInt {
		return Self::from_i32(i as i32);
	}
	pub fn from_i16(i: i16) -> BigInt {
		return Self::from_i32(i as i32);
	}
	pub fn from_u32(i: u32) -> BigInt {
		return Self::from_i64(i as i64);
	}
	pub fn from_i32(i: i32) -> BigInt {
		let unsigned = Self::from_u64((i as i64).abs() as u64);
		return BigInt{digits: unsigned.digits, positive: i >= 0}
	}
	pub fn from_u64(i: u64) -> BigInt {
		let mut remainder = i;
		let mut digs: Vec<u8> = Vec::with_capacity(20);
		while remainder >= 10{
			let digit = remainder % 10;
			digs.push(digit as u8);
			remainder = remainder / 10;
		}
		if remainder != 0 {
			digs.push(remainder as u8);
		}
		return BigInt{digits: digs, positive: true}
	}
	pub fn from_i64(i: i64) -> BigInt {
		let mut remainder = i;
		let mut digs: Vec<u8> = Vec::with_capacity(20);
		// special handling to avoid .abs() overflow
		digs.push(((remainder & 0x0F) % 10) as u8);
		let mut remainder: u64 = (remainder / 10).abs() as u64;
		while remainder >= 10{
			let digit = remainder % 10;
			digs.push(digit as u8);
			remainder = remainder / 10;
		}
		if remainder != 0 {
			digs.push(remainder as u8);
		}
		return BigInt{digits: digs, positive: i >= 0}
	}
	pub fn from_u128(i: u128) -> BigInt {
		let mut remainder = i;
		let mut digs: Vec<u8> = Vec::with_capacity(39);
		while remainder >= 10{
			let digit = remainder % 10;
			digs.push(digit as u8);
			remainder = remainder / 10;
		}
		if remainder != 0 {
			digs.push(remainder as u8);
		}
		return BigInt{digits: digs, positive: true}
	}
	pub fn from_i128(i: i128) -> BigInt {
		let mut remainder = i;
		let mut digs: Vec<u8> = Vec::with_capacity(39);
		// special handling to avoid .abs() overflow
		digs.push(((remainder & 0x0F) % 10) as u8);
		let mut remainder: u128 = (remainder / 10).abs() as u128;
		while remainder >= 10{
			let digit = remainder % 10;
			digs.push(digit as u8);
			remainder = remainder / 10;
		}
		if remainder != 0 {
			digs.push(remainder as u8);
		}
		return BigInt{digits: digs, positive: i >= 0}
	}

	fn trim_leading_zeroes(digits: &mut Vec<u8>) {
		if digits.len() == 1 {
			return;
		}
		for i in (0..digits.len()).rev() {
			if digits[i] != 0u8 {
				// [i] not zero, trim everything above i
				digits.truncate(i+1);
				break;
			}
		}

	}

	fn delta(a: &BigInt, b: &BigInt) -> BigInt {
		if b > a {return Self::delta(b, a);}
		if a == b {return Self::zero();}
		// a > b
		let mut top = a.digits.clone();
		let bottom = &b.digits;
		let mut output: Vec<u8> = Vec::with_capacity(top.len());
		for i in 0..top.len() {
			let mut x = top[i];
			if i < bottom.len() {
				let y = bottom[i];
				if y > x {
					let mut ii = i+1;
					while top[ii] == 0 {
						top[ii] = 9u8;
						ii += 1;
					}
					top[ii] = (top[ii] as i16 - 1i16) as u8;
					x += 10;
				}
				output.push((x as i16 - y as i16) as u8);
			} else {
				output.push(x);
			}
		}
		return BigInt{digits: output, positive: true};
	}
}
impl From<u8> for BigInt {
	fn from(i: u8) -> Self {
		Self::from_u8(i)
	}
}
impl From<i8> for BigInt {
	fn from(i: i8) -> Self {
		Self::from_i8(i)
	}
}
impl From<u16> for BigInt {
	fn from(i: u16) -> Self {
		Self::from_u16(i)
	}
}
impl From<i16> for BigInt {
	fn from(i: i16) -> Self {
		Self::from_i16(i)
	}
}
impl From<u32> for BigInt {
	fn from(i: u32) -> Self {
		Self::from_u32(i)
	}
}
impl From<i32> for BigInt {
	fn from(i: i32) -> Self {
		Self::from_i32(i)
	}
}
impl From<u64> for BigInt {
	fn from(i: u64) -> Self {
		Self::from_u64(i)
	}
}
impl From<i64> for BigInt {
	fn from(i: i64) -> Self {
		Self::from_i64(i)
	}
}
impl From<u128> for BigInt {
	fn from(i: u128) -> Self {
		Self::from_u128(i)
	}
}
impl From<i128> for BigInt {
	fn from(i: i128) -> Self {
		Self::from_i128(i)
	}
}



impl Display for BigInt{
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		let mut digits = self.digits.clone();
		BigInt::trim_leading_zeroes(& mut digits);
		let mut text = String::with_capacity(digits.len()+1);
		if !self.positive{
			text.push('-');
		}
		if self.digits.len() == 0{
			text.push(std::char::from_digit(0, 10).ok_or(std::fmt::Error)?);
		} else {
			for reverse_index in (0..digits.len()).rev() {
				text.push(std::char::from_digit(digits[reverse_index] as u32, 10)
					.ok_or(std::fmt::Error)?);
			}
		}
		f.write_str(text.as_ref())
	}
}

impl PartialEq<Self> for BigInt {
	fn eq(&self, other: &Self) -> bool {
		if self.is_zero() && other.is_zero() {
			return true;
		}
		let this_slice = self.get_nonzero_slice();
		let that_slice = other.get_nonzero_slice();
		return self.positive == other.positive && this_slice.eq(that_slice);
	}
}

impl PartialEq<&Self> for &BigInt {
	fn eq(&self, other: &&Self) -> bool {
		return BigInt::eq(*self, **other);
	}
}

impl Eq for BigInt{}

impl PartialOrd<Self> for BigInt {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		return Some(self.cmp(other));
	}
}

impl Ord for BigInt {
	fn cmp(&self, other: &Self) -> Ordering {
		if self.is_zero() && other.is_zero(){
			return Ordering::Equal;
		}
		let s1 = self.get_nonzero_slice();
		let s2 = other.get_nonzero_slice();
		let l1 = s1.len();
		let l2= s2.len();
		let len_cmp = usize::cmp(&l1, &l2);
		match len_cmp {
			Ordering::Equal => {
				for i in (0..l1).rev() {
					if s1[i] != s2[i]{
						return s1[i].cmp(&s2[i]);
					}
				}
				return Ordering::Equal;
			},
			_ => len_cmp
		}
	}
}

impl Hash for BigInt{
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.get_sign_num().hash(state);
		for d in &self.digits {
			d.hash(state);
		}
	}
}

// NOTE: 'rhs' is short for 'right-hand side', as in the number on the right of the operator
// (function is called by the number on the left)

impl Neg for BigInt {
	type Output = Self;
	fn neg(self) -> Self {
		return (&self).neg();
	}
}
impl Add for BigInt {
	type Output = Self;
	fn add(self, rhs: Self) -> Self {
		return (&self).add(&rhs);
	}
}
impl AddAssign for BigInt {
	fn add_assign(&mut self, rhs: Self) {
		let tmp = self.clone().add(rhs);
		self.digits = tmp.digits.clone();
		self.positive = tmp.positive;
	}
}
impl Sub for BigInt {
	type Output = Self;
	fn sub(self, rhs: Self) -> Self {
		return (&self).sub(&rhs);
	}
}
impl SubAssign for BigInt {
	fn sub_assign(&mut self, rhs: Self) {
		let tmp = self.clone().sub(rhs);
		self.digits = tmp.digits.clone();
		self.positive = tmp.positive;
	}
}
impl Mul for BigInt {
	type Output = Self;
	fn mul(self, rhs: Self) -> Self::Output {
		return (&self).mul(&rhs);
	}
}
impl MulAssign for BigInt {
	fn mul_assign(&mut self, rhs: Self) {
		let tmp = self.clone().mul(rhs);
		self.digits = tmp.digits.clone();
		self.positive = tmp.positive;
	}
}
impl Div for BigInt {
	type Output = Self;
	fn div(self, rhs: Self) -> Self::Output {
		return (&self).div(&rhs);
	}
}
impl DivAssign for BigInt {
	fn div_assign(&mut self, rhs: Self) {
		let tmp = self.clone().div(rhs);
		self.digits = tmp.digits.clone();
		self.positive = tmp.positive;
	}
}
impl Rem for BigInt {
	type Output = Self;
	fn rem(self, rhs: Self) -> Self::Output {
		return (&self).rem(&rhs);
	}
}
impl RemAssign for BigInt {
	fn rem_assign(&mut self, rhs: Self) {
		let tmp = self.clone().rem(rhs);
		self.digits = tmp.digits.clone();
		self.positive = tmp.positive;
	}
}

impl Neg for &BigInt {
	type Output = BigInt;
	fn neg(self) -> BigInt {
		return BigInt{digits: self.digits.clone(), positive: !self.positive};
	}
}
impl Add for &BigInt {
	type Output = BigInt;
	fn add(self, rhs: Self) -> BigInt {
		if self.positive == rhs.positive {
			// same sign
			let count = self.digits.len().max(rhs.digits.len());
			let mut new_digits: Vec<u8> = Vec::with_capacity(count+1);
			let mut i = 0;
			let mut carry: u8 = 0;
			while i < count {
				let a = self.digits.get(i).unwrap_or(&0u8);
				let b = rhs.digits.get(i).unwrap_or(&0u8);
				let sum = a + b + carry;
				new_digits.push(sum % 10);
				carry = sum / 10;
				i += 1;
			}
			if carry != 0 {
				new_digits.push(carry);
			}
			return BigInt{digits: new_digits, positive: self.positive};
		} else {
			// opposite signs: compute delta and return delta with sign of bigger (abs) number
			let a = self.abs();
			let b = rhs.abs();
			if a == b {
				return BigInt::zero();
			}else if a >= b {
				let delta = BigInt::delta(&a, &b);
				return delta.convert_sign(self.positive);
			} else {
				let delta = BigInt::delta(&b, &a);
				return delta.convert_sign(rhs.positive);
			}
		}
	}
}
impl Sub for &BigInt {
	type Output = BigInt;
	fn sub(self, rhs: Self) -> BigInt {
		return self.add(&rhs.neg());
	}
}
impl Mul for &BigInt {
	type Output = BigInt;
	fn mul(self, rhs: Self) -> BigInt {
		let sign = self.positive == rhs.positive;
		let count = usize::max(self.digits.len(), rhs.digits.len());
		if count < 9 {
			// small enough for i64 multiply
			return BigInt{digits: BigInt::i64_to_vec_u8(
				BigInt::vec_u8_to_i64(&self.digits)
				* BigInt::vec_u8_to_i64(&rhs.digits),
			), positive: sign};
		}
		// too big, time to do it old-school
		if rhs.digits.len() > self.digits.len(){
			// reverse to make sure the left side has most digits
			return rhs.mul(self);
		}
		let max_digits = self.digits.len()+rhs.digits.len();
		let mut new_digits: Vec<u8> = Vec::with_capacity(max_digits);
		for _ in 0..max_digits {new_digits.push(0u8);}
		let mut carry: u8 = 0u8;
		for bi in 0..rhs.digits.len() {
			let base_digit = rhs.digits[bi];
			for ai in 0..self.digits.len() {
				let upper_digit = self.digits[ai];
				let prod = upper_digit * base_digit + carry;
				let p_digit = new_digits[ai+bi] + prod % 10;
				new_digits[ai+bi] = p_digit % 10;
				carry = prod / 10 + p_digit / 10;
			}
			new_digits[self.digits.len()+bi] += carry;
			carry = 0;
		}
		BigInt::trim_leading_zeroes(&mut new_digits);
		return BigInt{digits: new_digits, positive: sign};
	}
}
impl Div for &BigInt {
	type Output = BigInt;
	fn div(self, rhs: Self) -> BigInt {
		if rhs.is_zero() {
			panic!("attempt to divide by zero");
		}
		let q = self.checked_div_rem(&rhs).unwrap().0;
		return q;
		//return self.checked_div_rem(&rhs).unwrap().0;
	}
}
impl Rem for &BigInt {
	type Output = BigInt;
	fn rem(self, rhs: Self) -> BigInt {
		if rhs.is_zero() {
			panic!("attempt to calculate remainder of divide by zero");
		}
		return self.checked_div_rem(&rhs).unwrap().1;
	}
}


impl BigInt {
	// division
	const BZ_DIV_LIMIT: i64 = 18;
	// safe-division
	pub fn checked_div_rem(&self, rhs: &Self) -> Result<(Self,Self),errors::MathError> {
		if rhs.is_zero() {
			return Err(errors::MathError::new("cannot divide by zero"));
		}
		if self == rhs {
			return Ok((Self::one(), Self::zero()));
		}
		let a = self.abs();
		let b = rhs.abs();
		if a == b {
			return Ok((Self::neg_one(), Self::zero()));
		} else if a < b {
			return Ok((Self::zero(), a));
		}
		let sign = a.positive == b.positive;
		let mut a_digits = a.digits.clone();
		let mut b_digits = b.digits.clone();
		if (a_digits.len() as i64) < Self::BZ_DIV_LIMIT {
			// small enough to use i64 integer division
			let (q, r) = Self::simple_division_64bit(&a_digits, &b_digits);
			return Ok((Self::from_i64(q).convert_sign(sign), Self::from_i64(r)));
		}
		// time for long division, oh boy!
		// (see Burnikel and Ziegler's 1998 paper)
		//    ____
		//  a) b
		let (q, r) = Self::burnikel_ziegler_division(&a_digits, &b_digits)?;
		println!("checked_div_rem({}, {}) -> ({:?}, {:?})", self, rhs, q, r); // uncomment to debug
		return Ok((BigInt{digits: q, positive: sign}, BigInt{digits: r, positive: true}))
	}

	fn is_usize_odd(u: &usize) -> bool {
		return u & 0x01 == 1;
	}

	#[allow(non_snake_case)]
	fn burnikel_ziegler_division(A: &Vec<u8>, B: &Vec<u8>) -> Result<(Vec<u8>, Vec<u8>), errors::MathError> {
		// NOTE: vec<u8>[0] is LEAST SIGNIFICANT DIGIT, not most significant digit
		// do recursive integer division A/B and return (quotient, remainder)
		let r = A.len() as i64;
		let s = B.len() as i64;
		if r < s {
			// A smaller than B
			return Ok((vec![0u8], A.clone()));
		}
		/*
A: [  _|___|___|___|___]     B: [ __]
   |---| <- n                   |---| <- n = j * m
   |-------------------| <- t * n
_=j

	"The main idea is to split up A into parts which are as long as B and to view these
parts as (very large) digits. Then we can conceptually apply an ordinary school division
to these large digits in which the base task of dividing two digits by one is carried out by
RecursiveDivision. The divisor for RecursiveDivision is always B and the dividend is
always the composition of the current remainder and the next digit of A. Figure 2 illustrates
the method.
	First we conceptually divide B into m so called \division blocks" of DIV_LIMIT digits each.
We compute m as the minimal 2k that is greater or equal than the number of division blocks of
B. In other words, k will be the depth of the recursion applied to B in RecursiveDivision.
We have to extend B to a number that has m*DIV_LIMIT digits. Note that the topmost
recursive call will work no matter how big DIV_LIMIT actually is, i.e., we can try to minimize
the number of digits in each division block (at least conceptually) as long as the overall
number of digits of the m division blocks is greater or equal than the number of digits of
B. This trick will minimize the number of 0s we waste when extending B and leave us with
smaller numbers in the base case of the recursion; our tests have indicated that we gain a
speedup of 25% in running time. So we compute n = m * j with j minimal such that n is
greater or equal than s. Then we extend B from the right to n digits. B has to be shifted
bitwise again for normalization until its most signicant digit fulfills bn-1 >= beta/2. Note that
A has also to be extended and shifted by the same amount.
	Then we conceptually divide A into t division blocks of length n each. The division we
will carry out will then conceptually be a school division of a t-block number by a 1-block
number (here the term \block" stands for a very large block of digits, namely for n digits).
This is a particularly simple type of school division since the divisor consists of only one
block, which now plays the role of a single digit. No backmultiplication is needed, so school
division becomes a linear time algorithm in the number of blocks."
- Burnikel & Ziegler, 1998*/
		// m is smallest power of 2 that is at least as big as the number of DIV_LIMIT blocks in B
		// (m = 2^k)
		let mut k: i64 = 0;//Self::log2_i64(1+(s-1)/Self::BZ_DIV_LIMIT);
		while (BigInt::BZ_DIV_LIMIT << k) <= s && k < 62 {k += 1;}
		if k >= 62 { return Err(errors::MathError::new(&format!("cannot compute division because algorithm cannot handle {} digits", s)));}
		let m = 1i64 << k; // m number of blocks
		let j = 1+((s-1) / m); // j is size of smallest chunk in B to contain B using m blocks
		let n = j * m; // n total digits for B (right-pad with 0)
		let mut sigma: i64 = n-s; // sigma number of zeros to right-pad
		println!("dividing {:?} by {:?}",
				 A, B); // uncomment to debug division
		println!("r\ts\tk\tm\tj\tn\tsigma\n{}\t{}\t{}\t{}\t{}\t{}\t{}",
			r,s,k,m,j,n,sigma); // uncomment to debug division
		let B = Self::left_shift(B, sigma as usize, 0u8);
		let mut A = Self::left_shift(A, sigma as usize, 0u8);
		let t = (1 + (r/n)).max(2); // t is number of n-sized blocked needed to hold A with an extra 0 on the left
		while (A.len() as i64) < t*n { // left-pad A with zeros
			A.push(0u8);
		}
		let mut Z_double_block = A[((t-2)*n) as usize .. ((t)*n) as usize].to_vec(); // Z initialized as upper two blocks od A
		let mut Q: Vec<u8> = Vec::with_capacity((r-s+1) as usize);
		let mut R: Vec<u8> = Vec::with_capacity((s) as usize);
		for i in (0..t-1).rev() {
			println!("\ti {}\n\tZi {:?}",
					 i,Z_double_block); // uncomment to debug division
			let (Qi, Ri) = Self::recursive_division(&Z_double_block, &B);
			assert_eq!(BigInt{digits: Z_double_block.clone(), positive: true},
					   BigInt{digits: B.clone(), positive: true} * BigInt{digits: Qi.clone(), positive: true}
						   + BigInt{digits: Ri.clone(), positive: true});// in-line unit test
			// push Qi to front of Q
			Q = Self::merge2(&Qi, &Q);
			R = Ri.clone();
			if i > 0 {
				// make new double-block with remainder as upper digits and next block from A as lower digits
				Z_double_block = Self::merge2(&A[((i-1)*n) as usize .. ((i)*n) as usize], &Ri);
			}
			println!("\tQi {:?}\n\tRi {:?}\n\tQ {:?}\n\tR {:?}\n\tZi-1 {:?}\n",
					 Qi,Ri,Q,R,Z_double_block); // uncomment to debug division
		}
		// right-shift remainder back
		let mut R: Vec<u8> = R[sigma as usize..].to_vec();
		// trim leading zeroes
		Self::trim_leading_zeroes(&mut Q);
		Self::trim_leading_zeroes(&mut R);
		// done!
		return Ok((Q, R));
	}


	fn recursive_division(a_digits: &[u8], b_digits: &[u8]) -> (Vec<u8>, Vec<u8>){
		let n = b_digits.len();
		if (a_digits.len() as i64) < Self::BZ_DIV_LIMIT { // || n.is_odd() ?
			let (q64, r64) = Self::simple_division_64bit(&a_digits.to_vec(), &b_digits.to_vec());
			let q = Self::i64_to_vec_u8(q64);
			let r = Self::i64_to_vec_u8(r64);
			println!("recursive_division({:?}, {:?} -> ({:?}, {:?})",
					a_digits, b_digits, Self::left_pad(&q, n, 0u8), Self::left_pad(&r, n, 0u8)); // uncomment to debug division
			return (Self::left_pad(&q, n, 0u8), Self::left_pad(&r, n, 0u8));
		} else {
			let (b2, b1) = Self::slice2(b_digits);
			let (a34, a12) = Self::slice2(a_digits);
			let (a2, a1) = Self::slice2(a12);
			let (a4, a3) = Self::slice2(a34);
			let (q2, r) = Self::div_three_long_halves_by_two(a1, a2, a3, b1, b2);
			let (r2, r1) = Self::slice2(&r);
			let (q1, s) = Self::div_three_long_halves_by_two(r1, r2, a4, b1, b2);
			let q = Self::merge2(&q2, &q1);
			println!("recursive_division({:?}, {:?} -> ({:?}, {:?})",
					 a_digits, b_digits, q, s); // uncomment to debug division
			return (q, s);
		}
	}
	fn div_three_long_halves_by_two(a1: &[u8], a2: &[u8], a3: &[u8], b1: &[u8], b2: &[u8])
			-> (Vec<u8>, Vec<u8>) {
		let n = b2.len(); // number of digits
		let a12 = Self::merge2(a2, a1);
		let b = BigInt{digits: Self::merge2(b2, b1), positive: true};
		let (mut q, mut c) = Self::recursive_division(&a12, b1);
		let mut qnum = BigInt{digits: q, positive:true};
		let b2num = BigInt{digits: b2.to_vec(), positive:true};
		let cnum = BigInt{digits: c, positive:true};
		let a3num = BigInt{digits: a3.to_vec(), positive:true};
		let d = &qnum*&b2num;
		let mut r = (cnum*BigInt::from_u8(10) + a3num) - d;
		while !r.positive { // negative r means q too big
			qnum -= BigInt::one();
			r += b.clone();
		}
		while qnum.digits.len() < n {qnum.digits.push(0u8);}
		while r.digits.len() < n {r.digits.push(0u8);}
		return (qnum.digits[0..n].to_vec(), r.digits[0..n].to_vec());
	}

	fn left_shift(v: &Vec<u8>, pad_count: usize, pad: u8) -> Vec<u8> {
		let mut out: Vec<u8> = Vec::with_capacity(v.len() + pad_count);
		for _ in 0..pad_count{
			out.push(pad);
		}
		for d in v{
			out.push(*d);
		}
		return out;
	}
	fn left_pad(v: &Vec<u8>, final_size: usize, pad: u8) -> Vec<u8> {
		let mut v2 = v.clone();
		while v2.len() < final_size {
			v2.push(pad);
		}
		return v2;
	}
	fn log2_i64(n: i64) -> i64{
		let mut l2: i64 = 0;
		let mut p: i64 = 1;
		while p < n {
			p = p << 1;
			l2 += 1;
		}
		return l2;
	}
	fn write_into_vec(dst: &mut Vec<u8>, src: &[u8], pos: usize){
		while dst.len() < pos + src.len() {
			dst.push(0u8);
		}
		for i in 0..src.len() {
			dst[i+pos] = src[i];
		}
	}
	fn slice2(v: &[u8]) -> (&[u8],&[u8]) {
		let w = v.len();
		let h = w / 2;
		return (&v[0..h], &v[h..w]);
	}
	fn merge2(v1: &[u8], v2: &[u8]) -> Vec<u8> {
		let mut v: Vec<u8> = Vec::with_capacity(v1.len() + v2.len());
		v.append(&mut v1.to_vec());
		v.append(&mut v2.to_vec());
		return v;
	}
	fn i64_to_vec_u8(integer: i64) -> Vec<u8>{
		let mut i = integer;
		let mut v: Vec<u8> = Vec::with_capacity(19);
		while i > 0 {
			v.push((i % 10) as u8);
			i = i / 10;
		}
		return v;
	}
	fn vec_u8_to_i64(v: &Vec<u8>) -> i64 {
		let mut integer: i64 = 0;
		for i in (0..v.len() as usize).rev() {
			integer = integer * 10 + v[i] as i64;
		}
		return integer;
	}
	fn simple_division_64bit(a_digits: &Vec<u8>, b_digits: &Vec<u8>) -> (i64, i64) {
		assert!(a_digits.len() < 19 && b_digits.len() < 19);
		// small enough to use i32 integer division
		let n = Self::vec_u8_to_i64(a_digits);
		let d = Self::vec_u8_to_i64(b_digits);
		let q = n / d;
		let r = n % d;
		return (q, r);
	}
}


// IMPLEMENTATION FOR INTERACTION WITH NATIVE INT TYPES //
impl PartialEq<u8> for BigInt {
	fn eq(&self, other: &u8) -> bool {
		todo!()
	}
}
impl PartialEq<BigInt> for u8 {
	fn eq(&self, other: &BigInt) -> bool {
		todo!()
	}
}
impl Add<u8> for BigInt {
	type Output = Self;
	fn add(self, rhs: u8) -> Self {
		todo!()
	}
}
impl Add<BigInt> for u8 {
	type Output = BigInt;
	fn add(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<u8> for BigInt {
	fn add_assign(&mut self, rhs: u8) {
		todo!()
	}
}
impl Sub<u8> for BigInt {
	type Output = Self;
	fn sub(self, rhs: u8) -> Self {
		todo!()
	}
}
impl Sub<BigInt> for u8 {
	type Output = BigInt;
	fn sub(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<u8> for BigInt {
	fn sub_assign(&mut self, rhs: u8) {
		todo!()
	}
}
impl Mul<u8> for BigInt {
	type Output = Self;
	fn mul(self, rhs: u8) -> Self {
		todo!()
	}
}
impl Mul<BigInt> for u8 {
	type Output = BigInt;
	fn mul(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<u8> for BigInt {
	fn mul_assign(&mut self, rhs: u8) {
		todo!()
	}
}
impl Div<u8> for BigInt {
	type Output = Self;
	fn div(self, rhs: u8) -> Self {
		todo!()
	}
}
impl Div<BigInt> for u8 {
	type Output = BigInt;
	fn div(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<u8> for BigInt {
	fn div_assign(&mut self, rhs: u8) {
		todo!()
	}
}
impl Rem<u8> for BigInt {
	type Output = Self;
	fn rem(self, rhs: u8) -> Self {
		todo!()
	}
}
impl Rem<BigInt> for u8 {
	type Output = BigInt;
	fn rem(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<u8> for BigInt {
	fn rem_assign(&mut self, rhs: u8) {
		todo!()
	}
}
impl PartialEq<u8> for &BigInt {
	fn eq(&self, other: &u8) -> bool {
		todo!()
	}
}
impl PartialEq<&BigInt> for u8 {
	fn eq(&self, other: &&BigInt) -> bool {
		todo!()
	}
}
impl Add<u8> for &BigInt {
	type Output = Self;
	fn add(self, rhs: u8) -> Self {
		todo!()
	}
}
impl Add<&BigInt> for u8 {
	type Output = BigInt;
	fn add(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<u8> for &BigInt {
	fn add_assign(&mut self, rhs: u8) {
		todo!()
	}
}
impl Sub<u8> for &BigInt {
	type Output = Self;
	fn sub(self, rhs: u8) -> Self {
		todo!()
	}
}
impl Sub<&BigInt> for u8 {
	type Output = BigInt;
	fn sub(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<u8> for &BigInt {
	fn sub_assign(&mut self, rhs: u8) {
		todo!()
	}
}
impl Mul<u8> for &BigInt {
	type Output = Self;
	fn mul(self, rhs: u8) -> Self {
		todo!()
	}
}
impl Mul<&BigInt> for u8 {
	type Output = BigInt;
	fn mul(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<u8> for &BigInt {
	fn mul_assign(&mut self, rhs: u8) {
		todo!()
	}
}
impl Div<u8> for &BigInt {
	type Output = Self;
	fn div(self, rhs: u8) -> Self {
		todo!()
	}
}
impl Div<&BigInt> for u8 {
	type Output = BigInt;
	fn div(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<u8> for &BigInt {
	fn div_assign(&mut self, rhs: u8) {
		todo!()
	}
}
impl Rem<u8> for &BigInt {
	type Output = Self;
	fn rem(self, rhs: u8) -> Self {
		todo!()
	}
}
impl Rem<&BigInt> for u8 {
	type Output = BigInt;
	fn rem(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<u8> for &BigInt {
	fn rem_assign(&mut self, rhs: u8) {
		todo!()
	}
}

impl PartialEq<i8> for BigInt {
	fn eq(&self, other: &i8) -> bool {
		todo!()
	}
}
impl PartialEq<BigInt> for i8 {
	fn eq(&self, other: &BigInt) -> bool {
		todo!()
	}
}
impl Add<i8> for BigInt {
	type Output = Self;
	fn add(self, rhs: i8) -> Self {
		todo!()
	}
}
impl Add<BigInt> for i8 {
	type Output = BigInt;
	fn add(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<i8> for BigInt {
	fn add_assign(&mut self, rhs: i8) {
		todo!()
	}
}
impl Sub<i8> for BigInt {
	type Output = Self;
	fn sub(self, rhs: i8) -> Self {
		todo!()
	}
}
impl Sub<BigInt> for i8 {
	type Output = BigInt;
	fn sub(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<i8> for BigInt {
	fn sub_assign(&mut self, rhs: i8) {
		todo!()
	}
}
impl Mul<i8> for BigInt {
	type Output = Self;
	fn mul(self, rhs: i8) -> Self {
		todo!()
	}
}
impl Mul<BigInt> for i8 {
	type Output = BigInt;
	fn mul(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<i8> for BigInt {
	fn mul_assign(&mut self, rhs: i8) {
		todo!()
	}
}
impl Div<i8> for BigInt {
	type Output = Self;
	fn div(self, rhs: i8) -> Self {
		todo!()
	}
}
impl Div<BigInt> for i8 {
	type Output = BigInt;
	fn div(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<i8> for BigInt {
	fn div_assign(&mut self, rhs: i8) {
		todo!()
	}
}
impl Rem<i8> for BigInt {
	type Output = Self;
	fn rem(self, rhs: i8) -> Self {
		todo!()
	}
}
impl Rem<BigInt> for i8 {
	type Output = BigInt;
	fn rem(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<i8> for BigInt {
	fn rem_assign(&mut self, rhs: i8) {
		todo!()
	}
}
impl PartialEq<i8> for &BigInt {
	fn eq(&self, other: &i8) -> bool {
		todo!()
	}
}
impl PartialEq<&BigInt> for i8 {
	fn eq(&self, other: &&BigInt) -> bool {
		todo!()
	}
}
impl Add<i8> for &BigInt {
	type Output = Self;
	fn add(self, rhs: i8) -> Self {
		todo!()
	}
}
impl Add<&BigInt> for i8 {
	type Output = BigInt;
	fn add(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<i8> for &BigInt {
	fn add_assign(&mut self, rhs: i8) {
		todo!()
	}
}
impl Sub<i8> for &BigInt {
	type Output = Self;
	fn sub(self, rhs: i8) -> Self {
		todo!()
	}
}
impl Sub<&BigInt> for i8 {
	type Output = BigInt;
	fn sub(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<i8> for &BigInt {
	fn sub_assign(&mut self, rhs: i8) {
		todo!()
	}
}
impl Mul<i8> for &BigInt {
	type Output = Self;
	fn mul(self, rhs: i8) -> Self {
		todo!()
	}
}
impl Mul<&BigInt> for i8 {
	type Output = BigInt;
	fn mul(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<i8> for &BigInt {
	fn mul_assign(&mut self, rhs: i8) {
		todo!()
	}
}
impl Div<i8> for &BigInt {
	type Output = Self;
	fn div(self, rhs: i8) -> Self {
		todo!()
	}
}
impl Div<&BigInt> for i8 {
	type Output = BigInt;
	fn div(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<i8> for &BigInt {
	fn div_assign(&mut self, rhs: i8) {
		todo!()
	}
}
impl Rem<i8> for &BigInt {
	type Output = Self;
	fn rem(self, rhs: i8) -> Self {
		todo!()
	}
}
impl Rem<&BigInt> for i8 {
	type Output = BigInt;
	fn rem(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<i8> for &BigInt {
	fn rem_assign(&mut self, rhs: i8) {
		todo!()
	}
}

impl PartialEq<u16> for BigInt {
	fn eq(&self, other: &u16) -> bool {
		todo!()
	}
}
impl PartialEq<BigInt> for u16 {
	fn eq(&self, other: &BigInt) -> bool {
		todo!()
	}
}
impl Add<u16> for BigInt {
	type Output = Self;
	fn add(self, rhs: u16) -> Self {
		todo!()
	}
}
impl Add<BigInt> for u16 {
	type Output = BigInt;
	fn add(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<u16> for BigInt {
	fn add_assign(&mut self, rhs: u16) {
		todo!()
	}
}
impl Sub<u16> for BigInt {
	type Output = Self;
	fn sub(self, rhs: u16) -> Self {
		todo!()
	}
}
impl Sub<BigInt> for u16 {
	type Output = BigInt;
	fn sub(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<u16> for BigInt {
	fn sub_assign(&mut self, rhs: u16) {
		todo!()
	}
}
impl Mul<u16> for BigInt {
	type Output = Self;
	fn mul(self, rhs: u16) -> Self {
		todo!()
	}
}
impl Mul<BigInt> for u16 {
	type Output = BigInt;
	fn mul(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<u16> for BigInt {
	fn mul_assign(&mut self, rhs: u16) {
		todo!()
	}
}
impl Div<u16> for BigInt {
	type Output = Self;
	fn div(self, rhs: u16) -> Self {
		todo!()
	}
}
impl Div<BigInt> for u16 {
	type Output = BigInt;
	fn div(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<u16> for BigInt {
	fn div_assign(&mut self, rhs: u16) {
		todo!()
	}
}
impl Rem<u16> for BigInt {
	type Output = Self;
	fn rem(self, rhs: u16) -> Self {
		todo!()
	}
}
impl Rem<BigInt> for u16 {
	type Output = BigInt;
	fn rem(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<u16> for BigInt {
	fn rem_assign(&mut self, rhs: u16) {
		todo!()
	}
}
impl PartialEq<u16> for &BigInt {
	fn eq(&self, other: &u16) -> bool {
		todo!()
	}
}
impl PartialEq<&BigInt> for u16 {
	fn eq(&self, other: &&BigInt) -> bool {
		todo!()
	}
}
impl Add<u16> for &BigInt {
	type Output = Self;
	fn add(self, rhs: u16) -> Self {
		todo!()
	}
}
impl Add<&BigInt> for u16 {
	type Output = BigInt;
	fn add(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<u16> for &BigInt {
	fn add_assign(&mut self, rhs: u16) {
		todo!()
	}
}
impl Sub<u16> for &BigInt {
	type Output = Self;
	fn sub(self, rhs: u16) -> Self {
		todo!()
	}
}
impl Sub<&BigInt> for u16 {
	type Output = BigInt;
	fn sub(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<u16> for &BigInt {
	fn sub_assign(&mut self, rhs: u16) {
		todo!()
	}
}
impl Mul<u16> for &BigInt {
	type Output = Self;
	fn mul(self, rhs: u16) -> Self {
		todo!()
	}
}
impl Mul<&BigInt> for u16 {
	type Output = BigInt;
	fn mul(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<u16> for &BigInt {
	fn mul_assign(&mut self, rhs: u16) {
		todo!()
	}
}
impl Div<u16> for &BigInt {
	type Output = Self;
	fn div(self, rhs: u16) -> Self {
		todo!()
	}
}
impl Div<&BigInt> for u16 {
	type Output = BigInt;
	fn div(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<u16> for &BigInt {
	fn div_assign(&mut self, rhs: u16) {
		todo!()
	}
}
impl Rem<u16> for &BigInt {
	type Output = Self;
	fn rem(self, rhs: u16) -> Self {
		todo!()
	}
}
impl Rem<&BigInt> for u16 {
	type Output = BigInt;
	fn rem(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<u16> for &BigInt {
	fn rem_assign(&mut self, rhs: u16) {
		todo!()
	}
}

impl PartialEq<i16> for BigInt {
	fn eq(&self, other: &i16) -> bool {
		todo!()
	}
}
impl PartialEq<BigInt> for i16 {
	fn eq(&self, other: &BigInt) -> bool {
		todo!()
	}
}
impl Add<i16> for BigInt {
	type Output = Self;
	fn add(self, rhs: i16) -> Self {
		todo!()
	}
}
impl Add<BigInt> for i16 {
	type Output = BigInt;
	fn add(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<i16> for BigInt {
	fn add_assign(&mut self, rhs: i16) {
		todo!()
	}
}
impl Sub<i16> for BigInt {
	type Output = Self;
	fn sub(self, rhs: i16) -> Self {
		todo!()
	}
}
impl Sub<BigInt> for i16 {
	type Output = BigInt;
	fn sub(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<i16> for BigInt {
	fn sub_assign(&mut self, rhs: i16) {
		todo!()
	}
}
impl Mul<i16> for BigInt {
	type Output = Self;
	fn mul(self, rhs: i16) -> Self {
		todo!()
	}
}
impl Mul<BigInt> for i16 {
	type Output = BigInt;
	fn mul(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<i16> for BigInt {
	fn mul_assign(&mut self, rhs: i16) {
		todo!()
	}
}
impl Div<i16> for BigInt {
	type Output = Self;
	fn div(self, rhs: i16) -> Self {
		todo!()
	}
}
impl Div<BigInt> for i16 {
	type Output = BigInt;
	fn div(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<i16> for BigInt {
	fn div_assign(&mut self, rhs: i16) {
		todo!()
	}
}
impl Rem<i16> for BigInt {
	type Output = Self;
	fn rem(self, rhs: i16) -> Self {
		todo!()
	}
}
impl Rem<BigInt> for i16 {
	type Output = BigInt;
	fn rem(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<i16> for BigInt {
	fn rem_assign(&mut self, rhs: i16) {
		todo!()
	}
}
impl PartialEq<i16> for &BigInt {
	fn eq(&self, other: &i16) -> bool {
		todo!()
	}
}
impl PartialEq<&BigInt> for i16 {
	fn eq(&self, other: &&BigInt) -> bool {
		todo!()
	}
}
impl Add<i16> for &BigInt {
	type Output = Self;
	fn add(self, rhs: i16) -> Self {
		todo!()
	}
}
impl Add<&BigInt> for i16 {
	type Output = BigInt;
	fn add(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<i16> for &BigInt {
	fn add_assign(&mut self, rhs: i16) {
		todo!()
	}
}
impl Sub<i16> for &BigInt {
	type Output = Self;
	fn sub(self, rhs: i16) -> Self {
		todo!()
	}
}
impl Sub<&BigInt> for i16 {
	type Output = BigInt;
	fn sub(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<i16> for &BigInt {
	fn sub_assign(&mut self, rhs: i16) {
		todo!()
	}
}
impl Mul<i16> for &BigInt {
	type Output = Self;
	fn mul(self, rhs: i16) -> Self {
		todo!()
	}
}
impl Mul<&BigInt> for i16 {
	type Output = BigInt;
	fn mul(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<i16> for &BigInt {
	fn mul_assign(&mut self, rhs: i16) {
		todo!()
	}
}
impl Div<i16> for &BigInt {
	type Output = Self;
	fn div(self, rhs: i16) -> Self {
		todo!()
	}
}
impl Div<&BigInt> for i16 {
	type Output = BigInt;
	fn div(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<i16> for &BigInt {
	fn div_assign(&mut self, rhs: i16) {
		todo!()
	}
}
impl Rem<i16> for &BigInt {
	type Output = Self;
	fn rem(self, rhs: i16) -> Self {
		todo!()
	}
}
impl Rem<&BigInt> for i16 {
	type Output = BigInt;
	fn rem(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<i16> for &BigInt {
	fn rem_assign(&mut self, rhs: i16) {
		todo!()
	}
}

impl PartialEq<u32> for BigInt {
	fn eq(&self, other: &u32) -> bool {
		todo!()
	}
}
impl PartialEq<BigInt> for u32 {
	fn eq(&self, other: &BigInt) -> bool {
		todo!()
	}
}
impl Add<u32> for BigInt {
	type Output = Self;
	fn add(self, rhs: u32) -> Self {
		todo!()
	}
}
impl Add<BigInt> for u32 {
	type Output = BigInt;
	fn add(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<u32> for BigInt {
	fn add_assign(&mut self, rhs: u32) {
		todo!()
	}
}
impl Sub<u32> for BigInt {
	type Output = Self;
	fn sub(self, rhs: u32) -> Self {
		todo!()
	}
}
impl Sub<BigInt> for u32 {
	type Output = BigInt;
	fn sub(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<u32> for BigInt {
	fn sub_assign(&mut self, rhs: u32) {
		todo!()
	}
}
impl Mul<u32> for BigInt {
	type Output = Self;
	fn mul(self, rhs: u32) -> Self {
		todo!()
	}
}
impl Mul<BigInt> for u32 {
	type Output = BigInt;
	fn mul(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<u32> for BigInt {
	fn mul_assign(&mut self, rhs: u32) {
		todo!()
	}
}
impl Div<u32> for BigInt {
	type Output = Self;
	fn div(self, rhs: u32) -> Self {
		todo!()
	}
}
impl Div<BigInt> for u32 {
	type Output = BigInt;
	fn div(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<u32> for BigInt {
	fn div_assign(&mut self, rhs: u32) {
		todo!()
	}
}
impl Rem<u32> for BigInt {
	type Output = Self;
	fn rem(self, rhs: u32) -> Self {
		todo!()
	}
}
impl Rem<BigInt> for u32 {
	type Output = BigInt;
	fn rem(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<u32> for BigInt {
	fn rem_assign(&mut self, rhs: u32) {
		todo!()
	}
}
impl PartialEq<u32> for &BigInt {
	fn eq(&self, other: &u32) -> bool {
		todo!()
	}
}
impl PartialEq<&BigInt> for u32 {
	fn eq(&self, other: &&BigInt) -> bool {
		todo!()
	}
}
impl Add<u32> for &BigInt {
	type Output = Self;
	fn add(self, rhs: u32) -> Self {
		todo!()
	}
}
impl Add<&BigInt> for u32 {
	type Output = BigInt;
	fn add(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<u32> for &BigInt {
	fn add_assign(&mut self, rhs: u32) {
		todo!()
	}
}
impl Sub<u32> for &BigInt {
	type Output = Self;
	fn sub(self, rhs: u32) -> Self {
		todo!()
	}
}
impl Sub<&BigInt> for u32 {
	type Output = BigInt;
	fn sub(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<u32> for &BigInt {
	fn sub_assign(&mut self, rhs: u32) {
		todo!()
	}
}
impl Mul<u32> for &BigInt {
	type Output = Self;
	fn mul(self, rhs: u32) -> Self {
		todo!()
	}
}
impl Mul<&BigInt> for u32 {
	type Output = BigInt;
	fn mul(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<u32> for &BigInt {
	fn mul_assign(&mut self, rhs: u32) {
		todo!()
	}
}
impl Div<u32> for &BigInt {
	type Output = Self;
	fn div(self, rhs: u32) -> Self {
		todo!()
	}
}
impl Div<&BigInt> for u32 {
	type Output = BigInt;
	fn div(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<u32> for &BigInt {
	fn div_assign(&mut self, rhs: u32) {
		todo!()
	}
}
impl Rem<u32> for &BigInt {
	type Output = Self;
	fn rem(self, rhs: u32) -> Self {
		todo!()
	}
}
impl Rem<&BigInt> for u32 {
	type Output = BigInt;
	fn rem(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<u32> for &BigInt {
	fn rem_assign(&mut self, rhs: u32) {
		todo!()
	}
}

impl PartialEq<i32> for BigInt {
	fn eq(&self, other: &i32) -> bool {
		todo!()
	}
}
impl PartialEq<BigInt> for i32 {
	fn eq(&self, other: &BigInt) -> bool {
		todo!()
	}
}
impl Add<i32> for BigInt {
	type Output = Self;
	fn add(self, rhs: i32) -> Self {
		todo!()
	}
}
impl Add<BigInt> for i32 {
	type Output = BigInt;
	fn add(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<i32> for BigInt {
	fn add_assign(&mut self, rhs: i32) {
		todo!()
	}
}
impl Sub<i32> for BigInt {
	type Output = Self;
	fn sub(self, rhs: i32) -> Self {
		todo!()
	}
}
impl Sub<BigInt> for i32 {
	type Output = BigInt;
	fn sub(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<i32> for BigInt {
	fn sub_assign(&mut self, rhs: i32) {
		// use std::backtrace::Backtrace;
		// println!("{}", Backtrace::force_capture());
		todo!()
	}
}
impl Mul<i32> for BigInt {
	type Output = Self;
	fn mul(self, rhs: i32) -> Self {
		return self.mul(BigInt::from_i32(rhs));
	}
}
impl Mul<BigInt> for i32 {
	type Output = BigInt;
	fn mul(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<i32> for BigInt {
	fn mul_assign(&mut self, rhs: i32) {
		todo!()
	}
}
impl Div<i32> for BigInt {
	type Output = Self;
	fn div(self, rhs: i32) -> Self {
		todo!()
	}
}
impl Div<BigInt> for i32 {
	type Output = BigInt;
	fn div(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<i32> for BigInt {
	fn div_assign(&mut self, rhs: i32) {
		todo!()
	}
}
impl Rem<i32> for BigInt {
	type Output = Self;
	fn rem(self, rhs: i32) -> Self {
		todo!()
	}
}
impl Rem<BigInt> for i32 {
	type Output = BigInt;
	fn rem(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<i32> for BigInt {
	fn rem_assign(&mut self, rhs: i32) {
		todo!()
	}
}
impl PartialEq<i32> for &BigInt {
	fn eq(&self, other: &i32) -> bool {
		todo!()
	}
}
impl PartialEq<&BigInt> for i32 {
	fn eq(&self, other: &&BigInt) -> bool {
		todo!()
	}
}
impl Add<i32> for &BigInt {
	type Output = Self;
	fn add(self, rhs: i32) -> Self {
		todo!()
	}
}
impl Add<&BigInt> for i32 {
	type Output = BigInt;
	fn add(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<i32> for &BigInt {
	fn add_assign(&mut self, rhs: i32) {
		todo!()
	}
}
impl Sub<i32> for &BigInt {
	type Output = Self;
	fn sub(self, rhs: i32) -> Self {
		todo!()
	}
}
impl Sub<&BigInt> for i32 {
	type Output = BigInt;
	fn sub(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<i32> for &BigInt {
	fn sub_assign(&mut self, rhs: i32) {
		todo!()
	}
}
impl Mul<i32> for &BigInt {
	type Output = Self;
	fn mul(self, rhs: i32) -> Self {
		todo!()
	}
}
impl Mul<&BigInt> for i32 {
	type Output = BigInt;
	fn mul(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<i32> for &BigInt {
	fn mul_assign(&mut self, rhs: i32) {
		todo!()
	}
}
impl Div<i32> for &BigInt {
	type Output = Self;
	fn div(self, rhs: i32) -> Self {
		todo!()
	}
}
impl Div<&BigInt> for i32 {
	type Output = BigInt;
	fn div(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<i32> for &BigInt {
	fn div_assign(&mut self, rhs: i32) {
		todo!()
	}
}
impl Rem<i32> for &BigInt {
	type Output = Self;
	fn rem(self, rhs: i32) -> Self {
		todo!()
	}
}
impl Rem<&BigInt> for i32 {
	type Output = BigInt;
	fn rem(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<i32> for &BigInt {
	fn rem_assign(&mut self, rhs: i32) {
		todo!()
	}
}

impl PartialEq<u64> for BigInt {
	fn eq(&self, other: &u64) -> bool {
		todo!()
	}
}
impl PartialEq<BigInt> for u64 {
	fn eq(&self, other: &BigInt) -> bool {
		todo!()
	}
}
impl Add<u64> for BigInt {
	type Output = Self;
	fn add(self, rhs: u64) -> Self {
		todo!()
	}
}
impl Add<BigInt> for u64 {
	type Output = BigInt;
	fn add(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<u64> for BigInt {
	fn add_assign(&mut self, rhs: u64) {
		todo!()
	}
}
impl Sub<u64> for BigInt {
	type Output = Self;
	fn sub(self, rhs: u64) -> Self {
		todo!()
	}
}
impl Sub<BigInt> for u64 {
	type Output = BigInt;
	fn sub(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<u64> for BigInt {
	fn sub_assign(&mut self, rhs: u64) {
		todo!()
	}
}
impl Mul<u64> for BigInt {
	type Output = Self;
	fn mul(self, rhs: u64) -> Self {
		todo!()
	}
}
impl Mul<BigInt> for u64 {
	type Output = BigInt;
	fn mul(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<u64> for BigInt {
	fn mul_assign(&mut self, rhs: u64) {
		todo!()
	}
}
impl Div<u64> for BigInt {
	type Output = Self;
	fn div(self, rhs: u64) -> Self {
		todo!()
	}
}
impl Div<BigInt> for u64 {
	type Output = BigInt;
	fn div(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<u64> for BigInt {
	fn div_assign(&mut self, rhs: u64) {
		todo!()
	}
}
impl Rem<u64> for BigInt {
	type Output = Self;
	fn rem(self, rhs: u64) -> Self {
		todo!()
	}
}
impl Rem<BigInt> for u64 {
	type Output = BigInt;
	fn rem(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<u64> for BigInt {
	fn rem_assign(&mut self, rhs: u64) {
		todo!()
	}
}
impl PartialEq<u64> for &BigInt {
	fn eq(&self, other: &u64) -> bool {
		todo!()
	}
}
impl PartialEq<&BigInt> for u64 {
	fn eq(&self, other: &&BigInt) -> bool {
		todo!()
	}
}
impl Add<u64> for &BigInt {
	type Output = Self;
	fn add(self, rhs: u64) -> Self {
		todo!()
	}
}
impl Add<&BigInt> for u64 {
	type Output = BigInt;
	fn add(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<u64> for &BigInt {
	fn add_assign(&mut self, rhs: u64) {
		todo!()
	}
}
impl Sub<u64> for &BigInt {
	type Output = Self;
	fn sub(self, rhs: u64) -> Self {
		todo!()
	}
}
impl Sub<&BigInt> for u64 {
	type Output = BigInt;
	fn sub(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<u64> for &BigInt {
	fn sub_assign(&mut self, rhs: u64) {
		todo!()
	}
}
impl Mul<u64> for &BigInt {
	type Output = Self;
	fn mul(self, rhs: u64) -> Self {
		todo!()
	}
}
impl Mul<&BigInt> for u64 {
	type Output = BigInt;
	fn mul(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<u64> for &BigInt {
	fn mul_assign(&mut self, rhs: u64) {
		todo!()
	}
}
impl Div<u64> for &BigInt {
	type Output = Self;
	fn div(self, rhs: u64) -> Self {
		todo!()
	}
}
impl Div<&BigInt> for u64 {
	type Output = BigInt;
	fn div(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<u64> for &BigInt {
	fn div_assign(&mut self, rhs: u64) {
		todo!()
	}
}
impl Rem<u64> for &BigInt {
	type Output = Self;
	fn rem(self, rhs: u64) -> Self {
		todo!()
	}
}
impl Rem<&BigInt> for u64 {
	type Output = BigInt;
	fn rem(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<u64> for &BigInt {
	fn rem_assign(&mut self, rhs: u64) {
		todo!()
	}
}

impl PartialEq<i64> for BigInt {
	fn eq(&self, other: &i64) -> bool {
		todo!()
	}
}
impl PartialEq<BigInt> for i64 {
	fn eq(&self, other: &BigInt) -> bool {
		todo!()
	}
}
impl Add<i64> for BigInt {
	type Output = Self;
	fn add(self, rhs: i64) -> Self {
		todo!()
	}
}
impl Add<BigInt> for i64 {
	type Output = BigInt;
	fn add(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<i64> for BigInt {
	fn add_assign(&mut self, rhs: i64) {
		todo!()
	}
}
impl Sub<i64> for BigInt {
	type Output = Self;
	fn sub(self, rhs: i64) -> Self {
		todo!()
	}
}
impl Sub<BigInt> for i64 {
	type Output = BigInt;
	fn sub(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<i64> for BigInt {
	fn sub_assign(&mut self, rhs: i64) {
		todo!()
	}
}
impl Mul<i64> for BigInt {
	type Output = Self;
	fn mul(self, rhs: i64) -> Self {
		todo!()
	}
}
impl Mul<BigInt> for i64 {
	type Output = BigInt;
	fn mul(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<i64> for BigInt {
	fn mul_assign(&mut self, rhs: i64) {
		todo!()
	}
}
impl Div<i64> for BigInt {
	type Output = Self;
	fn div(self, rhs: i64) -> Self {
		todo!()
	}
}
impl Div<BigInt> for i64 {
	type Output = BigInt;
	fn div(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<i64> for BigInt {
	fn div_assign(&mut self, rhs: i64) {
		todo!()
	}
}
impl Rem<i64> for BigInt {
	type Output = Self;
	fn rem(self, rhs: i64) -> Self {
		todo!()
	}
}
impl Rem<BigInt> for i64 {
	type Output = BigInt;
	fn rem(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<i64> for BigInt {
	fn rem_assign(&mut self, rhs: i64) {
		todo!()
	}
}
impl PartialEq<i64> for &BigInt {
	fn eq(&self, other: &i64) -> bool {
		todo!()
	}
}
impl PartialEq<&BigInt> for i64 {
	fn eq(&self, other: &&BigInt) -> bool {
		todo!()
	}
}
impl Add<i64> for &BigInt {
	type Output = Self;
	fn add(self, rhs: i64) -> Self {
		todo!()
	}
}
impl Add<&BigInt> for i64 {
	type Output = BigInt;
	fn add(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<i64> for &BigInt {
	fn add_assign(&mut self, rhs: i64) {
		todo!()
	}
}
impl Sub<i64> for &BigInt {
	type Output = Self;
	fn sub(self, rhs: i64) -> Self {
		todo!()
	}
}
impl Sub<&BigInt> for i64 {
	type Output = BigInt;
	fn sub(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<i64> for &BigInt {
	fn sub_assign(&mut self, rhs: i64) {
		todo!()
	}
}
impl Mul<i64> for &BigInt {
	type Output = Self;
	fn mul(self, rhs: i64) -> Self {
		todo!()
	}
}
impl Mul<&BigInt> for i64 {
	type Output = BigInt;
	fn mul(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<i64> for &BigInt {
	fn mul_assign(&mut self, rhs: i64) {
		todo!()
	}
}
impl Div<i64> for &BigInt {
	type Output = Self;
	fn div(self, rhs: i64) -> Self {
		todo!()
	}
}
impl Div<&BigInt> for i64 {
	type Output = BigInt;
	fn div(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<i64> for &BigInt {
	fn div_assign(&mut self, rhs: i64) {
		todo!()
	}
}
impl Rem<i64> for &BigInt {
	type Output = Self;
	fn rem(self, rhs: i64) -> Self {
		todo!()
	}
}
impl Rem<&BigInt> for i64 {
	type Output = BigInt;
	fn rem(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<i64> for &BigInt {
	fn rem_assign(&mut self, rhs: i64) {
		todo!()
	}
}

impl PartialEq<u128> for BigInt {
	fn eq(&self, other: &u128) -> bool {
		todo!()
	}
}
impl PartialEq<BigInt> for u128 {
	fn eq(&self, other: &BigInt) -> bool {
		todo!()
	}
}
impl Add<u128> for BigInt {
	type Output = Self;
	fn add(self, rhs: u128) -> Self {
		todo!()
	}
}
impl Add<BigInt> for u128 {
	type Output = BigInt;
	fn add(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<u128> for BigInt {
	fn add_assign(&mut self, rhs: u128) {
		todo!()
	}
}
impl Sub<u128> for BigInt {
	type Output = Self;
	fn sub(self, rhs: u128) -> Self {
		todo!()
	}
}
impl Sub<BigInt> for u128 {
	type Output = BigInt;
	fn sub(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<u128> for BigInt {
	fn sub_assign(&mut self, rhs: u128) {
		todo!()
	}
}
impl Mul<u128> for BigInt {
	type Output = Self;
	fn mul(self, rhs: u128) -> Self {
		todo!()
	}
}
impl Mul<BigInt> for u128 {
	type Output = BigInt;
	fn mul(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<u128> for BigInt {
	fn mul_assign(&mut self, rhs: u128) {
		todo!()
	}
}
impl Div<u128> for BigInt {
	type Output = Self;
	fn div(self, rhs: u128) -> Self {
		todo!()
	}
}
impl Div<BigInt> for u128 {
	type Output = BigInt;
	fn div(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<u128> for BigInt {
	fn div_assign(&mut self, rhs: u128) {
		todo!()
	}
}
impl Rem<u128> for BigInt {
	type Output = Self;
	fn rem(self, rhs: u128) -> Self {
		todo!()
	}
}
impl Rem<BigInt> for u128 {
	type Output = BigInt;
	fn rem(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<u128> for BigInt {
	fn rem_assign(&mut self, rhs: u128) {
		todo!()
	}
}
impl PartialEq<u128> for &BigInt {
	fn eq(&self, other: &u128) -> bool {
		todo!()
	}
}
impl PartialEq<&BigInt> for u128 {
	fn eq(&self, other: &&BigInt) -> bool {
		todo!()
	}
}
impl Add<u128> for &BigInt {
	type Output = Self;
	fn add(self, rhs: u128) -> Self {
		todo!()
	}
}
impl Add<&BigInt> for u128 {
	type Output = BigInt;
	fn add(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<u128> for &BigInt {
	fn add_assign(&mut self, rhs: u128) {
		todo!()
	}
}
impl Sub<u128> for &BigInt {
	type Output = Self;
	fn sub(self, rhs: u128) -> Self {
		todo!()
	}
}
impl Sub<&BigInt> for u128 {
	type Output = BigInt;
	fn sub(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<u128> for &BigInt {
	fn sub_assign(&mut self, rhs: u128) {
		todo!()
	}
}
impl Mul<u128> for &BigInt {
	type Output = Self;
	fn mul(self, rhs: u128) -> Self {
		todo!()
	}
}
impl Mul<&BigInt> for u128 {
	type Output = BigInt;
	fn mul(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<u128> for &BigInt {
	fn mul_assign(&mut self, rhs: u128) {
		todo!()
	}
}
impl Div<u128> for &BigInt {
	type Output = Self;
	fn div(self, rhs: u128) -> Self {
		todo!()
	}
}
impl Div<&BigInt> for u128 {
	type Output = BigInt;
	fn div(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<u128> for &BigInt {
	fn div_assign(&mut self, rhs: u128) {
		todo!()
	}
}
impl Rem<u128> for &BigInt {
	type Output = Self;
	fn rem(self, rhs: u128) -> Self {
		todo!()
	}
}
impl Rem<&BigInt> for u128 {
	type Output = BigInt;
	fn rem(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<u128> for &BigInt {
	fn rem_assign(&mut self, rhs: u128) {
		todo!()
	}
}

impl PartialEq<i128> for BigInt {
	fn eq(&self, other: &i128) -> bool {
		todo!()
	}
}
impl PartialEq<BigInt> for i128 {
	fn eq(&self, other: &BigInt) -> bool {
		todo!()
	}
}
impl Add<i128> for BigInt {
	type Output = Self;
	fn add(self, rhs: i128) -> Self {
		todo!()
	}
}
impl Add<BigInt> for i128 {
	type Output = BigInt;
	fn add(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<i128> for BigInt {
	fn add_assign(&mut self, rhs: i128) {
		todo!()
	}
}
impl Sub<i128> for BigInt {
	type Output = Self;
	fn sub(self, rhs: i128) -> Self {
		todo!()
	}
}
impl Sub<BigInt> for i128 {
	type Output = BigInt;
	fn sub(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<i128> for BigInt {
	fn sub_assign(&mut self, rhs: i128) {
		todo!()
	}
}
impl Mul<i128> for BigInt {
	type Output = Self;
	fn mul(self, rhs: i128) -> Self {
		todo!()
	}
}
impl Mul<BigInt> for i128 {
	type Output = BigInt;
	fn mul(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<i128> for BigInt {
	fn mul_assign(&mut self, rhs: i128) {
		todo!()
	}
}
impl Div<i128> for BigInt {
	type Output = Self;
	fn div(self, rhs: i128) -> Self {
		todo!()
	}
}
impl Div<BigInt> for i128 {
	type Output = BigInt;
	fn div(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<i128> for BigInt {
	fn div_assign(&mut self, rhs: i128) {
		todo!()
	}
}
impl Rem<i128> for BigInt {
	type Output = Self;
	fn rem(self, rhs: i128) -> Self {
		todo!()
	}
}
impl Rem<BigInt> for i128 {
	type Output = BigInt;
	fn rem(self, rhs: BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<i128> for BigInt {
	fn rem_assign(&mut self, rhs: i128) {
		todo!()
	}
}
impl PartialEq<i128> for &BigInt {
	fn eq(&self, other: &i128) -> bool {
		todo!()
	}
}
impl PartialEq<&BigInt> for i128 {
	fn eq(&self, other: &&BigInt) -> bool {
		todo!()
	}
}
impl Add<i128> for &BigInt {
	type Output = Self;
	fn add(self, rhs: i128) -> Self {
		todo!()
	}
}
impl Add<&BigInt> for i128 {
	type Output = BigInt;
	fn add(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl AddAssign<i128> for &BigInt {
	fn add_assign(&mut self, rhs: i128) {
		todo!()
	}
}
impl Sub<i128> for &BigInt {
	type Output = Self;
	fn sub(self, rhs: i128) -> Self {
		todo!()
	}
}
impl Sub<&BigInt> for i128 {
	type Output = BigInt;
	fn sub(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl SubAssign<i128> for &BigInt {
	fn sub_assign(&mut self, rhs: i128) {
		todo!()
	}
}
impl Mul<i128> for &BigInt {
	type Output = Self;
	fn mul(self, rhs: i128) -> Self {
		todo!()
	}
}
impl Mul<&BigInt> for i128 {
	type Output = BigInt;
	fn mul(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl MulAssign<i128> for &BigInt {
	fn mul_assign(&mut self, rhs: i128) {
		todo!()
	}
}
impl Div<i128> for &BigInt {
	type Output = Self;
	fn div(self, rhs: i128) -> Self {
		todo!()
	}
}
impl Div<&BigInt> for i128 {
	type Output = BigInt;
	fn div(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl DivAssign<i128> for &BigInt {
	fn div_assign(&mut self, rhs: i128) {
		todo!()
	}
}
impl Rem<i128> for &BigInt {
	type Output = Self;
	fn rem(self, rhs: i128) -> Self {
		todo!()
	}
}
impl Rem<&BigInt> for i128 {
	type Output = BigInt;
	fn rem(self, rhs: &BigInt) -> BigInt {
		todo!()
	}
}
impl RemAssign<i128> for &BigInt {
	fn rem_assign(&mut self, rhs: i128) {
		todo!()
	}
}


// END OF NATIVE INT IMPLEMENTATIONS //

pub mod errors {
	use std::fmt::{Display, Formatter};

	#[derive(Debug, PartialEq)]
	pub struct ParseError {
		invalid_input: String,
		type_name: String
	}
	impl ParseError{
		pub fn new(invalid_input: &str, type_name: &str) -> ParseError{
			return ParseError{invalid_input:String::from(invalid_input),
				type_name:String::from(type_name)};
		}
	}

	impl Display for ParseError{
		fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
			f.write_str(
				format!("ParseError: Could not parse '{}' as {}",
				self.invalid_input, self.type_name).as_ref()
			)
		}
	}

	#[derive(Debug, PartialEq)]
	pub struct InvalidConversionError {
		source_name: String,
		dest_name: String,
		explain: String
	}
	impl InvalidConversionError{
		pub fn new(source_name: &str, dest_name: &str, explain: &str) -> InvalidConversionError{
			return InvalidConversionError{source_name:String::from(source_name),
				dest_name:String::from(dest_name), explain:String::from(explain)};
		}
	}

	impl Display for InvalidConversionError{
		fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
			f.write_str(
				format!("InvalidConversionError: Could not convert {} to {} {}",
						self.source_name, self.dest_name, self.explain).as_ref()
			)
		}
	}


	#[derive(Debug, PartialEq)]
	pub struct MathError {
		msg: String
	}
	impl MathError{
		pub fn new(msg: &str) -> MathError{
			return MathError{msg:String::from(msg)};
		}
	}

	impl Display for MathError{
		fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
			f.write_str(
				format!("MathError: {}", self.msg).as_ref()
			)
		}
	}

}

#[cfg(test)]
mod tests {
	use crate::*;
	use std::collections::HashMap;

	#[test]
	fn bigint_badparse() {
		let result = BigInt::from_str("1234567890A");
		println!("{}", result.unwrap_err());
		let result = BigInt::from_str("1234.567890");
		println!("{}", result.unwrap_err());
		let result = BigInt::from_str(" ");
		println!("{}", result.unwrap_err());
		let result = BigInt::from_str("\t");
		println!("{}", result.unwrap_err());
		let result = BigInt::from_str("\n");
		println!("{}", result.unwrap_err());
		let result = BigInt::from_str(",1234");
		println!("{}", result.unwrap_err());
		let result = BigInt::from_str(" 1234567890");
		println!("{}", result.unwrap()); // should NOT error
		let result = BigInt::from_str(" -1234567890");
		println!("{}", result.unwrap()); // should NOT error
		let result = BigInt::from_str("\t1234567890");
		println!("{}", result.unwrap()); // should NOT error
		let result = BigInt::from_str("1234567890 ");
		println!("{}", result.unwrap()); // should NOT error
		let result = BigInt::from_str("01234567890 ");
		println!("{}", result.unwrap()); // should NOT error
		let result = BigInt::from_str("1234567890\t");
		println!("{}", result.unwrap()); // should NOT error
		let result = BigInt::from_str(" 1234567890\n");
		println!("{}", result.unwrap()); // should NOT error
		let result = BigInt::from_str(" 1234567890\r\n");
		println!("{}", result.unwrap()); // should NOT error
	}

	#[test]
	fn long_division_test_1() {
		println!("testing long integer division...");
		// remember, index 0 is ones' digit (least significant digit first, aka little-endian)
		let (q, r) = BigInt::simple_division_64bit(&vec![1,2,3,4,6], &vec![3,0,3]);
		assert_eq!(q, 64321i64 / 303i64);
		assert_eq!(r, 64321i64 % 303i64);
		let (q, r) = BigInt::simple_division_64bit(&vec![1,0,0,0,2,3,2,2,2,5,9,9,5,4,3,1,6], &vec![7,1,5,4,3,2,1,6,7,3,0,3]);
		assert_eq!(q, 61345995222320001i64 / 303761234517i64);
		assert_eq!(r, 61345995222320001i64 % 303761234517i64);
		// n = 2011175439743600021634000000000000000000000000000586423463
		let n = vec![3,6,4,3,2,4,6,8,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,3,6,1,2,0,0,0,6,3,4,7,9,3,4,5,7,1,1,1,0,2];
		let (mut q, mut r) = BigInt::burnikel_ziegler_division(&n, &vec![7,1]).unwrap();
		BigInt::trim_leading_zeroes(&mut q);BigInt::trim_leading_zeroes(&mut r);
		assert_eq!(q, vec![3,3,7,3,8,0,5,0,5,6,7,1,1,4,9,2,5,3,2,8,8,5,0,7,4,6,7,1,1,4,9,2,5,3,2,8,0,6,8,1,7,4,6,7,9,1,3,6,7,3,4,4,0,3,8,1,1]);
		// divide by 17 = 118304437631976471860823529411764705882352941176505083733
		assert_eq!(r, vec![2]); // mod by 17 = 2
		let (mut q, mut r) = BigInt::burnikel_ziegler_division(&n, &vec![2,1,4,9,3,0,0,0,0,2,9,0,4,8,3,7,6,9,4,3,7,1]).unwrap();
		BigInt::trim_leading_zeroes(&mut q);BigInt::trim_leading_zeroes(&mut r);
		assert_eq!(q,
				   vec![2,9,1,0,1,6,8,0,7,4,2,9,0,0,0,8,4,7,0,4,7,1,1,9,1,8,3,9,1,7,0,0,2,9,5,1,1]);
		// divide by 1734967384092000039412 = 1159200719381911740748000924708610192
		assert_eq!(r,
				   vec![9,5,3,6,3,5,1,4,8,0,2,7,9,8,6,7,1,6,6,7,4,1]);
		// mod by 1734967384092000039412 = 1476617689720841536359
	}

	#[test]
	fn bigint_operator_test() {
		println!("testing BigInt operators...");
		let i1 = BigInt::from_str("1234567890987654321").unwrap();
		let i2 = BigInt::from_str("0123456780876543210").unwrap();
		println!("BigInt comparisons");
		assert!(i1 > i2);
		assert!(i2 < i1);
		assert_ne!(i1, i2);
		assert_ne!(i1, BigInt::from_str("-1234567890987654321").unwrap()); // sign check
		println!("BigInt operators:");
		println!("+");
		assert_eq!(i1.clone() + i2.clone(), BigInt::from_str("1358024671864197531").unwrap());
		assert_eq!(&i1 + &i2, BigInt::from_str("1358024671864197531").unwrap());
		println!("-");
		assert_eq!(i1.clone() - i2.clone(), BigInt::from_str("1111111110111111111").unwrap());
		assert_eq!(&i1 - &i2, BigInt::from_str("1111111110111111111").unwrap());
		println!("*");
		assert_eq!(i1.clone() * i2.clone(), BigInt::from_str("152415777594878924352994968899710410").unwrap());
		assert_eq!(&i1 * &i2, BigInt::from_str("152415777594878924352994968899710410").unwrap());
		println!("/");
		assert_eq!(i1.clone() / i2.clone(), BigInt::from_str("10").unwrap());
		assert_eq!(&i1 / &i2, BigInt::from_str("10").unwrap()); // remember, it's integer division
		println!("%");
		assert_eq!(i1.clone() % i2.clone(), BigInt::from_str("82222222221").unwrap());
		assert_eq!(&i1 % &i2, BigInt::from_str("82222222221").unwrap());
		assert_eq!((i1.clone()/BigInt::from_i32(101)).to_string(), "12223444465224300");
		assert_eq!((i1.clone() % BigInt::from_i32(101)).to_string(), "21");
		println!("pow");
		assert_eq!(BigInt::from_i32(101).pow(11).to_string(),"11156683466653165551101");
		println!("== to i32 and i64");
		assert_eq!(BigInt::from_i32(101), 101i32);
		assert_ne!(BigInt::from_i32(100), 101i32);
		assert_eq!(BigInt::from_i64(9223372036854775807i64), 9223372036854775807i64);
	}

	#[test]
	fn bigint_num_math(){
		assert_eq!(BigInt::from_i32(17) * 13, BigInt::from_i32(17 * 13));
		assert_eq!(BigInt::from_i32(-17) * 13, BigInt::from_i32(-17 * 13));
		assert_eq!(BigInt::from_i32(256) / 8, BigInt::from_i32(256 / 8));
		assert_eq!(BigInt::from_i32(256) / 7, BigInt::from_i32(256 / 7));
		assert_eq!(BigInt::from_i32(256) % 7, BigInt::from_i32(256 % 7));
		assert_eq!(BigInt::from_i32(256) + 7, BigInt::from_i32(256 + 7));
		assert_eq!(BigInt::from_i32(7) - 256, BigInt::from_i32(7 - 256));
		assert_eq!(BigInt::from_i32(127) + 13u8, BigInt::from_i32(127 + 13));
		assert_eq!(127u8 + BigInt::from_i32(13), BigInt::from_i32(127 + 13));
		assert_eq!(BigInt::from_i32(127) - 13u8, BigInt::from_i32(127 - 13));
		assert_eq!(127u8 - BigInt::from_i32(13), BigInt::from_i32(127 - 13));
		assert_eq!(BigInt::from_i32(17) * 13u8, BigInt::from_i32(17 * 13));
		assert_eq!(127u8 * BigInt::from_i32(13), BigInt::from_i32(127 * 13));
		assert_eq!(BigInt::from_i32(127) / 13u8, BigInt::from_i32(127 / 13));
		assert_eq!(127u8 / BigInt::from_i32(13), BigInt::from_i32(127 / 13));
		assert_eq!(BigInt::from_i32(127) % 13u8, BigInt::from_i32(127 % 13));
		assert_eq!(127u8 % BigInt::from_i32(13), BigInt::from_i32(127 % 13));
		assert_eq!(BigInt::from_i32(127) + 13i8, BigInt::from_i32(127 + 13));
		assert_eq!(127i8 + BigInt::from_i32(13), BigInt::from_i32(127 + 13));
		assert_eq!(BigInt::from_i32(127) - 13i8, BigInt::from_i32(127 - 13));
		assert_eq!(127i8 - BigInt::from_i32(13), BigInt::from_i32(127 - 13));
		assert_eq!(BigInt::from_i32(17) * 13i8, BigInt::from_i32(17 * 13));
		assert_eq!(127i8 * BigInt::from_i32(13), BigInt::from_i32(127 * 13));
		assert_eq!(BigInt::from_i32(127) / 13i8, BigInt::from_i32(127 / 13));
		assert_eq!(127i8 / BigInt::from_i32(13), BigInt::from_i32(127 / 13));
		assert_eq!(BigInt::from_i32(127) % 13i8, BigInt::from_i32(127 % 13));
		assert_eq!(127i8 % BigInt::from_i32(13), BigInt::from_i32(127 % 13));
		assert_eq!(BigInt::from_i32(127) + 13u16, BigInt::from_i32(127 + 13));
		assert_eq!(127u16 + BigInt::from_i32(13), BigInt::from_i32(127 + 13));
		assert_eq!(BigInt::from_i32(127) - 13u16, BigInt::from_i32(127 - 13));
		assert_eq!(127u16 - BigInt::from_i32(13), BigInt::from_i32(127 - 13));
		assert_eq!(BigInt::from_i32(127) * 13u16, BigInt::from_i32(127 * 13));
		assert_eq!(127u16 * BigInt::from_i32(13), BigInt::from_i32(127 * 13));
		assert_eq!(BigInt::from_i32(127) / 13u16, BigInt::from_i32(127 / 13));
		assert_eq!(127u16 / BigInt::from_i32(13), BigInt::from_i32(127 / 13));
		assert_eq!(BigInt::from_i32(127) % 13u16, BigInt::from_i32(127 % 13));
		assert_eq!(127u16 % BigInt::from_i32(13), BigInt::from_i32(127 % 13));
		assert_eq!(BigInt::from_i32(127) + 13i16, BigInt::from_i32(127 + 13));
		assert_eq!(127i16 + BigInt::from_i32(13), BigInt::from_i32(127 + 13));
		assert_eq!(BigInt::from_i32(127) - 13i16, BigInt::from_i32(127 - 13));
		assert_eq!(127i16 - BigInt::from_i32(13), BigInt::from_i32(127 - 13));
		assert_eq!(BigInt::from_i32(127) * 13i16, BigInt::from_i32(127 * 13));
		assert_eq!(127i16 * BigInt::from_i32(13), BigInt::from_i32(127 * 13));
		assert_eq!(BigInt::from_i32(127) / 13i16, BigInt::from_i32(127 / 13));
		assert_eq!(127i16 / BigInt::from_i32(13), BigInt::from_i32(127 / 13));
		assert_eq!(BigInt::from_i32(127) % 13i16, BigInt::from_i32(127 % 13));
		assert_eq!(127i16 % BigInt::from_i32(13), BigInt::from_i32(127 % 13));
		assert_eq!(BigInt::from_i32(127) + 13u32, BigInt::from_i32(127 + 13));
		assert_eq!(127u32 + BigInt::from_i32(13), BigInt::from_i32(127 + 13));
		assert_eq!(BigInt::from_i32(127) - 13u32, BigInt::from_i32(127 - 13));
		assert_eq!(127u32 - BigInt::from_i32(13), BigInt::from_i32(127 - 13));
		assert_eq!(BigInt::from_i32(127) * 13u32, BigInt::from_i32(127 * 13));
		assert_eq!(127u32 * BigInt::from_i32(13), BigInt::from_i32(127 * 13));
		assert_eq!(BigInt::from_i32(127) / 13u32, BigInt::from_i32(127 / 13));
		assert_eq!(127u32 / BigInt::from_i32(13), BigInt::from_i32(127 / 13));
		assert_eq!(BigInt::from_i32(127) % 13u32, BigInt::from_i32(127 % 13));
		assert_eq!(127u32 % BigInt::from_i32(13), BigInt::from_i32(127 % 13));
		assert_eq!(BigInt::from_i32(127) + 13i32, BigInt::from_i32(127 + 13));
		assert_eq!(127i32 + BigInt::from_i32(13), BigInt::from_i32(127 + 13));
		assert_eq!(BigInt::from_i32(127) - 13i32, BigInt::from_i32(127 - 13));
		assert_eq!(127i32 - BigInt::from_i32(13), BigInt::from_i32(127 - 13));
		assert_eq!(BigInt::from_i32(127) * 13i32, BigInt::from_i32(127 * 13));
		assert_eq!(127i32 * BigInt::from_i32(13), BigInt::from_i32(127 * 13));
		assert_eq!(BigInt::from_i32(127) / 13i32, BigInt::from_i32(127 / 13));
		assert_eq!(127i32 / BigInt::from_i32(13), BigInt::from_i32(127 / 13));
		assert_eq!(BigInt::from_i32(127) % 13i32, BigInt::from_i32(127 % 13));
		assert_eq!(127i32 % BigInt::from_i32(13), BigInt::from_i32(127 % 13));
		assert_eq!(BigInt::from_i32(127) + 13u64, BigInt::from_i32(127 + 13));
		assert_eq!(127u64 + BigInt::from_i32(13), BigInt::from_i32(127 + 13));
		assert_eq!(BigInt::from_i32(127) - 13u64, BigInt::from_i32(127 - 13));
		assert_eq!(127u64 - BigInt::from_i32(13), BigInt::from_i32(127 - 13));
		assert_eq!(BigInt::from_i32(127) * 13u64, BigInt::from_i32(127 * 13));
		assert_eq!(127u64 * BigInt::from_i32(13), BigInt::from_i32(127 * 13));
		assert_eq!(BigInt::from_i32(127) / 13u64, BigInt::from_i32(127 / 13));
		assert_eq!(127u64 / BigInt::from_i32(13), BigInt::from_i32(127 / 13));
		assert_eq!(BigInt::from_i32(127) % 13u64, BigInt::from_i32(127 % 13));
		assert_eq!(127u64 % BigInt::from_i32(13), BigInt::from_i32(127 % 13));
		assert_eq!(BigInt::from_i32(127) + 13i64, BigInt::from_i32(127 + 13));
		assert_eq!(127i64 + BigInt::from_i32(13), BigInt::from_i32(127 + 13));
		assert_eq!(BigInt::from_i32(127) - 13i64, BigInt::from_i32(127 - 13));
		assert_eq!(127i64 - BigInt::from_i32(13), BigInt::from_i32(127 - 13));
		assert_eq!(BigInt::from_i32(127) * 13i64, BigInt::from_i32(127 * 13));
		assert_eq!(127i64 * BigInt::from_i32(13), BigInt::from_i32(127 * 13));
		assert_eq!(BigInt::from_i32(127) / 13i64, BigInt::from_i32(127 / 13));
		assert_eq!(127i64 / BigInt::from_i32(13), BigInt::from_i32(127 / 13));
		assert_eq!(BigInt::from_i32(127) % 13i64, BigInt::from_i32(127 % 13));
		assert_eq!(127i64 % BigInt::from_i32(13), BigInt::from_i32(127 % 13));
		assert_eq!(BigInt::from_i32(127) + 13u128, BigInt::from_i32(127 + 13));
		assert_eq!(127u128 + BigInt::from_i32(13), BigInt::from_i32(127 + 13));
		assert_eq!(BigInt::from_i32(127) - 13u128, BigInt::from_i32(127 - 13));
		assert_eq!(127u128 - BigInt::from_i32(13), BigInt::from_i32(127 - 13));
		assert_eq!(BigInt::from_i32(127) * 13u128, BigInt::from_i32(127 * 13));
		assert_eq!(127u128 * BigInt::from_i32(13), BigInt::from_i32(127 * 13));
		assert_eq!(BigInt::from_i32(127) / 13u128, BigInt::from_i32(127 / 13));
		assert_eq!(127u128 / BigInt::from_i32(13), BigInt::from_i32(127 / 13));
		assert_eq!(BigInt::from_i32(127) % 13u128, BigInt::from_i32(127 % 13));
		assert_eq!(127u128 % BigInt::from_i32(13), BigInt::from_i32(127 % 13));
		assert_eq!(BigInt::from_i32(127) + 13i128, BigInt::from_i32(127 + 13));
		assert_eq!(127i128 + BigInt::from_i32(13), BigInt::from_i32(127 + 13));
		assert_eq!(BigInt::from_i32(127) - 13i128, BigInt::from_i32(127 - 13));
		assert_eq!(127i128 - BigInt::from_i32(13), BigInt::from_i32(127 - 13));
		assert_eq!(BigInt::from_i32(127) * 13i128, BigInt::from_i32(127 * 13));
		assert_eq!(127i128 * BigInt::from_i32(13), BigInt::from_i32(127 * 13));
		assert_eq!(BigInt::from_i32(127) / 13i128, BigInt::from_i32(127 / 13));
		assert_eq!(127i128 / BigInt::from_i32(13), BigInt::from_i32(127 / 13));
		assert_eq!(BigInt::from_i32(127) % 13i128, BigInt::from_i32(127 % 13));
		assert_eq!(127i128 % BigInt::from_i32(13), BigInt::from_i32(127 % 13));
	}
	#[test]
	fn bigint_containers() {
		let mut list = vec![BigInt::from_i32(-33), BigInt::from_i32(111), BigInt::from_i32(-1),
							BigInt::from_i32(5), BigInt::from_i32(0), BigInt::from_i32(2)];
		list.sort();
		assert_eq!(list, vec![BigInt::from_i32(-33), BigInt::from_i32(-1), BigInt::from_i32(0),
							BigInt::from_i32(2), BigInt::from_i32(5), BigInt::from_i32(111)]);
		let mut map: HashMap<BigInt, BigInt> = HashMap::new();
		map.insert(BigInt::from_i32(5), BigInt::from_i32(55555));
		assert_eq!(map.get(&BigInt::from_i32(5)).unwrap().to_string(), "55555");
		assert!(map.get(&BigInt::from_i32(6)).is_none());
		assert!(map.get(&BigInt::from_i32(-5)).is_none());
	}
	#[test]
	fn bigint_trim_leading_zeroes(){
		let mut before: Vec<u8> = vec![0,1,2,3,4,5,0,0,0];
		let after: Vec<u8> = vec![0,1,2,3,4,5];
		BigInt::trim_leading_zeroes(&mut before);
		assert_eq!(before, after, "trim_leading_zeroes failure: {:?} not equal to {:?}", before, after);
	}
}

