use std::fmt::{Debug, Formatter, Display};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::*;
use std::borrow::Borrow;

#[derive(Clone, Eq, PartialOrd, Debug)]
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
		let mut i = self.digits.len()-1;
		while i > 0 && self.digits[i] != 0u8 {
			i -= 1;
		}
		return &self.digits[0..i];
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
		return BigInt::from_localized_str(s, '.', vec![',', '_']);
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
		return Self::from_i64(i as i64);
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
		let unsigned = Self::from_u64(i.abs() as u64);
		return BigInt{digits: unsigned.digits, positive: i >= 0}
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
		let unsigned = Self::from_u128(i.abs() as u128);
		return BigInt{digits: unsigned.digits, positive: i >= 0}
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
		let mut text = String::with_capacity(self.digits.len()+1);
		if !self.positive{
			text.push('-');
		}
		for i in 0..self.digits.len() {
			let reverse_index = self.digits.len() - i - 1;
			text.push(std::char::from_digit(self.digits[reverse_index] as u32, 10)
				.ok_or(std::fmt::Error)?);
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

impl Ord for BigInt {
	fn cmp(&self, other: &Self) -> Ordering {
		if self.is_zero() && other.is_zero(){
			return Ordering::Equal;
		}
		let s1 = self.get_nonzero_slice();
		let s2 = other.get_nonzero_slice();
		let l1 = s1.len();
		let l2= s2.len();
		let len_cmp = l1.cmp(&l2);
		match len_cmp {
			Ordering::Equal => {
				for i in (0..l1).rev() {
					if s1[i] != s2[i]{
						return s1.cmp(&s2);
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
		return BigInt{digits: self.digits.clone(), positive: !self.positive};
	}
}
impl Add for BigInt {
	type Output = Self;
	fn add(self, rhs: Self) -> Self {
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
			if a >= b {
				let delta = a.sub(b);
				return delta.convert_sign(self.positive);
			} else {
				let delta = b.sub(a);
				return delta.convert_sign(rhs.positive);
			}
		}
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
		return self.add(rhs.neg());
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
		let sign = self.positive == rhs.positive;
		let count = self.digits.len().max(rhs.digits.len());
		let mut new_digits: Vec<u8> = Vec::with_capacity(count+1);
		let mut i = 0;
		let mut carry: u8 = 0;
		while i < count {
			let a = self.digits.get(i).unwrap_or(&1u8);
			let b = rhs.digits.get(i).unwrap_or(&1u8);
			let product = a * b + carry;
			new_digits.push(product % 10);
			carry = product / 10;
			i += 1;
		}
		if carry != 0 {
			new_digits.push(carry);
		}
		return BigInt{digits: new_digits, positive: sign};
	}
}
impl MulAssign for BigInt {
	fn mul_assign(&mut self, rhs: Self) {
		let tmp = self.clone().mul(rhs);
		self.digits = tmp.digits.clone();
		self.positive = tmp.positive;
	}
}

impl BigInt {
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
		let positive = a.positive == b.positive;
		let mut a_digits = a.digits.clone();
		let mut b_digits = b.digits.clone();
		if a_digits.len() < 9 {
			// small enough to use i32 integer division
			let (q, r) = BigInt::simple_division_32bit(&a_digits, &b_digits);
			return Ok((BigInt::from_i32(q).convert_sign(positive), BigInt::from_i32(r)));
		}
		// time for long division, oh boy!
		// (see Burnikel and Ziegler's 1998 paper)
		//    ____
		//  a) b
		todo!()
	}

	fn is_usize_odd(u: &usize) -> bool {
		return u & 0x01 == 1;
	}

	fn burnikel_ziegler_division_3(number: &Vec<u8>, divisor: &Vec<u8>) -> (Vec<u8>, Vec<u8>) {

		todo!() // return quotient and remainder
	}

	fn recursive_division(a_digits: &[u8], b_digits: &[u8], n: usize) -> (Vec<u8>, Vec<u8>){
		const DIV_LIMIT: usize = 4;
		if n < DIV_LIMIT { // || n.is_odd() ?
			let (q32, r32) = BigInt::simple_division_32bit(&a_digits.to_vec(), &b_digits.to_vec());
			let q = BigInt::i32_to_vec_u8(q32);
			let r = BigInt::i32_to_vec_u8(r32);
			return (q, r);
		} else {/*
			let (b1, b2) = BigInt::slice2(b_digits);
			let (a12, a34) = BigInt::slice2(a_digits);
			let (a1, a2) = BigInt::slice2(a12);
			let (a3, a4) = BigInt::slice2(a34);
			let (q1, r) = BigInt::div_three_long_halves_by_two(a1, a2, a3, b1, b2, n/2);
			let (r1, r2) = BigInt::slice2(r.as_slice());
			let (q2, s) = BigInt::div_three_long_halves_by_two(r1, r2, a4, b1, b2, n/2);
			let q = BigInt::merge2(q2, q1);
			return (q, s);*/
			todo!()
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
	fn i32_to_vec_u8(integer: i32) -> Vec<u8>{
		let mut i = integer;
		let mut v: Vec<u8> = Vec::with_capacity(10);
		while i > 0 {
			v.push((i % 10) as u8);
			i = i / 10;
		}
		return v;
	}
	fn vec_u8_to_i32(v: &Vec<u8>) -> i32 {
		let mut integer: i32 = 0;
		for i in (0..v.len() as usize).rev() {
			integer = integer * 10 + v[i] as i32;
		}
		return integer;
	}
	fn simple_division_32bit(a_digits: &Vec<u8>, b_digits: &Vec<u8>) -> (i32, i32) {
		assert!(a_digits.len() < 9 && b_digits.len() < 9);
		// small enough to use i32 integer division
		let n = BigInt::vec_u8_to_i32(a_digits);
		let d = BigInt::vec_u8_to_i32(b_digits);
		let q = n / d;
		let r = n % d;
		return (q, r);
	}
	fn concat_bigint(a1: &BigInt, a2: &BigInt) -> BigInt{
		return BigInt{digits: BigInt::merge2(&a1.digits, &a2.digits), positive:a1.positive};
	}
	fn div_three_long_halves_by_two(a1: &BigInt, a2: &BigInt, a3: &BigInt, b1: &BigInt, b2: &BigInt, n: usize)
		-> (BigInt, BigInt) {
		// let a12 = concat_bigint(a1, a2);
		// let (mut q, mut c) = recursive_division(a12, n);
		// let d = q*(b2 as i16);
		// let mut r = (c*10 + (a3 as i16)) - d;
		// while r < 0 { // negative r means q too big
		// 	q -= 1;
		// 	r += b;
		// }
		todo!()
	}
	fn div_two_wholes_by_one(a1: u8, a2: u8, a3: u8, a4: u8, b1: u8, b2: u8) -> (u8, u8) {
		let ah = (a1 * 10 + a2);
		let al = (a3 * 10 + a4);
		let bb = (b1 * 10 + b2);
		let (q1, r) = BigInt::div_three_halfs_by_two(a1, a2, a3, b1, b2);
		let r1 = r/10; let r2 = r % 10;
		let (q2, s) = BigInt::div_three_halfs_by_two(r1, r2, a4, b1, b2);
		let q = (q1 * 10 + q2);
		return (q, s);

	}
	fn div_three_halfs_by_two(a1: u8, a2: u8, a3: u8, b1: u8, b2: u8) -> (u8, u8){
		let b = (b1*10 + b2) as i16;
		let a12 = (a1*10 + a2) as i16;
		let mut q = a12/(b1 as i16);
		let c = a12 - q*(b1 as i16);
		let d = q*(b2 as i16);
		let mut r = (c*10 + (a3 as i16)) - d;
		while r < 0 { // negative r means q too big
			q -= 1;
			r += b;
		}
		return (q as u8, r as u8);
	}
}

impl Div for BigInt {
	type Output = Self;
	fn div(self, rhs: Self) -> Self::Output {
		if rhs.is_zero() {
			panic!("attempt to divide by zero");
		}
		return self.checked_div_rem(&rhs).unwrap().0;
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
		todo!()
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
		todo!()
	}
}
impl Add for &BigInt {
	type Output = BigInt;
	fn add(self, rhs: Self) -> BigInt {
		todo!()
	}
}
impl AddAssign for &BigInt {
	fn add_assign(&mut self, rhs: Self) {
		todo!()
	}
}
impl Sub for &BigInt {
	type Output = BigInt;
	fn sub(self, rhs: Self) -> BigInt {
		todo!()
	}
}
impl SubAssign for &BigInt {
	fn sub_assign(&mut self, rhs: Self) {
		todo!()
	}
}
impl Mul for &BigInt {
	type Output = BigInt;
	fn mul(self, rhs: Self) -> BigInt {
		todo!()
	}
}
impl MulAssign for &BigInt {
	fn mul_assign(&mut self, rhs: Self) {
		todo!()
	}
}
impl Div for &BigInt {
	type Output = BigInt;
	fn div(self, rhs: Self) -> BigInt {
		todo!()
	}
}
impl DivAssign for &BigInt {
	fn div_assign(&mut self, rhs: Self) {
		todo!()
	}
}
impl Rem for &BigInt {
	type Output = BigInt;
	fn rem(self, rhs: Self) -> BigInt {
		todo!()
	}
}
impl RemAssign for &BigInt {
	fn rem_assign(&mut self, rhs: Self) {
		todo!()
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
		todo!()
	}
}
impl Mul<i32> for BigInt {
	type Output = Self;
	fn mul(self, rhs: i32) -> Self {
		todo!()
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
		for n in 0..10000 {
			for d in 1..100{
				let qq: i32 = n / d;
				let rr: i32 = n % d;
				let a1 = n / 1000;      let a2 = (n / 100) % 10;
				let a3 = (n / 10) % 10; let a4 = n % 10;
				let b1 = d / 10; let b2 = d % 10;
				let (q, r) = BigInt::div_two_wholes_by_one(
					a1 as u8, a2 as u8, a3 as u8, a4 as u8,
					b1 as u8, b1 as u8
				);
				let qqp = (n / 10) / d;
				let rrp = (n / 10) % d;
				let (qp, rp) = BigInt::div_three_halfs_by_two(
					a1 as u8, a2 as u8, a3 as u8,
					b1 as u8, b1 as u8
				);

				assert_eq!(qp as i32, qqp,"{}/{} does not equal div_three_halfs_by_two({}, {}, {}, \
						   {}, {}).0", n/10, d, a1 as u8, a2 as u8, a3 as u8,
								   b1 as u8, b1 as u8);
				assert_eq!(rp as i32, rrp,"{}%{} does not equal div_three_halfs_by_two({}, {}, {}, \
						   {}, {}).1", n/10, d, a1 as u8, a2 as u8, a3 as u8,
								   b1 as u8, b1 as u8);
				assert_eq!(q as i32, qq,"{}/{} does not equal div_two_wholes_by_one({}, {}, {}, {}, \
						   {}, {}).0", n, d, a1 as u8, a2 as u8, a3 as u8, a4 as u8,
								   b1 as u8, b1 as u8);
				assert_eq!(r as i32, rr,"{}%{} does not equal div_two_wholes_by_one({}, {}, {}, {}, \
						   {}, {}).1", n, d, a1 as u8, a2 as u8, a3 as u8, a4 as u8,
								   b1 as u8, b1 as u8);
			}
		}
	}

	#[test]
	fn bigint_operator_test() {
		let i1 = BigInt::from_str("1234567890987654321").unwrap();
		let i2 = BigInt::from_str("0123456780876543210").unwrap();
		assert!(i1 > i2);
		assert!(i2 < i1);
		assert_ne!(i1, i2);
		assert_ne!(i1, BigInt::from_str("-1234567890987654321").unwrap()); // sign check
		assert_eq!(i1.clone() + i2.clone(), BigInt::from_str("1358024671864197531").unwrap());
		assert_eq!(&i1 + &i2, BigInt::from_str("1358024671864197531").unwrap());
		assert_eq!(i1.clone() - i2.clone(), BigInt::from_str("1111111110111111111").unwrap());
		assert_eq!(&i1 - &i2, BigInt::from_str("1111111110111111111").unwrap());
		assert_eq!(i1.clone() * i2.clone(), BigInt::from_str("152415777594878924352994968899710410").unwrap());
		assert_eq!(&i1 * &i2, BigInt::from_str("152415777594878924352994968899710410").unwrap());
		assert_eq!(i1.clone() / i2.clone(), BigInt::from_str("10").unwrap());
		assert_eq!(&i1 / &i2, BigInt::from_str("10").unwrap()); // remember, it's integer division
		assert_eq!(i1.clone() % i2.clone(), BigInt::from_str("82222222221").unwrap());
		assert_eq!(&i1 % &i2, BigInt::from_str("82222222221").unwrap());
		assert_eq!((i1.clone()/BigInt::from_i32(101)).to_string(), "12223444465224300");
		assert_eq!((i1.clone() % BigInt::from_i32(101)).to_string(), "21");
		assert_eq!(BigInt::from_i32(101).pow(11).to_string(),"11156683466653165551101");
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
}

