use std::{
    io::{stdin, BufRead, BufReader},
    ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg},
    f64::consts::PI,
    fmt::{Display, Formatter}
};

#[derive(Copy, Clone, Debug)]
struct Complex<T> {
    real_number: T,
    imaginary_number: T
}

impl<T> Complex<T> 
where 
    T: Neg<Output = T>
{
    fn new(real_number: T, imaginary_number: T) -> Self {
        Complex { real_number, imaginary_number }
    }
    
    fn conj(self) -> Self {
        Self::new(self.real_number, -self.imaginary_number)
    }
}

impl Add for Complex<f64> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.real_number + rhs.real_number,
            self.imaginary_number + rhs.imaginary_number
        )
    }
}

impl Sub for Complex<f64> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.real_number - rhs.real_number,
            self.imaginary_number - rhs.imaginary_number
        )
    }
}

impl Mul for Complex<f64> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.real_number * rhs.real_number - self.imaginary_number * rhs.imaginary_number,
            self.real_number * rhs.imaginary_number + self.imaginary_number * rhs.real_number
        )
    }
}

impl Div for Complex<f64> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let c_pow_plus_d_pow = rhs.real_number * rhs.real_number + rhs.imaginary_number * rhs.imaginary_number;
        Self::new(
            (self.real_number * rhs.real_number + self.imaginary_number * rhs.imaginary_number) / c_pow_plus_d_pow,
            (self.imaginary_number * rhs.real_number - self.real_number * rhs.imaginary_number) / c_pow_plus_d_pow
        )
    }
}

impl AddAssign for Complex<f64> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Complex<f64> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Complex<f64> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}


impl DivAssign for Complex<f64> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Display for Complex<f64> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ re: {}, im: {} }}", self.real_number, self.imaginary_number)
    }
}

impl Default for Complex<f64> {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

fn main() {
    let split = BufReader::new(stdin().lock()).lines().map(|a| a.unwrap()).next().unwrap().split_whitespace().map(|a| a.chars().rev().map(|b| u32_to_complex64(b.to_digit(10).unwrap())).collect::<Vec<_>>()).collect::<Vec<_>>();
    let (mut num1, mut num2) = (split[0].clone(), split[1].clone());

    let mut n = 1;
    while n < num1.len() + num2.len() {
        n <<= 1;
    }
    num1.resize(n, Complex::default());
    num2.resize(n, Complex::default());
    let log_n = n.ilog2();
    
    println!("{}", log_n);

    let result;
    //if log_n % 2 == 0 {
        println!("True");
        let num1_fft = radix4_dit_fft(num1, false);
        let num2_fft = radix4_dit_fft(num2, false);
        let convolution = convolution(num1_fft, num2_fft);
        convolution.iter().for_each(|num| println!("{}", num));
        let ifft = radix4_dit_fft(convolution, true);
        result = ifft;
    //} else {
        println!("False");/*
        let num1_fft = radix2_dit_fft_for_radix4_dit_fft(num1);
        let num2_fft = radix2_dit_fft_for_radix4_dit_fft(num2);
        let convolution = convolution(num1_fft, num2_fft);
        let ifft = radix2_dit_fft_for_radix4_ifft(convolution);
        result = ifft;*/
    //}

    result.iter().for_each(|num| println!("{}", num));

    let mut output = Vec::new();
    let mut carry = 0;

    for complex in result {
        let digit = complex.real_number.round() as i32 + carry;
        output.push((digit % 10) as u32);
        carry = digit / 10;
    }
    
    output.iter().for_each(|num| println!("{}", num));

    while let Some(&last) = output.last() {
        if last == 0 {
            output.pop();
        } else {
            break;
        }
    }

    if output.is_empty() {
        println!("0");
    } else {
        let print: String = output.iter().rev().map(|&d| (d as u8 + b'0') as char).collect();
        println!("{}", print);
    }
}

fn u32_to_complex64(input: u32) -> Complex<f64> {
    Complex::new(input as f64, 0.0)
}

fn convolution(signal1: Vec<Complex<f64>>, signal2: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    signal1.into_iter().zip(signal2).map(|(a, b)| a * b).collect()
}

fn radix2_split(signal: Vec<Complex<f64>>) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let n = signal.len();
    let (mut even, mut odd) = (Vec::with_capacity(n / 2), Vec::with_capacity(n / 2));
    for i in 0..n {
        if i % 2 == 0 {
            even.push(signal[i]);
        } else {
            odd.push(signal[i]);
        }
    }
    (even, odd)
}
/*
fn radix2_dit_fft_for_radix4_dit_fft(signal: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let signal_len = signal.len();

    let (even, odd) = radix2_split(signal);

    let even_fft = radix4_dit_fft(even);
    let odd_fft = radix4_dit_fft(odd);

    let wkn = |k| {
        let angle = 2.0 * PI * k as f64 / signal_len as f64;
        Complex::new(angle.cos(), -angle.sin())
    };

    let mut result = vec![Complex::default(); signal_len];
    for k in 0..signal_len / 2 {
        let twiddle = wkn(k);
        result[k] = even_fft[k] + twiddle * odd_fft[k];
        result[k + signal_len / 2] = even_fft[k] - twiddle * odd_fft[k];
    }

    result
}
*/
fn radix4_split(signal: Vec<Complex<f64>>) -> (Vec<Complex<f64>>, Vec<Complex<f64>>, Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let quarter_size = signal.len() / 4;
    let mut parts = (
        Vec::with_capacity(quarter_size),
        Vec::with_capacity(quarter_size),
        Vec::with_capacity(quarter_size),
        Vec::with_capacity(quarter_size)
    );
    
    for chunk in signal.chunks(4) {
        parts.0.push(chunk[0]);
        parts.1.push(chunk[1]);
        parts.2.push(chunk[2]);
        parts.3.push(chunk[3]);
    }
    parts
}

fn radix4_dit_fft(signal: Vec<Complex<f64>>, inverse: bool) -> Vec<Complex<f64>> {
    let signal_len = signal.len();

    if signal_len == 1 {
        return signal;
    }

    let (part1, part2, part3, part4) = radix4_split(signal);

    let part1_fft = radix4_dit_fft(part1, inverse);
    let part2_fft = radix4_dit_fft(part2, inverse);
    let part3_fft = radix4_dit_fft(part3, inverse);
    let part4_fft = radix4_dit_fft(part4, inverse);

    let angle_cache = 2.0 * PI / signal_len as f64 * if inverse { 1.0 } else { -1.0 };

    let wkn = |k| {
        let angle = angle_cache * k as f64;
        Complex::new(angle.cos(), angle.sin())
    };

    let quarter = signal_len / 4;
    let mut result = vec![Complex::default(); signal_len];
    for k in 0..quarter {
        let twiddle1 = wkn(k);
        let twiddle2 = wkn(2 * k);
        let twiddle3 = wkn(3 * k);

        let part1_cache = part1_fft[k];
        let part2_cache = part2_fft[k] * twiddle1;
        let part3_cache = part3_fft[k] * twiddle2;
        let part4_cache = part4_fft[k] * twiddle3;

        let cache1 = part1_cache + part3_cache;
        let cache2 = part1_cache - part3_cache;
        let cache3 = part2_cache + part4_cache;
        let cache4 = part2_cache - part4_cache;
        
        let i = Complex::new(0.0, 1.0);
        result[k] = cache1 + cache3;
        result[k + quarter] = cache2 - i * cache4;
        result[k + 2 * quarter] = cache1 - cache3;
        result[k + 3 * quarter] = cache2 + i * cache4;
        
    }
    
    if inverse { result.iter_mut().for_each(|x| *x /= Complex::new(signal_len as f64, 0.0)) }

    result
}
/*
fn radix4_ifft(signal: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let signal_len = signal.len();
    let fft_result = radix4_dit_fft(signal);
    fft_result.iter().map(|a| a.conj() / Complex::new(signal_len as f64, 0.0)).collect()
}
*/
/*
fn radix2_dit_fft_for_radix4_ifft(signal: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let signal_len = signal.len();
    let fft_result = radix2_dit_fft_for_radix4_dit_fft(signal);
    fft_result.iter().map(|a| a.conj() / Complex::new(signal_len as f64, 0.0)).collect()
}*/