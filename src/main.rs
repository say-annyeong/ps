use std::io::{stdin, BufRead};

const W: i64 = 3;
const MOD_P: i64 = 998244353;


fn pow_mod(a: i64, b: i64, m: i64) -> i64{
    let (mut a, mut b, mut r) = (a, b, 1);

    while b > 0 {
        if b & 1 == 1 { r = r * a % m };
        a = a * a % m;
        b >>= 1;
    }

    r
}

fn inv_pow_mod(a: i64, m: i64) -> i64 {
    pow_mod(a, m - 2, m)
}

fn convolution(v1: Vec<i64>, v2: Vec<i64>) -> Vec<i64> {
    let (mut v1, mut v2) = (v1, v2);
    let mut s = 2;
    while s < v1.len() + v2.len() {
        s <<= 1;
    }

    v1.resize(s, 0);
    v2.resize(s, 0);

    let mut v1 = fft(v1, false);
    let v2 = fft(v2, false);

    for i in 0..s {
        v1[i] = (v1[i] * v2[i]) % MOD_P;
    }

    let v1 = fft(v1, true);

    v1
}

fn fft(v: Vec<i64>, inv: bool) -> Vec<i64> {
    let n = v.len();
    let mut v = v;

    let mut j = 0;
    for i in 1..n {
        let mut b = n >> 1;
        while (j ^ b) & b == 0 {
            j ^= b;
            b >>= 1;
        }
        j ^= b;
        if i < j {
            v.swap(i, j);
        }
    }

    let mut x = pow_mod(W, (MOD_P - 1) / n as i64, MOD_P);
    if inv {
        x = inv_pow_mod(x, MOD_P);
    }

    let mut root = vec![1; n >> 1];
    for i in 1..(n >> 1) {
        root[i] = (root[i - 1] * x) % MOD_P;
    }

    let mut i = 2;
    while i <= n {
        let c = n / i;
        for j in (0..n).step_by(i) {
            for k in 0..(i >> 1) {
                let a = v[j | k];
                let b = (v[j | k | (i >> 1)] * root[c * k]) % MOD_P;

                v[j | k] = (a + b) % MOD_P;
                v[j | k | (i >> 1)] = (a - b).rem_euclid(MOD_P);
            }
        }
        i <<= 1;
    }

    if inv {
        let t = inv_pow_mod(n as i64, MOD_P);
        for x in v.iter_mut() {
            *x = (*x * t) % MOD_P;
        }
    }

    v
}

fn carry(input: Vec<i64>) -> String {
    let mut input = input;
    while let Some(&last) = input.last() {
        if last == 0 {
            input.pop();
        } else {
            break;
        }
    }
    let mut output = Vec::new();
    let mut carry = 0;
    for num in input {
        let digit = num + carry;
        output.push((digit % 10) as u32);
        carry = digit / 10;
    }
    if output.is_empty() {
        "0".to_string()
    } else {
        output.iter().rev().map(|&d| (d as u8 + b'0') as char).collect()
    }
}

fn main() {
    let mut reader = stdin().lock().lines().map(|line| line.unwrap());
    let binding = reader.next().unwrap();
    let mut neg = false;
    let mut buffer= binding.split_ascii_whitespace().map(|s| s.chars().map(|c| c.to_digit(10).unwrap_or({
        neg = !neg;
        0
    }) as i64).rev().collect());
    let (input1, input2): (Vec<_>, Vec<_>) = (buffer.next().unwrap(), buffer.next().unwrap());
    let conv = convolution(input1, input2);
    let output = carry(conv);
    println!("{}", output);
}