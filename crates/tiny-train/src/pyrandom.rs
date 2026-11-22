//! A bit-exact port of Python 3.11's `random.Random`: the MT19937 core plus
//! the helper methods the question generator calls, so a seed gives the same
//! stream of choices as the original Python kit.

/// Size of the Mersenne Twister state in 32-bit words.
const N: usize = 624;

/// Offset of the word mixed into each twisted word.
const M: usize = 397;

/// Python's `random.Random`, seeded from an integer.
#[derive(Clone)]
pub struct PyRandom {
    /// The MT19937 state words.
    mt: [u32; N],
    /// Index of the next state word to temper; `N` means "twist first".
    index: usize,
}

impl PyRandom {
    /// Seeds like `random.Random(seed)`: `init_by_array` over the 32-bit
    /// little-endian chunks of the seed (one zero chunk for seed 0).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let lo = (seed & 0xffff_ffff) as u32;
        let hi = (seed >> 32) as u32;
        let key: &[u32] = if hi == 0 { &[lo] } else { &[lo, hi] };
        let mut rng = Self::init_genrand(19_650_218);
        rng.init_by_array(key);
        rng
    }

    /// MT19937's `init_genrand`, the base state `init_by_array` mixes into.
    fn init_genrand(seed: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = seed;
        for (i, step) in (1..N).zip(1u32..) {
            let prev = mt[i - 1];
            mt[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(step);
        }

        Self { mt, index: N }
    }

    /// MT19937's `init_by_array`, exactly as `_randommodule.c` does it.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "indices are below N = 624, as in the C original"
    )]
    fn init_by_array(&mut self, key: &[u32]) {
        let mt = &mut self.mt;
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let prev = mt[i - 1];
            mt[i] = (mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }

            if j >= key.len() {
                j = 0;
            }
        }

        for _ in 0..N - 1 {
            let prev = mt[i - 1];
            mt[i] =
                (mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }

        mt[0] = 0x8000_0000;
    }

    /// Regenerates all `N` state words (the "twist").
    fn twist(&mut self) {
        for k in 0..N {
            let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % N] & 0x7fff_ffff);
            let mag = if y & 1 == 0 { 0 } else { 0x9908_b0df };
            self.mt[k] = self.mt[(k + M) % N] ^ (y >> 1) ^ mag;
        }

        self.index = 0;
    }

    /// The next raw 32-bit output (`genrand_uint32`).
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }

        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random()`: a float in `[0, 1)` with 53 random bits from two outputs.
    pub fn random(&mut self) -> f64 {
        let a = f64::from(self.next_u32() >> 5);
        let b = f64::from(self.next_u32() >> 6);
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `getrandbits(k)` for `1 <= k <= 64`; wider values fill from the low
    /// word up, as Python does.
    ///
    /// # Panics
    /// If `k` is outside `1..=64`.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!((1..=64).contains(&k), "getrandbits({k}) is not supported");
        if k <= 32 {
            return u64::from(self.next_u32() >> (32 - k));
        }

        let lo = u64::from(self.next_u32());
        let hi = u64::from(self.next_u32() >> (64 - k));
        lo | (hi << 32)
    }

    /// `_randbelow(n)`: rejection sampling over `n.bit_length()` bits.
    ///
    /// # Panics
    /// If `n` is zero (Python raises there too).
    pub fn randbelow(&mut self, n: usize) -> usize {
        assert!(n > 0, "randbelow(0)");
        let n = n as u64;
        let k = 64 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }

        usize::try_from(r).expect("r < n fits usize")
    }

    /// `randrange(n)`: an integer in `[0, n)`.
    pub fn randrange(&mut self, n: usize) -> usize {
        self.randbelow(n)
    }

    /// `randint(a, b)`: an integer in `[a, b]`, both ends included.
    ///
    /// # Panics
    /// If `b < a`.
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        let width = usize::try_from(b - a + 1).expect("randint range");
        a + i64::try_from(self.randbelow(width)).expect("width fits i64")
    }

    /// `choice(seq)`: one element, uniformly.
    ///
    /// # Panics
    /// If `seq` is empty.
    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> &'a T {
        &seq[self.randbelow(seq.len())]
    }

    /// `shuffle(x)`: Fisher-Yates from the end.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i + 1);
            x.swap(i, j);
        }
    }

    /// `sample(population, k)`: `k` distinct elements, with the C module's switch
    /// between the pool and the set algorithm.
    ///
    /// # Panics
    /// If `k` exceeds the population size.
    pub fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
        let n = population.len();
        assert!(k <= n, "sample larger than population");
        let mut setsize = 21usize;
        if k > 5 {
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "mirrors `4 ** ceil(log(k * 3, 4))` on small k"
            )]
            let exp = (((k * 3) as f64).ln() / 4f64.ln()).ceil() as u32;
            setsize += 4usize.pow(exp);
        }

        if n <= setsize {
            let mut pool = population.to_vec();
            return (0..k)
                .map(|i| {
                    let j = self.randbelow(n - i);
                    let picked = pool[j].clone();
                    pool[j] = pool[n - i - 1].clone();
                    picked
                })
                .collect();
        }

        let mut selected = std::collections::HashSet::new();
        (0..k)
            .map(|_| {
                let mut j = self.randbelow(n);
                while !selected.insert(j) {
                    j = self.randbelow(n);
                }

                population[j].clone()
            })
            .collect()
    }

    /// `choices(population, weights)[0]`, returned as an index: bisect of
    /// `random() * total` into the running sums of the weights.
    ///
    /// # Panics
    /// If `weights` is empty.
    pub fn choices_index(&mut self, weights: &[f64]) -> usize {
        let cum: Vec<f64> = weights
            .iter()
            .scan(0.0, |acc, w| {
                *acc += w;
                Some(*acc)
            })
            .collect();
        let total = cum[cum.len() - 1] + 0.0;
        let x = self.random() * total;
        cum[..cum.len() - 1].partition_point(|&c| c <= x)
    }
}

#[cfg(test)]
mod tests {
    use super::PyRandom;

    /// Raw outputs match `random.Random(1).getrandbits(32)`.
    #[test]
    fn raw_outputs() {
        let mut r = PyRandom::new(1);
        let got: Vec<u64> = (0..3).map(|_| r.getrandbits(32)).collect();
        assert_eq!(got, [577_090_037, 2_444_712_010, 3_639_700_191]);
        assert_eq!(PyRandom::new(0).getrandbits(32), 3_626_764_237);
    }

    /// Seeds wider than 32 bits use two key words; 40-bit draws use two outputs.
    #[test]
    fn wide_seed_and_bits() {
        let mut r = PyRandom::new((1 << 40) + 5);
        assert_eq!(r.getrandbits(32), 2_166_296_868);
        assert_eq!(r.getrandbits(40), 294_277_936_956);
    }

    /// `random()` matches bit for bit.
    #[test]
    fn floats() {
        let mut r = PyRandom::new(1);
        let want = [
            0.134_364_244_112_401_22,
            0.847_433_736_937_232_7,
            0.763_774_618_976_614,
        ];
        for w in want {
            assert_eq!(r.random().to_bits(), f64::to_bits(w));
        }
    }

    /// `randint`, `randrange` and `choice` from `random.Random(123)`.
    #[test]
    fn integers() {
        let mut r = PyRandom::new(123);
        let got: Vec<i64> = (0..5).map(|_| r.randint(0, 999)).collect();
        assert_eq!(got, [53, 274, 89, 787, 417]);
        assert_eq!(r.randrange(26), 8);
        assert_eq!(*r.choice(&['a', 'b', 'c', 'd', 'e', 'f', 'g']), 'a');
    }

    /// `shuffle` and both `sample` algorithms from `random.Random(7)`.
    #[test]
    fn shuffle_and_sample() {
        let mut r = PyRandom::new(7);
        let mut x: Vec<u32> = (0..10).collect();
        r.shuffle(&mut x);
        assert_eq!(x, [8, 3, 1, 4, 7, 0, 9, 6, 2, 5]);

        let mut r = PyRandom::new(7);
        let big: Vec<u32> = (0..100).collect();
        assert_eq!(r.sample(&big, 10), [41, 19, 50, 83, 6, 9, 68, 12, 46, 74]);
        let small: Vec<u32> = (0..30).collect();
        assert_eq!(r.sample(&small, 5), [1, 29, 16, 6, 2]);

        let mut r = PyRandom::new(7);
        let pool: Vec<u32> = (0..21).collect();
        assert_eq!(r.sample(&pool, 2), [10, 4]);
        let pool: Vec<u32> = (0..231).collect();
        let want = [
            101, 166, 12, 18, 210, 137, 24, 93, 149, 14, 129, 54, 9, 22, 111, 107, 17, 61, 23, 141,
            108, 15, 144,
        ];
        assert_eq!(r.sample(&pool, 23), want);
    }

    /// Weighted `choices` from `random.Random(9)`.
    #[test]
    fn weighted() {
        let mut r = PyRandom::new(9);
        let w = [0.2, 0.5, 0.3];
        let got: Vec<usize> = (0..11).map(|_| r.choices_index(&w)).collect();
        assert_eq!(got, [1, 1, 0, 2, 0, 1, 2, 0, 1, 1, 0]);
    }
}
